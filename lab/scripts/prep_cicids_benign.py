#!/usr/bin/env python3
"""
CICIDS2017 Benign Background Traffic Preprocessing.

Extracts and validates genuine benign background traffic from external benchmark captures
for comparative baseline analysis against synthetic lab background traffic.
"""

import subprocess
import sys
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# These paths work both on the host and inside the monitor container (via volume)
SCRIPT_DIR   = Path(__file__).resolve().parent
REPO_ROOT    = SCRIPT_DIR.parent.parent
PCAP_DIR     = REPO_ROOT / "data" / "raw" / "pcaps"
MONDAY_PCAP  = REPO_ROOT / "data" / "raw" / "cicids2017" / "pcaps" / "Monday-WorkingHours.pcap"
SLICED_PCAP  = PCAP_DIR / "monday_sliced.pcap"
TEMPLATE     = PCAP_DIR / "benign_template.pcap"

# When running inside monitor container, paths are remapped via Docker volume.
# The monitor mounts data/raw/pcaps → /data/pcaps, so adjust if needed:
if not MONDAY_PCAP.exists():
    # Try container path
    MONDAY_PCAP  = Path("/data/cicids2017/pcaps/Monday-WorkingHours.pcap")
    SLICED_PCAP  = Path("/data/pcaps/monday_sliced.pcap")
    TEMPLATE     = Path("/data/pcaps/benign_template.pcap")

# ── IP remapping ──────────────────────────────────────────────────────────────
# CICIDS2017 internal victim subnet → Docker lab subnet
# All CICIDS victim machines (192.168.10.x) map to our lab_net (172.20.x)
# External internet IPs are left as-is (they produce realistic external-facing flows)
PNAT = "192.168.10.0/24:172.20.0.0/24"

# ── Slice parameters ──────────────────────────────────────────────────────────
# Take the first 20 minutes of the Monday PCAP.
#
# WHY: tshark can stop reading after 1200s — much faster than slicing from the
# middle of the 11GB file. The 0–1200s window gives ~80% TCP / 20% UDP by
# packet count, which is representative of Monday's overall workday mix
# (full-day CSV: 71% TCP / 28% UDP). The slight morning skew is acceptable
# because the template is looped throughout the entire session, averaging out
# any startup transients over the replay cycles.
SLICE_START   = 0      # start from beginning of Monday PCAP
SLICE_SECONDS = 1200   # 20 minutes — fast (tshark stops early at 1200s)

def run(cmd: list, label: str = "") -> subprocess.CompletedProcess:
    tag = f"[{label}] " if label else ""
    print(f"  {tag}$ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  STDERR: {result.stderr[-400:]}", file=sys.stderr)
        raise RuntimeError(f"Command failed (rc={result.returncode}): {cmd[0]}")
    return result

def check_tools() -> None:
    missing = []
    for tool in ["tshark", "tcprewrite"]:
        r = subprocess.run(["which", tool], capture_output=True)
        if r.returncode != 0:
            missing.append(tool)
    if missing:
        print(f"[ERROR] Missing tools: {missing}")
        print("Install: sudo apt install tshark tcpreplay")
        print("Or run this script inside the monitor container:")
        print("  docker exec ids_v2_monitor python3 /scripts/prep_cicids_benign.py")
        sys.exit(1)

def pcap_info(path: Path) -> dict:
    """Execute internal routine."""
    r = subprocess.run(
        ["tshark", "-r", str(path), "-q", "-z", "io,stat,0"],
        capture_output=True, text=True
    )
    packets = 0
    for line in r.stdout.splitlines():
        if "Frames:" in line:
            try:
                packets = int(line.split()[-1])
            except ValueError:
                pass
    return {"packets": packets, "size_mb": path.stat().st_size / 1e6}

def main() -> None:
    print("V2 Benign Template Preparation")
    print("="*50)
    print(f"Source : {MONDAY_PCAP}")
    print(f"Output : {TEMPLATE}")
    print(f"Slice  : {SLICE_START}–{SLICE_START+SLICE_SECONDS}s of Monday capture (first 20 min, fast tshark stop)")
    print(f"IP map : {PNAT}")
    print()

    # Pre-checks
    if not MONDAY_PCAP.exists():
        print(f"[ERROR] Monday PCAP not found: {MONDAY_PCAP}")
        print("Download Monday-WorkingHours.pcap from:")
        print("  https://www.unb.ca/cic/datasets/ids-2017.html")
        print("and place it in data/raw/cicids2017/pcaps/")
        sys.exit(1)

    check_tools()
    PCAP_DIR.mkdir(parents=True, exist_ok=True)

    if TEMPLATE.exists() and TEMPLATE.stat().st_size > 1024:
        info = pcap_info(TEMPLATE)
        print(f"[OK] Template already exists: {info['size_mb']:.0f} MB, ~{info['packets']:,} packets")
        print("Delete it and re-run to regenerate.")
        return

    # ── Step 1: Slice mid-morning window from Monday PCAP ─────────────────────
    SLICE_END = SLICE_START + SLICE_SECONDS
    # Check size > 1024 to handle the WSL2 bind-mount stub:
    # The caller must pre-create monday_sliced.pcap with New-Item (0 bytes)
    # so tshark can overwrite it. A 0-byte file is not a valid slice.
    if not SLICED_PCAP.exists() or SLICED_PCAP.stat().st_size < 1024:
        print(f"[1/2] Slicing {SLICE_START}–{SLICE_END}s window from Monday PCAP …")
        print(f"      (first {SLICE_SECONDS}s window; tshark stops early — fast even on 11GB source)")
        run([
            "tshark",
            "-r", str(MONDAY_PCAP),
            "-Y", f"frame.time_relative >= {SLICE_START} and frame.time_relative < {SLICE_END}",
            "-w", str(SLICED_PCAP),
        ], label="tshark")
        info = pcap_info(SLICED_PCAP)
        print(f"      Sliced: {info['size_mb']:.0f} MB, ~{info['packets']:,} packets")
    else:
        info = pcap_info(SLICED_PCAP)
        print(f"[1/2] Using existing slice: {info['size_mb']:.0f} MB (~{info['packets']:,} packets)")

    # ── Step 2: Remap IPs to Docker subnet + fix checksums ────────────────────
    print(f"\n[2/2] Remapping {PNAT} and fixing checksums …")

    # Docker Desktop/WSL2 bind mounts block creating NEW files from inside the container.
    # tcprewrite (and touch) can only overwrite EXISTING files on Windows-hosted volumes.
    # If the template doesn't exist, abort with a clear message.
    if not TEMPLATE.exists():
        host_path = r"S:\ML IDS THESIS\data\raw\pcaps\benign_template.pcap"
        print(f"\n[ERROR] Cannot create new file through Docker volume mount (WSL2 bind mount restriction).")
        print(f"Create the file on Windows first, then re-run:")
        print(f"  PowerShell: New-Item \"{host_path}\" -ItemType File -Force")
        print(f"  Then:       docker exec ids_v2_monitor python3 /scripts/prep_cicids_benign.py")
        sys.exit(1)

    run([
        "tcprewrite",
        f"--pnat={PNAT}",
        f"--infile={SLICED_PCAP}",
        f"--outfile={TEMPLATE}",
        "--fixcsum",        # recompute TCP/UDP/IP checksums after IP rewrite
        "--mtu=1500",       # Docker bridge MTU
        "--mtu-trunc",      # truncate oversized packets instead of dropping (avoids errno=90 at replay)
    ], label="tcprewrite")

    info = pcap_info(TEMPLATE)
    print(f"\n[OK] Template ready: {TEMPLATE}")
    print(f"     Size: {info['size_mb']:.0f} MB | Packets: {info['packets']:,}")
    print(f"\n     tcpreplay will loop this {SLICE_SECONDS}s window indefinitely during")
    print(f"     each session, giving continuous realistic benign traffic.")
    print(f"\nNext steps:")
    print(f"  docker compose up -d --build")
    print(f"  python scripts/orchestrate.py --domain D01 --pilot")

if __name__ == "__main__":
    main()
