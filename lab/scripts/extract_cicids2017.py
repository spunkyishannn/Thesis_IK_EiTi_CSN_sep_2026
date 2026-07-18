#!/usr/bin/env python3
"""
External Benchmark Re-Extraction Pipeline (CICIDS2017).

Re-extracts raw network captures from the CICIDS2017 evaluation dataset using the identical
CICFlowMeter release to eliminate temporal unit discrepancies in cross-dataset evaluations.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLOWS_DIR = REPO_ROOT / "data" / "flows"

MONITOR_CONTAINER = "ids_v2_monitor"
CICFLOWMETER_JAR = "/opt/CICFlowMeter/bin/CICFlowMeter"
CICIDS_PCAP_DIR_IN_CONTAINER = "/data/cicids2017/pcaps"
FLOWS_DIR_IN_CONTAINER = "/data/flows"
OUT_SUBDIR = "CICIDS2017_L6"

SLICE_SECONDS = 600  # see module docstring for why this differs from extract_and_label.py's 60s
CFM_TIMEOUT = 10800  # 3h hedge -- generous given these files are 2-3x the largest D0x file that needed slicing

DAY_FILES = {
    "Tuesday": "Tuesday-WorkingHours.pcap",
    "Wednesday": "Wednesday-workingHours.pcap",
    "Friday": "Friday-WorkingHours.pcap",
}

def extract_day(day: str) -> Path | None:
    if day not in DAY_FILES:
        raise ValueError(f"day must be one of {sorted(DAY_FILES)} (Monday excluded -- leakage guard; "
                          f"Thursday not present on disk)")
    pcap_name = DAY_FILES[day]
    out_dir = FLOWS_DIR / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_csv = out_dir / f"{pcap_name}_ISCX.csv"

    if expected_csv.exists():
        print(f"[extract] Already extracted: {expected_csv.name}")
        return expected_csv

    container_pcap = f"{CICIDS_PCAP_DIR_IN_CONTAINER}/{pcap_name}"
    container_out = f"{FLOWS_DIR_IN_CONTAINER}/{OUT_SUBDIR}"
    stem = Path(pcap_name).stem
    # FOUND 2026-08-18: using /tmp (the container's own writable layer,
    # backed by Docker Desktop's WSL2 VHDX on C:) for slicing staging filled
    # C: to 1.4GB free during Wednesday's extraction (13GB+ of sliced pcap
    # data sitting there for the whole run, only cleaned up on success) --
    # the exact same VHDX-doesn't-auto-shrink problem already diagnosed
    # earlier this session, just triggered by a much bigger file than
    # extract_and_label.py's /tmp usage ever produced for D01-D04. Fixed by
    # staging under the ALREADY bind-mounted /data/flows (S:-backed, 100GB+
    # free) instead -- same lab/docker-compose.yml mount extract_and_label.py
    # already relies on, just used for scratch space too, not only final
    # output. Removes this failure mode regardless of how well VHDX
    # compaction works.
    work = f"{FLOWS_DIR_IN_CONTAINER}/_scratch_cicids/{stem}"
    out_csv = f"{container_out}/{pcap_name}_ISCX.csv"

    # Same proven pattern as extract_and_label.py's sliced path: pre-emptive
    # pkill (kills any orphaned JVM from a prior timed-out attempt), editcap
    # slice, per-slice CICFlowMeter run with SLICE_START/SLICE_DONE progress
    # markers, then concatenate (first header kept, later headers stripped).
    shell_cmd = f'''set -e
(command -v pkill >/dev/null 2>&1 && pkill -9 java 2>/dev/null) || true
rm -rf "{work}"; mkdir -p "{work}/slices" "{container_out}"
echo "SLICING start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
editcap -i {SLICE_SECONDS} "{container_pcap}" "{work}/slices/s.pcap"
n=0
total=$(ls "{work}"/slices/*.pcap 2>/dev/null | wc -l)
echo "SLICE_PLAN total=$total window={SLICE_SECONDS}s start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for f in "{work}"/slices/*.pcap; do
  sin="{work}/in_$n"; sout="{work}/out_$n"
  rm -rf "$sin" "$sout"; mkdir -p "$sin" "$sout"
  ln -s "$f" "$sin/$(basename "$f")"
  echo "SLICE_START $n/$total $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  {CICFLOWMETER_JAR} "$sin/" "$sout/"
  echo "SLICE_DONE $n/$total $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  n=$((n+1))
done
first=1
: > "{work}/combined.csv"
for c in "{work}"/out_*/*.csv; do
  [ -f "$c" ] || continue
  if [ "$first" = 1 ]; then head -n 1 "$c" > "{work}/combined.csv"; first=0; fi
  tail -n +2 "$c" >> "{work}/combined.csv"
done
if [ "$first" = 1 ]; then echo "ERROR: no per-slice CSVs produced" >&2; exit 4; fi
cp "{work}/combined.csv" "{out_csv}"
rm -rf "{work}"
echo "SLICED_OK slices=$n -> {out_csv}"
'''
    print(f"[extract] {day} ({pcap_name}) -> {out_dir}/  (sliced, {SLICE_SECONDS}s windows, timeout={CFM_TIMEOUT}s)",
          flush=True)
    t0 = time.time()
    try:
        result = subprocess.run(
            ["docker", "exec", MONITOR_CONTAINER, "sh", "-c", shell_cmd],
            capture_output=True, text=True, timeout=CFM_TIMEOUT
        )
    except subprocess.TimeoutExpired as e:
        crash_log = out_dir / f"{stem}_CRASH.log"
        crash_log.write_text(
            f"TIMED OUT after {e.timeout}s\n--- stdout ---\n{e.stdout or ''}\n--- stderr ---\n{e.stderr or ''}",
            encoding="utf-8", errors="replace")
        print(f"[ERROR] {day}: CICFlowMeter TIMED OUT after {e.timeout}s -- see {crash_log}", file=sys.stderr)
        subprocess.run(["docker", "exec", MONITOR_CONTAINER, "sh", "-c",
                         "(pkill -9 java 2>/dev/null) || true"], capture_output=True, timeout=30)
        return None

    elapsed = time.time() - t0
    print(result.stdout)
    if result.returncode != 0:
        crash_log = out_dir / f"{stem}_CRASH.log"
        crash_log.write_text(f"EXIT {result.returncode}\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}",
                              encoding="utf-8", errors="replace")
        print(f"[ERROR] {day}: CICFlowMeter exited {result.returncode} after {elapsed:.0f}s -- see {crash_log}",
              file=sys.stderr)
        return None

    if not expected_csv.exists():
        print(f"[ERROR] {day}: expected {expected_csv} not found after successful exit", file=sys.stderr)
        return None
    size_mb = expected_csv.stat().st_size / (1024 * 1024)
    print(f"[extract] {day} DONE in {elapsed:.0f}s -> {expected_csv.name} ({size_mb:.0f} MB)", flush=True)
    return expected_csv

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, choices=sorted(DAY_FILES))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    result = extract_day(args.day)
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    main()
