#!/usr/bin/env python3
"""
CICFlowMeter Flow Extraction and Ground-Truth Labelling Pipeline.

Executes CICFlowMeter over raw PCAP captures and matches bi-directional flow timestamps
against session manifests to assign unambiguous ground-truth attack labels.
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
PCAP_DIR     = REPO_ROOT / "data" / "raw" / "pcaps"
MANIFEST_DIR = REPO_ROOT / "lab" / "manifests"
FLOWS_DIR    = REPO_ROOT / "data" / "flows"
PROCESSED_DIR= REPO_ROOT / "data" / "processed"

# CICFlowMeter only exists INSIDE the monitor container (built from source in
# lab/docker/monitor/Dockerfile) — it is never installed on the Windows host.
# This script itself runs natively on Windows (invoked as `python
# scripts/extract_and_label.py`, not via docker exec), so every CICFlowMeter
# call must be dispatched into the container via `docker exec`, the same way
# orchestrate.py dispatches tcpdump/tcpreplay. Path below is the in-container
# path to the CLI wrapper script.
MONITOR_CONTAINER = "ids_v2_monitor"
CICFLOWMETER_JAR = "/opt/CICFlowMeter/bin/CICFlowMeter"

# Host directories below are bind-mounted into the monitor container at fixed
# paths (see lab/docker-compose.yml). We only ever pass CICFlowMeter the
# in-container path; pcap_path/out_dir on the Python/host side are used just
# to build that container path and to check the resulting CSV afterward.
PCAP_DIR_IN_CONTAINER  = "/data/pcaps"
FLOWS_DIR_IN_CONTAINER = "/data/flows"

# ── Oversized-PCAP slicing (added 2026-08-14) ────────────────────────────────
# DOS_HULK's high-thread HTTP floods produce multi-GB PCAPs whose in-memory flow
# table does not fit CICFlowMeter's JVM heap even at -Xmx6g: dh-1 (1.6GB, 500
# threads) OOM'd three separate times at three DIFFERENT allocation sites — the
# signature of genuine heap exhaustion, not a deterministic parse bug (see
# problems_notes.txt 2026-08-14). CICFlowMeter's peak heap tracks the number of
# flows held live in memory, which scales with the attack's CONCURRENCY (thread
# count), not raw file size — which is exactly why the SMALLER dh-1 (500 threads)
# OOM'd where the LARGER dh-2 (7.5GB, 250 threads) extracted cleanly. Rather than
# chase ever-larger -Xmx (fragile, host-RAM-dependent, and D02-D04 have their own
# large HULK sessions that would re-trigger it mid-run), any PCAP over the
# threshold below is time-sliced into short windows, each window is extracted
# separately, and the per-window CSVs are concatenated. This bounds peak heap
# regardless of file size or thread count, needs NO RAM change, and is
# label-safe: sliced flows keep their original packet timestamps and IPs so
# label_flows() matches them identically. Slice length is well under
# CICFlowMeter's own 120s flow-activity-timeout, and HULK flows are sub-second
# (connect -> GET -> close), so boundary-splitting of flows is negligible.
SLICE_THRESHOLD_BYTES = 1_200_000_000   # ~1.2 GB — only DOS_HULK exceeds this in D01
SLICE_SECONDS         = 60              # per-window length; << 120s CICFlowMeter flow timeout

# ── CICFlowMeter wrapper ─────────────────────────────────────────────────────

def _wait_for_stable_file(path: Path, checks: int = 3, interval: float = 1.0) -> bool:
    """Execute internal routine."""
    last_size = -1
    stable_count = 0
    for _ in range(checks + 5):  # a few extra polls in case it's still growing
        if not path.exists():
            time.sleep(interval)
            continue
        size = path.stat().st_size
        if size == last_size and size > 0:
            stable_count += 1
            if stable_count >= checks - 1:
                return True
        else:
            stable_count = 0
        last_size = size
        time.sleep(interval)
    return False

def extract_pcap(pcap_path: Path, out_dir: Path) -> Path | None:
    """Execute internal routine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # CICFlowMeter names its output after the FULL pcap filename (including the
    # ".pcap" extension), not the stem — confirmed 2026-08-12 from actual output
    # (e.g. "D01-BENIGN-benign-only-1-223344.pcap_ISCX.csv"). Using
    # pcap_path.stem here previously meant this "already extracted" check could
    # never match, so every re-run silently redid the full ~10min CICFlowMeter
    # pass even when a good CSV already existed.
    expected_csv = out_dir / (pcap_path.name + "_ISCX.csv")

    if expected_csv.exists():
        print(f"  [extract] Already extracted: {expected_csv.name}")
        return expected_csv

    # out_dir is always FLOWS_DIR / <domain> (see main()) — its final path
    # component is the domain name, which is what's mounted under
    # /data/flows inside the monitor container.
    container_pcap = f"{PCAP_DIR_IN_CONTAINER}/{pcap_path.name}"
    container_out  = f"{FLOWS_DIR_IN_CONTAINER}/{out_dir.name}"

    # FOUND 2026-08-15 (D03's dh-1, real full-generation run): the byte-size
    # threshold alone is not a reliable predictor of OOM risk for DOS_HULK.
    # D02's dh-3 (1007MB) extracted cleanly UN-sliced; D03's dh-1 (1053MB —
    # only ~46MB bigger, still under the 1.2GB threshold) OOM'd on the exact
    # same non-sliced path with the exact same signature as D01's original
    # dh-1 (java.lang.OutOfMemoryError: Java heap space in
    # BasicFlow.addPacket, confirmed by reading the real crash log, not
    # guessed). Heap pressure tracks concurrent-live-flow count, which for
    # DOS_HULK is driven by its per-session THREAD COUNT (varies by
    # domain/variant), not file size — so two DOS_HULK pcaps within ~5% of
    # each other in size can land on opposite sides of any byte threshold.
    # Slicing has a 100% success rate so far across every size it's been
    # tried at (D01 1.6GB, D02 5.8GB, D03 10.1GB and 3.1GB) with no observed
    # downside beyond modest extra wall-clock, so for DOS_HULK specifically
    # — the one family that has now OOM'd twice at two different sizes —
    # slice unconditionally instead of trying to guess a safe byte cutoff.
    # The byte-size threshold stays as the general backstop for every other
    # family, none of which has shown this failure mode even at comparable
    # or larger sizes (e.g. D01's multi-source ICMP_FLOOD at 948MB was fine
    # un-sliced). Family is parsed from the filename's own
    # <domain>-<FAMILY>-<variant>-<timestamp>.pcap convention, the same
    # convention merge_regenerated.py already relies on elsewhere.
    pcap_family = pcap_path.name.split("-")[1] if pcap_path.name.count("-") >= 1 else ""
    needs_slice = (pcap_path.stat().st_size > SLICE_THRESHOLD_BYTES) or (pcap_family == "DOS_HULK")

    # CICFlowMeter's CLI only accepts a DIRECTORY as its first argument — it
    # scans that directory for *.pcap files rather than accepting a single
    # file path. Passing a file path directly produces "Sorry, no pcap files
    # can be found under: <path>" (discovered 2026-08-12). To process exactly
    # one file we symlink just that pcap into a private staging directory
    # under /tmp (the container's own overlay filesystem, not bind-mounted —
    # sidesteps any WSL2 bind-mount concerns entirely), run CICFlowMeter
    # against that staging dir, then copy the resulting CSV out to the real
    # /data/flows location (a plain `cp`, which we've confirmed new-file
    # writes work fine on this particular mount).
    stem = pcap_path.stem
    stage_in  = f"/tmp/cfm_in/{stem}"
    stage_out = f"/tmp/cfm_out/{stem}"
    # FOUND 2026-08-15 (D03's dh-1, second attempt -- full diagnosis in the
    # TimeoutExpired handler below): killing subprocess.run's own Popen on a
    # timeout only kills the Windows-side `docker exec` CLIENT -- already
    # documented as NOT propagating into the container ("Windows
    # Popen.terminate() doesn't signal through docker exec", first hit on
    # D01's dh-2 hang). Every CICFlowMeter invocation now defensively kills
    # any already-running CICFlowMeter JVM inside the container FIRST, so a
    # process orphaned by an earlier timeout can't silently keep competing
    # for the container's CPU/heap with this new attempt. Matches on the
    # process NAME ("java"), not `-f` full-cmdline text -- an `-f CICFlowMeter`
    # pattern would also match THIS WRAPPING sh -c invocation's own argv
    # (which literally contains the word "CICFlowMeter" via this very
    # invocation text), and pkill does not exclude its own parent shell, only
    # itself -- that would have SIGKILLed the shell running this cleanup
    # command mid-script. Caught in review before shipping, not from a real
    # incident. No other persistent Java service is known to run in this
    # container (its only job is hosting CICFlowMeter for on-demand `docker
    # exec` calls) -- if that assumption ever changes, this needs to get more
    # specific. `|| true` throughout because pkill exits 1 when nothing
    # matches (the common case) -- must not abort the chain either here or
    # under the sliced path's `set -e` below.
    cleanup_prefix = "(command -v pkill >/dev/null 2>&1 && pkill -9 java 2>/dev/null) || true; "
    shell_cmd = (
        cleanup_prefix +
        f"rm -rf '{stage_in}' '{stage_out}' && "
        f"mkdir -p '{stage_in}' '{stage_out}' '{container_out}' && "
        f"ln -s '{container_pcap}' '{stage_in}/{pcap_path.name}' && "
        # NOTE (2026-08-12): CICFlowMeter's internal directory-scan appears to
        # join dirArg + fileName with no path separator (observed error:
        # "/tmp/cfm_in/<stem><stem>.pcap" — dir and filename concatenated with
        # no "/" between). Passing the input dir WITH a trailing slash makes
        # that naive concatenation land on a correct path either way.
        # Same likely bug applies to the output arg (CICFlowMeter wrote its
        # CSV to /tmp/cfm_out/<stem><realfilename>.csv — one level UP from
        # stage_out, name mashed together with no separator — instead of
        # inside stage_out/). Trailing slash here too, per the same fix that
        # worked for the input dir. If this guess is wrong, the fallback
        # `find` dumps where the file actually landed instead of guessing again.
        f"{CICFLOWMETER_JAR} '{stage_in}/' '{stage_out}/' && "
        f"(cp '{stage_out}'/*.csv '{container_out}'/ || "
        f"(echo '--- cp failed; /tmp/cfm_out contents: ---'; find /tmp/cfm_out -type f -o -type l; exit 1)) && "
        f"rm -rf '{stage_in}' '{stage_out}'"
    )

    # Oversized PCAP → swap the single-shot command above for a time-sliced pass
    # (see SLICE_* constants). editcap (ships with the tshark already in the
    # monitor image) splits the pcap into SLICE_SECONDS windows; each window is
    # run through CICFlowMeter in its own staging dir (same trailing-slash
    # dir-arg workaround as the normal path), then the per-window CSVs are
    # concatenated into the single CSV the rest of the pipeline expects — the
    # first window's header is kept, later headers stripped with `tail -n +2`.
    if needs_slice:
        work    = f"/tmp/cfm_slice/{stem}"
        out_csv = f"{container_out}/{pcap_path.name}_ISCX.csv"
        # Same pre-emptive pkill as the non-sliced path above (see its
        # comment for why it's name-based, not `-f`-based), plus per-slice
        # SLICE_START/SLICE_DONE timestamps (2026-08-15): if this ever times
        # out again, subprocess.run's TimeoutExpired carries whatever stdout
        # was already captured before the kill (confirmed from CPython's own
        # subprocess.run source -- on Windows it calls process.communicate()
        # again after kill() specifically to collect this), so the crash log
        # will show how many slices actually finished and how long each one
        # took, instead of a single opaque "timed out after Ns" with zero
        # insight into whether it was one slow slice or all of them.
        shell_cmd = f'''set -e
(command -v pkill >/dev/null 2>&1 && pkill -9 java 2>/dev/null) || true
if ! command -v editcap >/dev/null 2>&1; then
  echo "ERROR: editcap not found in monitor image (needed to slice oversized pcaps); install wireshark-common" >&2
  exit 3
fi
rm -rf "{work}"; mkdir -p "{work}/slices" "{container_out}"
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

    # FOUND 2026-08-15 (D03's dh-1, second attempt): sliced timeout bumped
    # 3600->5400s as a hedge, NOT the primary fix -- the primary fix is the
    # pre-emptive pkill above plus the post-timeout container-side cleanup
    # below. Evidence points more toward a leaked process than "genuinely
    # needs more time": a 10.1GB slice already succeeded well inside 3600s
    # this same run, while dh-1's 1.05GB (SMALLER than D01's dh-1, which ALSO
    # succeeded inside 3600s with an identical 500-thread/300s config) did
    # not. Keeping extra headroom anyway in case a long unattended
    # multi-domain run also just has genuinely higher host load later on than
    # a short isolated run -- cheap insurance, does not replace the leak fix.
    cfm_timeout = 5400 if needs_slice else 1800
    if needs_slice:
        size_mb = pcap_path.stat().st_size // (1024 * 1024)
        if pcap_path.stat().st_size > SLICE_THRESHOLD_BYTES:
            reason = f"OVERSIZED ({size_mb} MB)"
        else:
            reason = f"DOS_HULK ({size_mb} MB, under the size threshold but sliced anyway — see problems_notes.txt 2026-08-15)"
        print(f"  [extract] {reason} → slicing into {SLICE_SECONDS}s windows before CICFlowMeter "
              f"(bounds JVM heap; see problems_notes.txt 2026-08-14)")
    print(f"  [extract] CICFlowMeter (in {MONITOR_CONTAINER}): {pcap_path.name} → {out_dir}/")
    # FOUND 2026-08-14 (first real full-generation run, D01 DOS_HULK
    # dh-1/dh-2 -- diagnosed with the real captured crash log, not guessed):
    # both DOS_HULK sessions are by far the largest PCAPs in the whole D01
    # run (1.6GB and 7.5GB, from 250/500-thread HTTP floods). dh-1's crash
    # log showed a genuine `java.lang.OutOfMemoryError: Java heap space` in
    # CICFlowMeter's own flow-bookkeeping code, after it had already
    # processed many flows cleanly -- the CICFlowMeter wrapper script
    # (lab/docker/monitor/Dockerfile) never set -Xmx, so the JVM was running
    # on its own default heap sizing, nowhere near enough for a multi-GB
    # pcap. dh-2 (4.7x larger again) didn't even get that far: it just hung
    # until THIS timeout killed it -- consistent with the same low-heap JVM
    # thrashing on GC rather than making progress, and previously this raised
    # an UNCAUGHT subprocess.TimeoutExpired that crashed the entire
    # extract_and_label.py run, losing the rest of the domain's already-
    # in-progress extraction, not just this one PCAP. Two fixes: (1) the
    # Dockerfile wrapper now sets -Xmx6g (needs `docker compose build
    # monitor` to take effect). (2) The timeout below is raised from 600s to
    # 1800s (multi-GB pcaps legitimately take longer even with enough heap),
    # AND the subprocess call is now wrapped so a timeout is treated exactly
    # like any other single-PCAP extraction failure -- skip and continue,
    # not crash the whole script.
    try:
        result = subprocess.run(
            ["docker", "exec", MONITOR_CONTAINER, "sh", "-c", shell_cmd],
            capture_output=True, text=True, timeout=cfm_timeout
        )
    except subprocess.TimeoutExpired as e:
        # FIXED 2026-08-15 (D03's dh-1, second attempt -- real traceback):
        # text=True on the subprocess.run() call above means e.stdout/e.stderr
        # on a TimeoutExpired are ALREADY str (Python decodes them before
        # attaching to the exception when the Popen is in text mode) -- the
        # previous .decode("utf-8", ...) calls here crashed with
        # `AttributeError: 'str' object has no attribute 'decode'`, which took
        # down the ENTIRE script with an unhandled exception instead of
        # gracefully marking just this one session as failed and continuing
        # (the exact behavior this except block exists to provide). No crash
        # log was written and labelled.parquet was never touched that run.
        crash_log = out_dir / f"{pcap_path.stem}_CRASH.log"
        timeout_stdout = e.stdout if e.stdout else ""
        timeout_stderr = e.stderr if e.stderr else ""
        crash_log.write_text(
            f"TIMED OUT after {e.timeout}s\n--- stdout ---\n{timeout_stdout}\n--- stderr ---\n{timeout_stderr}",
            encoding="utf-8", errors="replace"
        )
        print(f"  [ERROR] CICFlowMeter TIMED OUT on {pcap_path.name} after {e.timeout}s — "
              f"full output saved to {crash_log}", file=sys.stderr)
        # FOUND 2026-08-15: also clean up the container side explicitly --
        # killing THIS process only kills the Windows docker-exec client (see
        # the cleanup_prefix comment above); without this, whatever was still
        # running inside {MONITOR_CONTAINER} keeps running orphaned, eating
        # CPU/heap for every subsequent extraction in the rest of this domain
        # and any later domain, until something notices. Best-effort: if this
        # itself fails, the pre-emptive pkill at the start of the NEXT
        # extract_pcap() call is the backstop.
        try:
            subprocess.run(
                ["docker", "exec", MONITOR_CONTAINER, "sh", "-c",
                 "(command -v pkill >/dev/null 2>&1 && pkill -9 java 2>/dev/null) || true"],
                capture_output=True, text=True, timeout=30
            )
            print(f"  [cleanup] Killed any CICFlowMeter process still running in "
                  f"{MONITOR_CONTAINER} after the timeout (see problems_notes.txt 2026-08-15)",
                  file=sys.stderr)
        except Exception as cleanup_err:
            print(f"  [WARNING] Post-timeout container cleanup itself failed ({cleanup_err}) -- "
                  f"a stray CICFlowMeter process may still be running in {MONITOR_CONTAINER}. "
                  f"The next extract_pcap() call will still try to kill it before starting "
                  f"(see cleanup_prefix above), but if failures continue, run "
                  f"`docker restart {MONITOR_CONTAINER}` by hand before retrying.",
                  file=sys.stderr)
        return None

    if result.returncode != 0:
        # Full output saved to a dedicated per-session crash log (previously
        # only the last 800 characters were ever printed, which for a real
        # Java exception loses the exception class/message at the top --
        # that truncation is exactly what made dh-1/dh-2 undiagnosable from
        # the first crash log alone and required this second re-run).
        crash_log = out_dir / f"{pcap_path.stem}_CRASH.log"
        crash_log.write_text(result.stdout + result.stderr, encoding="utf-8", errors="replace")
        print(f"  [ERROR] CICFlowMeter failed on {pcap_path.name} — full output saved to {crash_log}", file=sys.stderr)
        print((result.stdout + result.stderr)[-2000:], file=sys.stderr)
        return None

    found = None
    if expected_csv.exists():
        found = expected_csv
    else:
        # CICFlowMeter sometimes uses the pcap filename directly
        for f in out_dir.glob("*.csv"):
            if pcap_path.stem in f.stem:
                found = f
                break

    if found is None:
        print(f"  [WARNING] No CSV found after extraction of {pcap_path.name}", file=sys.stderr)
        return None

    if not _wait_for_stable_file(found):
        print(f"  [WARNING] {found.name} size never stabilized after extraction — "
              f"proceeding anyway, but a read race with the WSL2/Docker bind mount "
              f"is possible (see problems_notes.txt)", file=sys.stderr)
    print(f"  [extract] OK: {found.stat().st_size // 1024} KB")
    return found

# ── Manifest loading ──────────────────────────────────────────────────────────

def load_manifest(domain: str) -> list[dict]:
    manifest_file = MANIFEST_DIR / f"{domain}_manifest.jsonl"
    if not manifest_file.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_file}")
    records = []
    with open(manifest_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"[manifest] Loaded {len(records)} session records")
    return records

# ── Flow loading ─────────────────────────────────────────────────────────────

# CICIDS_RENAME was originally written against the OLD CICFlowMeter column
# convention (leading-space-prefixed, full words: " Source IP", " Total Fwd
# Packets"). Confirmed 2026-08-12 by dumping df.columns straight from a real
# D01 CSV: the CICFlowMeter build actually installed in the monitor image
# (pulled from ahlashkari/CICFlowMeter at Docker build time) uses a NEWER,
# different column schema — no leading spaces, abbreviated "Src"/"Dst"
# instead of "Source"/"Destination", singular "Packet" instead of plural
# "Packets", lowercase "packets" in "Total Bwd packets". Because the rename
# dict's keys never matched, src_ip/dst_ip/src_port/dst_port/fwd_pkts/
# bwd_pkts/fwd_bytes/bwd_bytes were silently left un-renamed — label_flows()
# then read `row.get("src_ip", "")`, always got the "" default (the real data
# was still sitting under "Src IP"), and IP matching could never succeed for
# a single row. That's what was collapsing every attack session's labels to
# a mix of UNKNOWN/BENIGN with zero flows ever getting the real attack_family
# label, even after the timestamp fix. Rebuilt this map directly from the
# actual current output columns instead of guessing at another version.
CICIDS_RENAME = {
    "Flow Duration":                "flow_duration",
    "Total Fwd Packet":             "fwd_pkts",
    "Total Bwd packets":            "bwd_pkts",
    "Total Length of Fwd Packet":   "fwd_bytes",
    "Total Length of Bwd Packet":   "bwd_bytes",
    "Flow Bytes/s":                 "flow_bytes_s",
    "Flow Packets/s":               "flow_pkts_s",
    "Fwd Packets/s":                "fwd_pkts_s",
    "Bwd Packets/s":                "bwd_pkts_s",
    "Fwd Packet Length Min":        "fwd_pkt_len_min",
    "Fwd Packet Length Max":        "fwd_pkt_len_max",
    "Fwd Packet Length Mean":       "fwd_pkt_len_mean",
    "Fwd Packet Length Std":        "fwd_pkt_len_std",
    "Bwd Packet Length Min":        "bwd_pkt_len_min",
    "Bwd Packet Length Max":        "bwd_pkt_len_max",
    "Bwd Packet Length Mean":       "bwd_pkt_len_mean",
    "Bwd Packet Length Std":        "bwd_pkt_len_std",
    "Fwd IAT Mean":                 "fwd_iat_mean",
    "Fwd IAT Std":                  "fwd_iat_std",
    "Fwd IAT Min":                  "fwd_iat_min",
    "Fwd IAT Max":                  "fwd_iat_max",
    "Bwd IAT Mean":                 "bwd_iat_mean",
    "Bwd IAT Std":                  "bwd_iat_std",
    "Bwd IAT Min":                  "bwd_iat_min",
    "Bwd IAT Max":                  "bwd_iat_max",
    "SYN Flag Count":               "fwd_syn_cnt",
    "FIN Flag Count":               "fwd_fin_cnt",
    "RST Flag Count":               "fwd_rst_cnt",
    "PSH Flag Count":               "fwd_psh_cnt",
    "ACK Flag Count":               "fwd_ack_cnt",
    "Timestamp":                    "timestamp",
    "Src IP":                       "src_ip",
    "Dst IP":                       "dst_ip",
    "Src Port":                     "src_port",
    "Dst Port":                     "dst_port",
    "Protocol":                     "protocol",
}

# Canonical columns label_flows() and the quality gates depend on existing
# after rename. If CICFlowMeter's schema drifts again (another version bump,
# another fork), we want a loud crash here — not a silent 100%-UNKNOWN
# mislabel three steps downstream like this exact bug just caused. Same
# lesson as the tcpreplay DEVNULL incident: a fast loud failure beats a slow
# silent one.
REQUIRED_CANONICAL_COLUMNS = ["src_ip", "dst_ip", "timestamp"]

def load_csv(csv_path: Path, retries: int = 3) -> pd.DataFrame:
    # Belt-and-suspenders on top of _wait_for_stable_file(): retry a
    # ParserError a few times with a short backoff. Same root cause (WSL2 bind
    # mount write-visibility lag between the container's `cp` and this native
    # Windows read) — confirmed 2026-08-12 by reproducing a "clean load, zero
    # malformed lines" result on the exact same bytes moments after a crash.
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
            break
        except pd.errors.ParserError as e:
            last_err = e
            # FOUND 2026-08-16 (D03's dh-1, 602MB combined CSV): this same
            # except branch was already catching a second, unrelated failure
            # mode beyond the WSL2 read race it was written for -- "Error
            # tokenizing data. C error: out of memory" fired twice in a row
            # while the host had only 470MB-1.2GB free out of 16.6GB
            # (confirmed via Get-CimInstance Win32_OperatingSystem during the
            # failure, not guessed). The plain retry-with-backoff happened to
            # work that time only because free memory drifted back up
            # between attempts -- not something this code controls or can
            # rely on. On a genuine OOM (vs. the race), retrying the same C
            # engine read is retrying the same peak-memory footprint; the
            # pyarrow engine (already a hard dependency for to_parquet
            # elsewhere in this pipeline) parses in a columnar, lower-peak-
            # memory fashion, so it's used as the fallback on the FINAL
            # attempt specifically when the error looks memory-related,
            # rather than burning through all retries on an approach that
            # doesn't change the memory profile at all.
            is_oom = "memory" in str(e).lower()
            if is_oom and attempt == retries:
                print(f"  [WARNING] CSV parse still out of memory after {retries} attempt(s) "
                      f"({csv_path.name}) — falling back to the pyarrow parser engine "
                      f"(lower peak memory than the default C engine; see "
                      f"problems_notes.txt 2026-08-16)", file=sys.stderr)
                df = pd.read_csv(csv_path, engine="pyarrow")
                break
            wait = 2 * attempt
            print(f"  [WARNING] CSV parse failed (attempt {attempt}/{retries}): {e} "
                  f"— retrying in {wait}s (likely a WSL2/Docker bind-mount read race, "
                  f"not a real data problem — see problems_notes.txt)", file=sys.stderr)
            time.sleep(wait)
    else:
        raise last_err

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    # Rename to canonical
    df.rename(columns={k.strip(): v for k, v in CICIDS_RENAME.items()}, inplace=True)

    missing = [c for c in REQUIRED_CANONICAL_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"load_csv({csv_path.name}): CICFlowMeter's output columns "
            f"({list(df.columns)}) don't cover CICIDS_RENAME's expectations — "
            f"missing canonical column(s) {missing} after rename. This means "
            f"the CICFlowMeter schema has changed again; update CICIDS_RENAME "
            f"to match df.columns before trusting any labels from this run."
        )

    # FOUND 2026-08-13 via validate_dataset.py's QG-12 reporting "0 numeric
    # feature columns" on an otherwise-passing dataset: every CICFlowMeter CSV
    # produced so far has exactly one data row where the column values are
    # literally the column's own header text (e.g. the "Flow Duration" cell's
    # value is the string "Flow Duration") — CICFlowMeter appears to re-emit
    # its header line once mid-file rather than only at the top. pandas' C
    # parser can't declare a column numeric if even one row's value doesn't
    # parse as a number, so that single row silently downgraded EVERY column
    # (all 84 of them) to object/string dtype. Two knock-on effects, both
    # invisible until QG-12 caught it: (1) QG-01/QG-02 (NaN/Inf checks) in
    # BOTH this script and validate_dataset.py filter to numeric-dtype
    # columns first — with nothing at numeric dtype, those gates were
    # checking zero columns and passing vacuously, not actually validating
    # anything, in every prior "PASS" result; (2) the parquet shipped to any
    # ML pipeline would have every feature stored as a string, not a number.
    # This is also the same row behind the "1/N timestamps did not match"
    # warning we'd been seeing and dismissing every run (sample: ['Timestamp']
    # — the header text again, this time in the timestamp column). Fix:
    # explicitly coerce every feature column to numeric and drop any row that
    # fails to coerce cleanly, instead of silently leaving whole columns as
    # strings for the rest of the pipeline to trip over.
    non_numeric_cols = {"src_ip", "dst_ip", "timestamp", "Flow ID", "Label"}
    feature_cols = [c for c in df.columns if c not in non_numeric_cols]
    n_before = len(df)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad_mask = df[feature_cols].isna().any(axis=1)
    n_bad = int(bad_mask.sum())
    if n_bad:
        print(f"  [WARNING] Dropping {n_bad}/{n_before} row(s) with non-numeric "
              f"garbage in feature columns (embedded duplicate CSV header row(s) "
              f"— see problems_notes.txt 2026-08-13)")
        df = df.loc[~bad_mask].reset_index(drop=True)

    # FOUND 2026-08-14 (D01 DOS_HULK dh-2, real evidence via pandas on the
    # actual parquet, not guessed): QG-02 flagged 2 Inf values. Traced to
    # exactly ONE flow with flow_duration == 0.0 (1 fwd + 1 bwd packet, both
    # landing within the same CICFlowMeter timestamp tick) -- flow_bytes_s
    # and flow_pkts_s are computed as bytes/duration and pkts/duration, so a
    # zero-duration flow divides by zero and both become +inf. This is a
    # documented CICFlowMeter artifact (the same thing shows up occasionally
    # in real CICIDS2017/2018 data for degenerate single-packet-pair flows),
    # not a labelling or extraction defect -- CICFlowMeter's own rate
    # formulas don't guard against a flow completing inside one timestamp
    # tick. At extreme packet rates (DOS_HULK's 500-thread flood is the only
    # family that's produced this so far, out of every session run to date)
    # this becomes possible where it wasn't at lower rates. Dropping rows
    # with non-finite values in feature columns here, same principle as the
    # non-numeric-garbage drop above: a flow whose own engineered features
    # are undefined can't be trained on regardless of which family it
    # belongs to, and silently replacing inf with a magic sentinel value
    # would fabricate ground truth rather than represent it honestly.
    inf_mask = np.isinf(df[feature_cols].select_dtypes(include=[np.number])).any(axis=1)
    n_inf = int(inf_mask.sum())
    if n_inf:
        print(f"  [WARNING] Dropping {n_inf} row(s) with Inf feature values "
              f"(zero-duration flow -> divide-by-zero rate features — see "
              f"problems_notes.txt 2026-08-14)")
        df = df.loc[~inf_mask].reset_index(drop=True)

    return df

# ── Labelling ─────────────────────────────────────────────────────────────────

def parse_utc(s: str) -> float:
    """Execute internal routine."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

# FOUND 2026-08-13 via the PORT_SCAN pilot session (the 6th family added to
# PILOT_PLAN, and the first attack in this project shorter than 1 second):
# CICFlowMeter's Timestamp column has no fractional-seconds component
# (confirmed 2026-08-12, format "%d/%m/%Y %I:%M:%S %p"), so every flow_ts is
# FLOORED to the start of its second. For every attack run before this one
# (180s+) that's negligible. nmap -sS --top-ports 1000 over the low-latency
# Docker bridge finished in 0.42s (manifest window
# 2026-08-13T16:05:21.466497 .. 16:05:21.888660), entirely inside a single
# second. CICFlowMeter's floored timestamp for every scan packet came out as
# 16:05:21.000000 -- earlier than our own recorded start_ts of
# 16:05:21.466497 -- so the exact `start_ts <= t <= end_ts` window check
# failed for essentially every real port-scan flow. Confirmed: 1000/1194
# flows in that session went UNKNOWN with zero flows ever getting the
# PORT_SCAN label, and it's the only session in the run whose manifest window
# is sub-second (every other family's window is tens of seconds or more).
# These are genuine ip_match=True attacker<->target flows, not benign
# traffic, so they couldn't fall into the "different-IP" BENIGN branch
# either -- straight to the UNKNOWN catch-all with no rescue path. Fix: pad
# the attack-window comparison with a small tolerance on both sides, enough
# to absorb up to ~1s of floor-truncation error plus margin. Safe to apply
# globally (not just to short attacks) because only the attack script itself
# ever talks between the attacker and target IPs in this lab -- a couple of
# seconds of slack can't accidentally pull in unrelated traffic.
ATTACK_WINDOW_TOLERANCE_S = 2.0

def label_flows(df: pd.DataFrame, manifest: list[dict], session_id: str) -> pd.DataFrame:
    """Execute internal routine."""
    # Find the manifest record for this session
    session_records = [r for r in manifest if r["session_id"] == session_id]
    if not session_records:
        print(f"  [label] No manifest record for session {session_id} — all UNKNOWN")
        df["label"] = "UNKNOWN"
        df["session_id"] = session_id
        return df

    record = session_records[0]
    attack_family = record["attack_family"]  # None for benign-only
    start_ts = parse_utc(record["start_time_utc"])
    end_ts   = parse_utc(record["end_time_utc"])
    src_ip   = record["src_ip"]
    dst_ip   = record["dst_ip"]

    # Multi-source ICMP flood (2026-08-14): ICMP has no ports to vary, so the
    # flood is generated from a set of REAL attacker aliases (see icmp_flood.sh)
    # recorded in the manifest as attack_src_ips. When present, an attack flow is
    # matched to the target by SOURCE-SET membership rather than the single
    # src_ip. Empty/absent for every other family and for older manifests, which
    # fall through to the original single-IP tuple match below.
    attack_src_set = set(record.get("attack_src_ips") or [])

    # FOUND 2026-08-13 during V2.1 Section 21/22 approval review: start_ts/
    # end_ts above bracket only the attack's own execution window, but the
    # spec's labelling logic (Section 9) says concurrent benign traffic
    # should be labelled BENIGN for any flow "within an ATTACK SESSION"
    # whose IPs don't match the attack tuple — i.e. the whole session
    # (warmup + attack + drain), not just the narrow attack window. Without
    # a wider bound, every bit of legitimate background/tcpreplay traffic
    # happening during the 120s warmup or 60s drain phases (180s out of a
    # ~360s session — half of it) was falling into neither the attack-match
    # branch nor the in-window-but-different-IP branch, landing in UNKNOWN
    # instead of BENIGN. This is what was actually behind ICMP_FLOOD's 51.6%
    # and FTP_BRUTE's 7.4% PER-SESSION UNKNOWN rates — both fail the spec's
    # own <5%-per-session QG-03 threshold, even though the pooled rate
    # across the whole pilot looked fine at 2.5%. orchestrate.py now writes
    # capture_start_utc/capture_end_utc (the true PCAP bounds) alongside the
    # existing attack-only start_time_utc/end_time_utc. Falls back to the
    # attack window for older manifest records that predate this fix (all
    # quarantined already, but don't want a KeyError if one somehow gets
    # re-processed).
    capture_start_ts = parse_utc(record["capture_start_utc"]) if "capture_start_utc" in record else start_ts
    capture_end_ts   = parse_utc(record["capture_end_utc"]) if "capture_end_utc" in record else end_ts

    # Parse flow timestamps. Format confirmed 2026-08-12 from real CICFlowMeter
    # output: "11/08/2026 10:31:49 PM" i.e. "%d/%m/%Y %I:%M:%S %p" (day-first,
    # 12-hour clock with AM/PM). This used to rely on pandas' format
    # auto-inference (dayfirst=True, no explicit format), which fell back to
    # slow per-row dateutil parsing and then raised on at least one row out of
    # 70-80k — with default errors="raise" that nulled out flow_ts for the
    # ENTIRE session, which was the real cause of every attack session coming
    # back 100% UNKNOWN (confirmed via debug prints — NOT a timezone offset:
    # the raw local timestamp above is within ~1s of the manifest's UTC
    # window, so the container clock is already UTC). Fix: parse with the
    # confirmed format explicitly and errors="coerce" so any row that still
    # doesn't match becomes NaT (safely excluded below) instead of aborting
    # the whole column.
    #
    # SECOND BUG (2026-08-12, found after the above fix): converting to POSIX
    # seconds via `parsed.astype("int64") / 1e9` assumes `parsed` is
    # datetime64[ns] (int64 view = nanoseconds since epoch). On this machine's
    # pandas (3.0.2) that assumption is false — to_datetime() with a format
    # string that has no fractional-seconds component returns
    # datetime64[us, UTC] (microsecond resolution), so .astype("int64") is
    # actually microseconds-since-epoch. Dividing that by 1e9 instead of 1e6
    # made every flow_ts exactly 1000x too small, landing every session in
    # January 1970 and putting it permanently outside any manifest window —
    # this is what produced the 78% UNKNOWN rate. Confirmed by reproducing the
    # exact factor-of-1000 discrepancy against a known real timestamp before
    # applying this fix. Fix: convert via a Timedelta division instead of a
    # raw int64 cast — this is correct regardless of the Series' internal
    # datetime64 resolution (s/ms/us/ns), so it can't silently break again if
    # a future pandas version infers yet another resolution. It also turns
    # NaT into NaN automatically, so the previous manual
    # `flow_ts[parsed.isna()] = np.nan` re-masking step is no longer needed.
    flow_ts = None
    if "timestamp" in df.columns:
        try:
            parsed = pd.to_datetime(
                df["timestamp"], format="%d/%m/%Y %I:%M:%S %p", utc=True, errors="coerce"
            )
            n_bad = int(parsed.isna().sum())
            if n_bad:
                bad_samples = df.loc[parsed.isna(), "timestamp"].head(3).tolist()
                print(f"  [WARNING] {n_bad}/{len(df)} timestamps did not match the expected "
                      f"format and were set to NaT — those flows can't be time-matched "
                      f"(sample: {bad_samples})")
            flow_ts = (parsed - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(seconds=1)
        except Exception as e:
            print(f"  [WARNING] Timestamp parsing raised an exception: {e}")
            flow_ts = None

    if attack_family and flow_ts is not None:
        print(f"  [check] manifest window (UTC): {record['start_time_utc']} .. {record['end_time_utc']}")
        sample_parsed = pd.to_datetime(flow_ts.dropna().head(3), unit="s", utc=True)
        print(f"  [check] parsed timestamp sample (UTC): {sample_parsed.tolist()}")

    labels = []
    for i, row in df.iterrows():
        # Match by IP tuple + time window
        row_src = str(row.get("src_ip", ""))
        row_dst = str(row.get("dst_ip", ""))

        in_attack_window = False
        in_capture_window = False
        if flow_ts is not None:
            t = flow_ts.iloc[i] if hasattr(flow_ts, "iloc") else None
            if t is not None and not pd.isna(t):
                in_attack_window = (start_ts - ATTACK_WINDOW_TOLERANCE_S) <= t <= (end_ts + ATTACK_WINDOW_TOLERANCE_S)
                in_capture_window = capture_start_ts <= t <= capture_end_ts

        if attack_src_set:
            # multi-source ICMP: any flow between an attacker alias and the target
            ip_match = (row_src in attack_src_set and row_dst == dst_ip) or \
                       (row_dst in attack_src_set and row_src == dst_ip)
        else:
            ip_match = (row_src == src_ip and row_dst == dst_ip) or \
                       (row_src == dst_ip and row_dst == src_ip)

        if attack_family and in_attack_window and ip_match:
            labels.append(attack_family)
        elif in_capture_window and not ip_match:
            labels.append("BENIGN")  # concurrent benign anywhere in the session's capture window
        elif attack_family is None:
            labels.append("BENIGN")  # benign-only session
        else:
            labels.append("UNKNOWN")

    df["label"] = labels
    df["session_id"] = session_id
    df["domain_id"]  = record["domain_id"]
    return df

# ── Quality gates ─────────────────────────────────────────────────────────────

def run_quality_gates(df: pd.DataFrame, domain: str) -> bool:
    """Execute internal routine."""
    passed = True

    # QG-01: No NaN
    nan_count = df.select_dtypes(include=[np.number]).isna().sum().sum()
    if nan_count > 0:
        print(f"  [QG-01 FAIL] {nan_count} NaN values in feature matrix")
        passed = False
    else:
        print(f"  [QG-01 PASS] No NaN")

    # QG-02: No Inf
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    if inf_count > 0:
        print(f"  [QG-02 FAIL] {inf_count} Inf values")
        passed = False
    else:
        print(f"  [QG-02 PASS] No Inf")

    # QG-03: UNKNOWN < 5% — PER SESSION, not pooled. V2.1 Section 8B states
    # the threshold explicitly as "UNKNOWN < 5% per session" and Section 9
    # repeats it as "UNKNOWN > 5% of any session's flows -> pipeline fails".
    # Found 2026-08-13 during Section 21/22 review: checking only the pooled
    # rate across all sessions let a domain through where the pooled rate
    # looked fine (2.5%) while two individual sessions were actually failing
    # (7.4% and 51.6%) — a large low-UNKNOWN session (SYN_FLOOD, 44k flows)
    # was diluting two much smaller high-UNKNOWN sessions in the average.
    # Report both: pooled rate for visibility, but gate on the worst session.
    pooled_rate = (df["label"] == "UNKNOWN").mean()
    per_session = df.groupby("session_id").apply(
        lambda g: (g["label"] == "UNKNOWN").mean(), include_groups=False
    )
    worst_session = per_session.idxmax()
    worst_rate = per_session.max()
    if worst_rate > 0.05:
        print(f"  [QG-03 FAIL] Worst-session UNKNOWN rate = {worst_rate:.1%} "
              f"in {worst_session} (threshold 5% per session; pooled rate "
              f"was {pooled_rate:.1%}, which would have looked fine)")
        failing = per_session[per_session > 0.05]
        print(f"    Sessions over threshold: {failing.round(3).to_dict()}")
        passed = False
    else:
        print(f"  [QG-03 PASS] Worst-session UNKNOWN rate = {worst_rate:.1%} "
              f"in {worst_session} (pooled rate {pooled_rate:.1%})")

    # QG-04: No CONFLICT
    conflict_count = (df["label"] == "CONFLICT").sum()
    if conflict_count > 0:
        print(f"  [QG-04 FAIL] {conflict_count} CONFLICT labels — investigate")
        passed = False
    else:
        print(f"  [QG-04 PASS] No CONFLICT labels")

    # QG-07: Session count per class (informational at this stage)
    class_sessions = df.groupby("label")["session_id"].nunique()
    print(f"  [QG-07 INFO] Sessions per class:\n{class_sessions.to_string()}")

    # QG-08: No single class > 50% of flows
    class_frac = df["label"].value_counts(normalize=True)
    dominant = class_frac[class_frac > 0.50]
    if not dominant.empty:
        print(f"  [QG-08 WARN] Class dominance > 50%: {dominant.to_dict()}")
    else:
        print(f"  [QG-08 PASS] No class dominates > 50%")

    return passed

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # FOUND 2026-08-16: the 2026-08-13 UnicodeEncodeError fix only forces
    # UTF-8 for children launched THROUGH run_full_generation.py's own
    # subprocess wrapper -- it never touched this script's own stdout, so a
    # direct invocation with redirected output (`python extract_and_label.py
    # ... > log.txt`, not via the wrapper) still hits the exact same crash on
    # the first non-ASCII print(). Same fix, same reasoning, applied at the
    # source instead of only at the one known caller.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--experiment-id", default=None,
                        help="Only process sessions from this experiment_id. "
                             "Default: the most recent experiment_id found in "
                             "the domain's manifest (manifest is append-only "
                             "across runs, so the last line is the newest).")
    parser.add_argument("--all-experiments", action="store_true",
                        help="Disable experiment-id filtering and process every "
                             "session ever recorded for this domain (old behavior).")
    parser.add_argument("--out", default="labelled.parquet",
                        help="Output parquet filename under data/processed/<domain>/ "
                             "(default labelled.parquet). Use a patch name like "
                             "labelled_patch.parquet when extracting a partial "
                             "--only-families regeneration, then splice it in with "
                             "merge_regenerated.py.")
    args = parser.parse_args()

    print(f"\nV2 Extract-and-Label Pipeline")
    print(f"Domain: {args.domain}")

    manifest = load_manifest(args.domain)

    # FOUND 2026-08-13: orchestrate.py appends to <domain>_manifest.jsonl on
    # every run and never cleans up old PCAPs from data/raw/pcaps/ either —
    # both are meant to accumulate across the full multi-session domain run,
    # but that same accumulation silently pulls STALE sessions from earlier,
    # already-superseded pilot attempts into every later extract-and-label
    # pass. Confirmed today: a fresh pilot re-run (post capture-window fix)
    # loaded 20 manifest records / found 8 PCAPs when the run itself only
    # produced 4 of each — the other 4 were leftovers from a run captured
    # BEFORE capture_start_utc/capture_end_utc existed, so they reproduced
    # the exact old high-UNKNOWN-rate bug (7.4%, 51.6%) and masked the fact
    # that the fix's own 4 fresh sessions were all well under 1% UNKNOWN.
    # Every manifest record already carries an experiment_id (written by
    # orchestrate.py's write_manifest()) that was simply never used
    # downstream. Default to only the latest experiment_id so a re-run
    # always evaluates itself, not itself plus every prior attempt.
    if not args.all_experiments:
        target_experiment = args.experiment_id or (manifest[-1]["experiment_id"] if manifest else None)
        if target_experiment:
            n_total = len(manifest)
            manifest = [r for r in manifest if r["experiment_id"] == target_experiment]
            n_excluded = n_total - len(manifest)
            print(f"[experiment-filter] Using experiment_id={target_experiment!r}: "
                  f"{len(manifest)}/{n_total} manifest record(s) "
                  f"({n_excluded} excluded from other/older runs)")

    session_ids = {r["session_id"] for r in manifest}

    # Find PCAPs for this domain
    pcaps = sorted(PCAP_DIR.glob(f"{args.domain}-*.pcap"))
    if not pcaps:
        pcaps = sorted(PCAP_DIR.glob("*.pcap"))  # pilot: no domain prefix yet
    if not args.all_experiments and session_ids:
        n_total_pcaps = len(pcaps)
        pcaps = [p for p in pcaps if p.stem in session_ids]
        n_excluded_pcaps = n_total_pcaps - len(pcaps)
        if n_excluded_pcaps:
            print(f"[experiment-filter] Excluding {n_excluded_pcaps} stale PCAP(s) "
                  f"not in the selected experiment")
    print(f"Found {len(pcaps)} PCAP(s)")

    flows_out = FLOWS_DIR / args.domain
    flows_out.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    failed_sessions = []  # FOUND 2026-08-14 (DOS_HULK dh-1/dh-2 silently
    # vanishing from D01): a PCAP that fails extract_pcap() was previously
    # just `continue`d past with a single [ERROR] line easy to miss in a
    # long multi-session run's output, and NOTHING at the end summarised
    # which sessions/families never made it into the final parquet. Tracking
    # them here so there's an unmissable summary right before Quality Gates,
    # instead of relying on a human spotting one buried line.
    for pcap in pcaps:
        session_id = pcap.stem
        print(f"\n── {session_id} ──")

        # Extract
        csv_path = extract_pcap(pcap, flows_out)
        if csv_path is None:
            failed_sessions.append(session_id)
            continue

        # Load
        df = load_csv(csv_path)
        print(f"  [load] {len(df)} flows, {df.shape[1]} columns")

        # Label
        df = label_flows(df, manifest, session_id)
        print(f"  [label] {df['label'].value_counts().to_dict()}")

        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No flows extracted.", file=sys.stderr)
        sys.exit(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    print(f"\n── Combined: {len(combined)} flows ──")
    print(combined["label"].value_counts())

    if failed_sessions:
        print(f"\n[!!] {len(failed_sessions)} session(s) FAILED EXTRACTION and are "
              f"MISSING from the parquet below — their attack family may now have "
              f"fewer sessions than DOMAIN_PLANS configured, or be entirely absent:")
        for s in failed_sessions:
            print(f"     - {s}  (see data/flows/{args.domain}/{s}_CRASH.log)")

    # Quality gates
    # NOTE: this is a lightweight PRELIMINARY check (QG-01/02/03/04/07-info/08
    # only) meant as an immediate sanity read right after extraction -- it does
    # NOT include QG-13 (missing-attack-family coverage against DOMAIN_PLANS),
    # QG-06, QG-09 through QG-12, etc. "[OK] All quality gates passed" below
    # means only THIS subset passed, not that the domain is validated.
    # validate_dataset.py --domain {domain} --min-sessions 2 is the
    # authoritative check -- always run it before treating a domain as done,
    # even if this preliminary check looks clean (confirmed 2026-08-14: this
    # exact check reported "[OK] All quality gates passed" for a D01 run
    # where DOS_HULK was completely missing from the data).
    print("\n── Quality Gates (PRELIMINARY -- run validate_dataset.py for the authoritative check) ──")
    passed = run_quality_gates(combined, args.domain)

    # Save
    out_path = PROCESSED_DIR / args.domain / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    print(f"\n[output] Saved: {out_path} ({out_path.stat().st_size // 1024} KB)")
    print(f"[NEXT] Run: python scripts/validate_dataset.py --domain {args.domain} --min-sessions 2")

    if not passed:
        print("\n[WARNING] Some quality gates failed. Investigate before ML training.")
        sys.exit(2)
    else:
        print("\n[OK] All quality gates passed.")

if __name__ == "__main__":
    main()
