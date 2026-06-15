#!/usr/bin/env python3
"""
End-to-End Dataset Generation and Verification Runner.

Executes sequential multi-domain testbed simulations, triggers PCAP capture, launches flow extraction,
and runs automated quality gate validations.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # ML IDS THESIS/
LAB_SCRIPTS = REPO_ROOT / "lab" / "scripts"

MIN_SESSIONS_PER_DOMAIN = 2  # see module docstring -- matches Section 8C, NOT validate_dataset.py's stale default of 10

ALL_DOMAINS = ["D01", "D02", "D03", "D04"]

def run(cmd: list[str], log_path: Path) -> int:
    # FOUND 2026-08-13 (first real full-generation attempt, D01 session 1/19):
    # orchestrate.py/extract_and_label.py/validate_dataset.py all print
    # non-ASCII characters (arrows, en-dashes, checkmarks -- confirmed via a
    # direct scan of all three files: -, ..., ->, >=, checkmark, x-mark all
    # appear in real print() calls, not just comments). When one of those
    # scripts is run directly in an interactive PowerShell console, Windows
    # Python detects a console and generally handles this fine. But here
    # they're launched as a SUBPROCESS with stdout redirected through a pipe
    # (not a console) -- and when Python's stdout isn't a real console on
    # Windows, it falls back to the OS's locale-preferred encoding (cp1252 on
    # this machine) instead of UTF-8, unless told otherwise. The very first
    # arrow character printed (orchestrate.py's "[capture] Started -> ...")
    # crashed with UnicodeEncodeError, killing the whole 19-session D01 run
    # ~2 minutes in, during benign warmup, before any attack had fired.
    # Fixed by forcing the CHILD to use UTF-8 for its own stdout
    # (PYTHONIOENCODING/PYTHONUTF8 env vars) and telling THIS process's Popen
    # to decode the pipe as UTF-8 too, rather than patching every individual
    # print() call in three separate scripts (fragile, and the next new
    # print() added to any of them would just reintroduce the same crash).
    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["PYTHONUTF8"] = "1"

    print(f"\n{'='*90}\n$ {' '.join(cmd)}\n{'='*90}")
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace", env=child_env,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            logf.write(line)
        proc.wait()
    return proc.returncode

def main() -> None:
    # Belt-and-suspenders: also guard THIS process's own stdout, in case it's
    # ever invoked non-interactively (e.g. redirected to a file/log) where
    # Windows would otherwise pick cp1252 the same way the child processes
    # did above. errors="replace" means a genuinely unrepresentable character
    # degrades to a placeholder instead of crashing a 14-18h run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run full D01-D04 generation, hard-stop on failure")
    parser.add_argument("--domains", nargs="+", default=ALL_DOMAINS, choices=ALL_DOMAINS)
    parser.add_argument("--skip-validate", action="store_true",
                         help="Skip validate_dataset.py per domain (NOT recommended)")
    parser.add_argument("--skip-training-pool", action="store_true",
                         help="Skip build_training_set.py per domain (labelled.parquet is still "
                              "written either way; this only skips deriving balanced_train.parquet)")
    parser.add_argument("--min-sessions", type=int, default=MIN_SESSIONS_PER_DOMAIN,
                         help="QG-07 threshold passed to validate_dataset.py (default: 2, matches Section 8C)")
    args = parser.parse_args()

    log_dir = REPO_ROOT / "lab" / "logs" / "full_generation"
    log_dir.mkdir(parents=True, exist_ok=True)

    overall_start = time.time()
    results = []

    for domain in args.domains:
        domain_start = time.time()
        print(f"\n\n######## DOMAIN {domain} — START ########")

        # Stage 1: generate traffic
        rc = run(
            [sys.executable, str(LAB_SCRIPTS / "orchestrate.py"), "--domain", domain],
            log_dir / f"{domain}_1_orchestrate.log",
        )
        if rc != 0:
            print(f"[HARD STOP] orchestrate.py failed for {domain} (exit {rc}). "
                  f"Check {log_dir / f'{domain}_1_orchestrate.log'} before doing anything else. "
                  f"Do NOT proceed to the next domain -- fix and re-run this domain first.")
            sys.exit(1)

        # Stage 2: extract + label (auto-picks the experiment_id orchestrate.py just wrote)
        rc = run(
            [sys.executable, str(LAB_SCRIPTS / "extract_and_label.py"), "--domain", domain],
            log_dir / f"{domain}_2_extract.log",
        )
        if rc != 0:
            print(f"[HARD STOP] extract_and_label.py failed for {domain} (exit {rc}). "
                  f"Check {log_dir / f'{domain}_2_extract.log'}.")
            sys.exit(1)

        # Stage 3: quality gates
        if not args.skip_validate:
            rc = run(
                [sys.executable, str(LAB_SCRIPTS / "validate_dataset.py"), "--domain", domain,
                 "--min-sessions", str(args.min_sessions)],
                log_dir / f"{domain}_3_validate.log",
            )
            # FOUND 2026-08-14 (D01's first real full-generation attempt):
            # validate_dataset.py's OWN documented exit-code convention
            # (its module docstring, verified by reading the actual sys.exit
            # calls) is 0=all gates passed, 1=hard failure (must fix),
            # 2=soft warning (investigate but may proceed). This wrapper
            # originally treated ANY non-zero exit as fatal, which meant a
            # completely clean D01 result -- 10/12 gates passed, only the
            # two already-understood soft warnings QG-06/QG-08 -- got
            # treated identically to a genuine hard failure and stopped the
            # whole 14-18h run after less than 4 hours, on a false alarm.
            # Only rc==1 (a real HARD_GATES failure -- QG-01/02/03/04/05/07/
            # 13) should actually stop the pipeline; rc==2 is the script
            # explicitly telling us it's safe to continue.
            if rc == 1:
                print(f"[HARD STOP] validate_dataset.py reported a HARD GATE failure for {domain} "
                      f"(exit 1). Check {log_dir / f'{domain}_3_validate.log'} -- this means the "
                      f"produced data genuinely does not meet a hard quality gate "
                      f"(QG-01/02/03/04/05/07/13). Do not proceed to the next domain with unvalidated data.")
                sys.exit(1)
            elif rc == 2:
                print(f"[SOFT WARNING] {domain} passed all hard gates but has soft-gate warnings "
                      f"(exit 2) -- see {log_dir / f'{domain}_3_validate.log'} for which ones. "
                      f"Proceeding to the next stage; investigate these before final ML training.")
            elif rc != 0:
                print(f"[HARD STOP] validate_dataset.py exited unexpectedly for {domain} (exit {rc}, "
                      f"neither the documented 0/1/2). Treating as fatal out of caution -- "
                      f"check {log_dir / f'{domain}_3_validate.log'}.")
                sys.exit(1)

        # Stage 4: derive the balanced TRAINING pool (labelled.parquet stays full/untouched --
        # R5 auditability -- this only ever writes balanced_train.parquet alongside it). Runs
        # whenever we reach this point, i.e. whenever Stage 3 didn't hard-stop above (a clean
        # pass, a soft-warning pass, or validate skipped entirely all count -- QG-06/QG-08-style
        # warnings don't block building the pool, same as the real D01 run).
        if not args.skip_training_pool:
            rc = run(
                [sys.executable, str(LAB_SCRIPTS / "build_training_set.py"), "--domain", domain],
                log_dir / f"{domain}_4_build_training_set.log",
            )
            if rc != 0:
                print(f"[HARD STOP] build_training_set.py failed for {domain} (exit {rc}). "
                      f"Check {log_dir / f'{domain}_4_build_training_set.log'}. labelled.parquet "
                      f"for this domain is fine either way (this stage only derives the training "
                      f"pool) -- but do not proceed to the next domain until this is understood.")
                sys.exit(1)

        elapsed = time.time() - domain_start
        print(f"######## DOMAIN {domain} — DONE ({elapsed/3600:.2f}h) ########")
        results.append((domain, elapsed))

    total = time.time() - overall_start
    print(f"\n\nALL DOMAINS COMPLETE. Total: {total/3600:.2f}h")
    for domain, elapsed in results:
        print(f"  {domain}: {elapsed/3600:.2f}h")

if __name__ == "__main__":
    main()
