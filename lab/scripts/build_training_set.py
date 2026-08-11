#!/usr/bin/env python3
"""
Session-Stratified Training Set Balancing and Flow Capping.

Applies proportional per-session capping to dominant attack classes (e.g., DoS Hulk)
to prevent gradient starvation on rare attacks while preserving natural intra-session variance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Meta-labels that are never training classes.
NON_TRAINING_LABELS = {"UNKNOWN", "CONFLICT"}

def stratified_cap(df_class: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    """Sample flows proportionally across recording sessions to meet target class quota."""
    if "session_id" not in df_class.columns or len(df_class) <= cap:
        return df_class if len(df_class) <= cap else df_class.sample(n=cap, random_state=seed)

    groups = list(df_class.groupby("session_id"))
    n_sessions = len(groups)
    per_session = max(1, cap // n_sessions)

    parts = []
    for sid, g in groups:
        parts.append(g if len(g) <= per_session else g.sample(n=per_session, random_state=seed))
    capped = pd.concat(parts, ignore_index=True)

    # If uneven session sizes left us short of the cap, top up from the remainder
    # (still deterministic) so we use the budget we asked for.
    if len(capped) < cap and len(df_class) > len(capped):
        remainder = df_class.drop(capped.index, errors="ignore")
        need = min(cap - len(capped), len(remainder))
        if need > 0:
            capped = pd.concat([capped, remainder.sample(n=need, random_state=seed)], ignore_index=True)
    return capped

def build(domain: str, per_class_cap: int, benign_cap: int, seed: int,
          min_floor: int) -> int:
    parquet = PROCESSED_DIR / domain / "labelled.parquet"
    if not parquet.exists():
        print(f"[ERROR] Parquet not found: {parquet}", file=sys.stderr)
        print("Run extract_and_label.py first.", file=sys.stderr)
        return 1

    df = pd.read_parquet(parquet)
    before = df["label"].value_counts().to_dict()
    print(f"\nBuild Balanced Training Pool — {domain}")
    print(f"  Source: {parquet}  ({len(df):,} flows)")
    print(f"  Seed={seed}  per-class cap={per_class_cap:,}  benign cap={benign_cap:,}  "
          f"min floor={min_floor}")

    # Drop non-training meta-labels
    df = df[~df["label"].isin(NON_TRAINING_LABELS)].copy()

    kept, thin = [], []
    for label, g in df.groupby("label"):
        cap = benign_cap if label == "BENIGN" else per_class_cap
        capped = stratified_cap(g, cap, seed)
        kept.append(capped)
        if len(g) < min_floor:
            thin.append((label, len(g)))

    balanced = pd.concat(kept, ignore_index=True)
    # Deterministic shuffle so downstream readers don't see class-ordered rows.
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    after = balanced["label"].value_counts().to_dict()

    print("\n  Per-class flow counts (raw -> balanced):")
    for label in sorted(before, key=lambda k: -before[k]):
        if label in NON_TRAINING_LABELS:
            print(f"    {label:<16} {before[label]:>10,}  ->  (dropped: non-training meta-label)")
        else:
            print(f"    {label:<16} {before[label]:>10,}  ->  {after.get(label, 0):>8,}")

    frac = balanced["label"].value_counts(normalize=True)
    print(f"\n  Balanced pool: {len(balanced):,} flows, max class share = {frac.max():.1%} "
          f"({frac.idxmax()})")

    if thin:
        print("\n  [!!] Classes below the min floor — they need REGENERATION, capping "
              "can't create data:")
        for label, n in thin:
            print(f"       {label}: {n} flows (< {min_floor})")
        print("       (ICMP_FLOOD / UDP_DDOS: regenerate with flow-rich variants before "
              "trusting this pool — see problems_notes.txt 2026-08-14)")

    out_path = PROCESSED_DIR / domain / "balanced_train.parquet"
    balanced.to_parquet(out_path, index=False)
    print(f"\n[output] Saved balanced TRAINING pool: {out_path} "
          f"({out_path.stat().st_size // 1024} KB)")
    print(f"[REMINDER] Evaluate on the NATURAL distribution (labelled.parquet), "
          f"never on this rebalanced pool.")
    return 0

def main() -> None:
    # FOUND 2026-08-16 (see extract_and_label.py's identical guard): the
    # 2026-08-13 UnicodeEncodeError fix only covers this script when it's
    # launched THROUGH run_full_generation.py's subprocess wrapper -- a
    # direct invocation with redirected stdout still hits the same crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--per-class-cap", type=int, default=25000,
                    help="Max flows per attack class in the training pool "
                         "(session-stratified; classes under it are kept whole).")
    ap.add_argument("--benign-cap", type=int, default=None,
                    help="Separate cap for BENIGN (default: same as --per-class-cap). "
                         "Raise it if the benign-FPR study wants more benign.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-floor", type=int, default=100,
                    help="Warn if a class has fewer than this many flows (matches QG-14).")
    args = ap.parse_args()
    benign_cap = args.benign_cap if args.benign_cap is not None else args.per_class_cap
    sys.exit(build(args.domain, args.per_class_cap, benign_cap, args.seed, args.min_floor))

if __name__ == "__main__":
    main()
