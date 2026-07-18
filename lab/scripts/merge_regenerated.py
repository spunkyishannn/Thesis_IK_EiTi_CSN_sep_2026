#!/usr/bin/env python3
"""
Dataset Partition Consolidation and Patch Utility.

Merges partitioned session extractions into unified domain Parquet datasets, ensuring
consistent schema alignment and deduplication across multi-session capture runs.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--families", default="ICMP_FLOOD,UDP_DDOS",
                    help="Comma-separated families being replaced (their OLD sessions "
                         "are dropped whole before the patch is appended).")
    ap.add_argument("--patch", default="labelled_patch.parquet",
                    help="Patch parquet filename under data/processed/<domain>/.")
    ap.add_argument("--base", default="labelled.parquet",
                    help="Base parquet filename under data/processed/<domain>/.")
    args = ap.parse_args()

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    ddir = PROCESSED_DIR / args.domain
    base_path  = ddir / args.base
    patch_path = ddir / args.patch

    for p in (base_path, patch_path):
        if not p.exists():
            print(f"[ERROR] Not found: {p}", file=sys.stderr)
            sys.exit(1)

    base  = pd.read_parquet(base_path)
    patch = pd.read_parquet(patch_path)
    if "session_id" not in base.columns or "session_id" not in patch.columns:
        print("[ERROR] Both parquets must have a session_id column.", file=sys.stderr)
        sys.exit(1)

    print(f"Merge regenerated families into {args.domain}")
    print(f"  base  : {base_path.name}  ({len(base):,} flows)")
    print(f"  patch : {patch_path.name}  ({len(patch):,} flows)")
    print(f"  families replaced: {families}")

    # 1. back up the base (reversible; keeps the raw record)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    backup = ddir / f"labelled_prepatch_{ts}.parquet"
    base.to_parquet(backup, index=False)
    print(f"  backup: {backup.name}")

    # 2. drop the OLD sessions of the regenerated families, keyed by session_id
    #    (session_id pattern: <domain>-<FAMILY>-<variant>-<time>), so their
    #    concurrent-benign flows go too and nothing is double-counted.
    drop_mask = pd.Series(False, index=base.index)
    for fam in families:
        drop_mask |= base["session_id"].astype(str).str.contains(f"-{fam}-", regex=False)
    n_dropped = int(drop_mask.sum())
    dropped_sessions = sorted(base.loc[drop_mask, "session_id"].unique())
    kept = base.loc[~drop_mask]

    # sanity: the patch should only contain the regenerated families' sessions
    patch_families = [l for l in patch["label"].unique()
                      if l not in {"BENIGN", "UNKNOWN", "CONFLICT"}]
    print(f"  dropped {n_dropped:,} flow(s) across {len(dropped_sessions)} old session(s): {dropped_sessions}")
    print(f"  patch attack families: {sorted(patch_families)}")

    # 3. concatenate and save
    merged = pd.concat([kept, patch], ignore_index=True)
    merged.to_parquet(base_path, index=False)

    print("\n  Per-class counts (base -> merged):")
    b = base["label"].value_counts().to_dict()
    m = merged["label"].value_counts().to_dict()
    for label in sorted(set(b) | set(m), key=lambda k: -m.get(k, 0)):
        print(f"    {label:<16} {b.get(label,0):>10,}  ->  {m.get(label,0):>10,}")
    print(f"\n[output] Wrote merged {base_path.name} ({len(merged):,} flows). "
          f"Backup: {backup.name}")
    print(f"[NEXT] python scripts/validate_dataset.py --domain {args.domain} --min-sessions 2")

if __name__ == "__main__":
    main()
