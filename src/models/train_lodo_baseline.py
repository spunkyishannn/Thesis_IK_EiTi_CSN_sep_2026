"""
Leave-One-Domain-Out (LODO) Baseline Generalization Model.

Trains a Random Forest classifier across three lab domains and tests on a held-out domain's
unmodified natural flow distribution to establish initial cross-domain performance.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
REPORT_DIR = REPO_ROOT / "reports"
DOMAINS = ["D01", "D02", "D03"]  # TEMP 2026-08-22: see build_primary_split.py's matching note

ID_COLS = ["Flow ID", "src_ip", "dst_ip", "timestamp", "Label", "label", "session_id", "domain_id"]

RF_KWARGS = dict(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced_subsample")

def load(domain: str, kind: str) -> pd.DataFrame:
    path = DATA_DIR / domain / f"{kind}.parquet"
    df = pd.read_parquet(path)
    before = len(df)
    df = df[df["label"] != "UNKNOWN"].copy()
    dropped = before - len(df)
    print(f"  [{domain}/{kind}] {len(df):,} flows (dropped {dropped:,} UNKNOWN)")
    return df

def feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]

def fit_predict_report(train_df: pd.DataFrame, test_df: pd.DataFrame, label: str) -> dict:
    feats = feature_cols(train_df)
    X_train, y_train = train_df[feats].to_numpy(dtype=np.float64), train_df["label"].to_numpy()
    X_test, y_test = test_df[feats].to_numpy(dtype=np.float64), test_df["label"].to_numpy()

    t0 = time.time()
    clf = RandomForestClassifier(**RF_KWARGS)
    clf.fit(X_train, y_train)
    train_s = time.time() - t0

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    labels_sorted = sorted(set(y_train) | set(y_test))
    report = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    print(f"  [{label}] train={len(train_df):,} test={len(test_df):,} "
          f"fit={train_s:.1f}s  macro-F1={macro_f1:.4f}")
    per_class = {c: round(report[c]["f1-score"], 4) for c in labels_sorted if c in report}
    for c, f1 in sorted(per_class.items(), key=lambda kv: kv[1]):
        print(f"      {c:15s} F1={f1:.3f}  recall={report[c]['recall']:.3f}  support={int(report[c]['support'])}")

    return {
        "label": label,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "fit_seconds": round(train_s, 1),
        "macro_f1": round(macro_f1, 4),
        "per_class_f1": per_class,
        "labels": labels_sorted,
        "confusion_matrix": cm.tolist(),
    }

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading balanced training pools (D01-D04)...")
    train_pools = {d: load(d, "balanced_train") for d in DOMAINS}
    # NOTE: natural-distribution labelled.parquet (up to ~2.5M rows/domain) is
    # loaded LAZILY, one domain at a time inside the LODO loop below, not all
    # four upfront -- host had only 3.67GB free RAM at script start (checked
    # directly, see problems_notes.txt 2026-08-16), and holding all four
    # domains' full natural data in memory simultaneously (~6GB+) risks the
    # same host-memory-pressure failure mode already hit twice this project.
    # LODO only ever needs ONE held-out domain's natural data live at a time.

    results = {}

    # ---- 1. Sanity check (L1): random flow split, NOT trusted ----
    print("\n" + "=" * 70)
    print("SANITY CHECK (L1) -- random 80/20 flow split, pooled balanced data")
    print("NOT a real result. Random-split F1 is inflated by session leakage")
    print("(same session's flows in both train and test). See R1/R4.")
    print("=" * 70)
    pooled = pd.concat(train_pools.values(), ignore_index=True)
    tr, te = train_test_split(pooled, test_size=0.2, random_state=42, stratify=pooled["label"])
    results["L1_sanity_random_split"] = fit_predict_report(tr, te, "L1 sanity (random split)")

    # ---- 2. LODO (L4): the real result ----
    print("\n" + "=" * 70)
    print("LODO (L4) -- leave-one-domain-out, 4-fold")
    print("Train on 3 domains' balanced pool, test on the 4th domain's FULL")
    print("natural-distribution labelled.parquet. This is the first honest")
    print("cross-domain number this project has been able to produce.")
    print("=" * 70)
    lodo_folds = {}
    for held_out in DOMAINS:
        train_domains = [d for d in DOMAINS if d != held_out]
        train_df = pd.concat([train_pools[d] for d in train_domains], ignore_index=True)
        print(f"\nLoading held-out domain {held_out}'s natural distribution for testing...")
        test_df = load(held_out, "labelled")
        fold = fit_predict_report(train_df, test_df, f"LODO fold (test={held_out})")
        del test_df  # free before the next fold loads its own held-out domain
        lodo_folds[held_out] = fold

    lodo_mean = round(float(np.mean([f["macro_f1"] for f in lodo_folds.values()])), 4)
    results["L4_lodo"] = {"folds": lodo_folds, "mean_macro_f1": lodo_mean}

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  L1 sanity (random split, not trusted): macro-F1 = {results['L1_sanity_random_split']['macro_f1']:.4f}")
    print(f"  L4 LODO mean (the real number):        macro-F1 = {lodo_mean:.4f}")
    for d, f in lodo_folds.items():
        print(f"    test={d}: {f['macro_f1']:.4f}")
    gap = results['L1_sanity_random_split']['macro_f1'] - lodo_mean
    print(f"\n  Gap (L1 - L4): {gap:.4f} -- this gap IS the generalisation story.")
    print(f"  V2.1 spec success criterion (Section 19): LODO mean (L4) >= 0.60")
    verdict = "MEETS" if lodo_mean >= 0.60 else "DOES NOT MEET"
    print(f"  Result: {lodo_mean:.4f} -- {verdict} the 0.60 threshold.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "lodo_baseline_results_20260816.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    main()
