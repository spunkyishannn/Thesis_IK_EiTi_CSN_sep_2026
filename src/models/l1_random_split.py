"""
L1 Random Flow Split Baseline Evaluation.

Performs a standard 80/20 stratified random flow split across pooled natural traffic
(D01-D04) using frozen model hyperparameters. Serves as the L1 baseline on the
generalization ladder to quantify in-domain session leakage against grouped evaluations (L2-L6).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_hpo import (  # noqa: E402
    DOMAINS, NON_TRAINING_LABELS, PER_CLASS_CAP, SEED, build_capped_train_pool, fit_model_with_class_weight,
)

CONFIG_NAME = "frozen_model_config_v2.json"
MODELS = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
OUT_PATH = REPORT_DIR / "l1_random_split_results.json"

def load_all_natural(feature_cols: list[str]) -> pd.DataFrame:
    needed = feature_cols + ["label", "session_id"]
    frames = []
    for d in DOMAINS:
        df = pd.read_parquet(DATA_DIR / d / "labelled.parquet", columns=needed)
        frames.append(df[~df["label"].isin(NON_TRAINING_LABELS)])
    return pd.concat(frames, ignore_index=True)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen = json.loads((CONFIGS_DIR / CONFIG_NAME).read_text(encoding="utf-8"))
    feature_cols = frozen["feature_columns"]

    print("Pooling ALL D01-D04 natural flows (ignoring the session split -- L1's whole point)...")
    all_natural = load_all_natural(feature_cols)
    print(f"  {len(all_natural):,} flows")

    print(f"Session-stratified capping ({PER_CLASS_CAP:,}/class, seed={SEED})...")
    pool = build_capped_train_pool(all_natural)
    print(f"  Capped pool: {len(pool):,} flows")
    print(pool["label"].value_counts().to_string())

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(pool["label"].to_numpy())
    X = pool[feature_cols].to_numpy(dtype=np.float64)

    print("\n80/20 stratified flow-level split (NOT session-aware)...")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    print(f"  train={len(X_tr):,}  test={len(X_te):,}")

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        results = {"description": "L1 evaluation: 80/20 random flow split baseline for quantifying session leakage."
,
                   "n_flows": len(pool), "models": {}}

    for name in MODELS:
        if name in results["models"]:
            print(f"[skip] {name} already has a result.")
            continue
        params = frozen["models"][name]["best_hyperparameters"]
        print(f"\n{name}: fitting, params={params}", flush=True)
        t0 = time.time()
        clf, fell_back = fit_model_with_class_weight(name, params, X_tr, y_tr)
        fit_s = time.time() - t0
        y_pred = clf.predict(X_te)
        macro_f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
        labels_sorted = sorted(set(y_tr) | set(y_te))
        report = classification_report(y_te, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
        per_class_f1 = {label_encoder.classes_[c]: round(report[str(c)]["f1-score"], 4)
                         for c in labels_sorted if str(c) in report}
        print(f"  macro-F1={macro_f1:.4f}  fit={fit_s:.0f}s"
              f"{'  [fell back to LogReg]' if fell_back else ''}", flush=True)
        results["models"][name] = {
            "hyperparameters": params, "linearsvc_fell_back_to_logreg": fell_back if name == "LinearSVC" else None,
            "macro_f1": round(float(macro_f1), 4), "per_class_f1": per_class_f1, "fit_seconds": round(fit_s, 1),
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [checkpoint] {name} saved to {OUT_PATH.name}")

    print(f"\nL1 COMPLETE. Written: {OUT_PATH}")
    for name in MODELS:
        print(f"  {name:10s} macro-F1={results['models'][name]['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
