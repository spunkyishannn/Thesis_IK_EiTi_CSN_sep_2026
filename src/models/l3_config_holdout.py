"""
L3 Attack-Configuration Holdout Evaluation.

Measures model sensitivity to unseen attack parameters and tool configurations by training
on baseline attack sessions and evaluating on held-out parameter variants.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "reports"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_hpo import (  # noqa: E402
    DOMAINS, NON_TRAINING_LABELS, SEED, build_capped_train_pool, fit_model_with_class_weight,
)

CONFIG_NAME = "frozen_model_config_v2.json"
MODELS = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
OUT_PATH = REPORT_DIR / "l3_config_holdout_results.json"

# family_prefix -> held-out variant substring (matched against session_id).
# DOS_HULK/DOS_SLOWLORIS deliberately absent -- no genuine configuration axis
# in the real generated data, see module docstring's "IMPORTANT CORRECTION".
HELD_OUT_VARIANT = {
    "SYN_FLOOD": "sf-high",
    "ICMP_FLOOD": "ic-high",
    "UDP_DDOS": "ud-med",
    "FTP_BRUTE": "fb-medusa",
    "SSH_BRUTE": "sb-ncrack",
    "PORT_SCAN": "ps-aggr",
}

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

    print("Loading all D01-D04 natural flows...")
    all_natural = load_all_natural(feature_cols)
    print(f"  {len(all_natural):,} flows")

    is_held_out = pd.Series(False, index=all_natural.index)
    print("\nHeld-out variant sessions found:")
    for family, variant_substr in HELD_OUT_VARIANT.items():
        mask = all_natural["session_id"].str.contains(variant_substr, regex=False)
        sids = sorted(all_natural.loc[mask, "session_id"].unique())
        print(f"  {family:15s} -> '{variant_substr}': {len(sids)} session(s): {sids}")
        is_held_out |= mask

    test_df = all_natural[is_held_out].copy()
    train_natural = all_natural[~is_held_out].copy()
    print(f"\nHeld-out test set: {len(test_df):,} flows from {test_df['session_id'].nunique()} sessions")
    print(f"Remaining train pool source: {len(train_natural):,} flows")

    print("\nSession-stratified capping the remaining pool...")
    train_pool = build_capped_train_pool(train_natural)
    print(f"  Capped pool: {len(train_pool):,} flows")

    # Sanity: no held-out session_id may leak into the capped train pool
    # (QG-06's own enforcement rule, re-checked here since this is a
    # DIFFERENT split axis than the one QG-06 normally validates).
    leaked = set(train_pool["session_id"].unique()) & set(test_df["session_id"].unique())
    if leaked:
        raise RuntimeError(f"Held-out session(s) leaked into the L3 train pool: {leaked}")
    print("  Verified: zero held-out sessions present in the train pool.")

    label_encoder = LabelEncoder()
    all_labels = sorted(set(train_pool["label"].unique()) | set(test_df["label"].unique()))
    label_encoder.fit(all_labels)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    y_train = label_encoder.transform(train_pool["label"].to_numpy())
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float64))
    y_test = label_encoder.transform(test_df["label"].to_numpy())

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    else:
        results = {
            "description": "L3 evaluation: Attack-configuration holdout testing across unseen network parameters."
,
            "held_out_variants": HELD_OUT_VARIANT,
            "n_test_flows": len(test_df), "n_test_sessions": int(test_df["session_id"].nunique()),
            "n_train_flows_capped": len(train_pool), "models": {},
        }

    for name in MODELS:
        if name in results["models"]:
            print(f"[skip] {name} already has a result.")
            continue
        params = frozen["models"][name]["best_hyperparameters"]
        print(f"\n{name}: fitting, params={params}", flush=True)
        t0 = time.time()
        clf, fell_back = fit_model_with_class_weight(name, params, X_train, y_train)
        fit_s = time.time() - t0
        y_pred = clf.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        labels_sorted = sorted(set(y_train) | set(y_test))
        report = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
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

    print(f"\nL3 COMPLETE. Written: {OUT_PATH}")
    for name in MODELS:
        print(f"  {name:10s} macro-F1={results['models'][name]['macro_f1']:.4f}")

if __name__ == "__main__":
    main()
