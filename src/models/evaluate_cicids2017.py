"""
External Benchmark Evaluation on CICIDS2017 Dataset.

Evaluates models trained on controlled lab data against external benchmark traffic
re-extracted with CICFlowMeter to assess real-world transfer and benign false positive rate (FPR).
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
CICIDS_DIR = DATA_DIR / "CICIDS2017_L6"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_hpo import (  # noqa: E402 -- reuse, don't reimplement
    load_split, load_feature_groups, load_split_flows, build_capped_train_pool,
    fit_model_with_class_weight,
)

MODELS = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
DAYS = ["tuesday", "wednesday", "friday"]
# v2 (2026-08-18): the one-time confirmation run against the class_weight-tuned
# config, per the plan agreed with Ishan -- L6 was not touched at all while
# choosing class_weight (that was decided purely from L2/LODO evidence). v1's
# own L6 result stays archived in reports/archive/2026-08-18_v1_run/.
FROZEN_CONFIG_NAME = "frozen_model_config_v2.json"
OUT_PATH = REPORT_DIR / "cicids2017_l6_results_v2.json"

def build_training_pool(feature_cols: list[str]):
    """Execute internal routine."""
    split = load_split()
    sessions = split["sessions"]
    print("Rebuilding Phase 2's training pool (deterministic replay, seed=42)...")
    train_natural = load_split_flows("train", sessions)
    train_pool = build_capped_train_pool(train_natural)
    print(f"  Capped train pool: {len(train_pool):,} flows (should match Phase 2's 99,670)")

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    y_train_str = train_pool["label"].to_numpy()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_str)
    groups_train = train_pool["session_id"].to_numpy()
    return X_train, y_train, groups_train, scaler, label_encoder

def load_cicids_eval_set(feature_cols: list[str]) -> pd.DataFrame:
    needed = feature_cols + ["label"]
    frames = []
    for day in DAYS:
        path = CICIDS_DIR / f"{day}_labelled.parquet"
        df = pd.read_parquet(path, columns=needed)
        df = df[df["label"] != "UNKNOWN"]
        print(f"  [{day}] {len(df):,} flows (direct-match classes only)")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

def benign_fpr(y_true_str: np.ndarray, y_pred_str: np.ndarray) -> float | None:
    is_benign = y_true_str == "BENIGN"
    n_benign = int(is_benign.sum())
    if n_benign == 0:
        return None
    false_positives = int(((y_pred_str[is_benign]) != "BENIGN").sum())
    return round(false_positives / n_benign, 4)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen = json.loads((CONFIGS_DIR / FROZEN_CONFIG_NAME).read_text(encoding="utf-8"))
    feature_cols = frozen["feature_columns"]
    frozen_models = frozen["models"]
    missing = [m for m in MODELS if m not in frozen_models]
    if missing:
        raise RuntimeError(f"{FROZEN_CONFIG_NAME} missing model(s) {missing}")

    print("Loading CICIDS2017 L6 evaluation set (Tuesday+Wednesday+Friday, direct-match classes)...")
    eval_df = load_cicids_eval_set(feature_cols)
    print(f"  Total: {len(eval_df):,} flows")
    print(eval_df["label"].value_counts().to_string())

    X_train, y_train, groups_train, scaler, label_encoder = build_training_pool(feature_cols)

    X_eval = scaler.transform(eval_df[feature_cols].to_numpy(dtype=np.float64))
    y_eval_str = eval_df["label"].to_numpy()
    # CICIDS2017's direct-match classes are a SUBSET of the trained label space
    # (see module docstring) -- transform() against the training-fit encoder,
    # not a fresh fit, so labels stay in the one true training encoding.
    y_eval = label_encoder.transform(y_eval_str)

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"\nResuming: {len(results.get('models', {}))} model(s) already done.")
    else:
        results = {
            "description": "L6 evaluation: External same-extractor benchmark evaluation on CICIDS2017."
,
            "eval_days": DAYS,
            "n_eval_flows": len(eval_df),
            "eval_label_distribution": eval_df["label"].value_counts().to_dict(),
            "models": {},
        }

    for name in MODELS:
        if name in results["models"]:
            print(f"\n[skip] {name} already has a result.")
            continue
        params = frozen_models[name]["best_hyperparameters"]
        print(f"\n{'=' * 70}\n{name}: refitting on Phase 2's training pool, params={params}\n{'=' * 70}", flush=True)
        t0 = time.time()
        clf, fell_back = fit_model_with_class_weight(name, params, X_train, y_train)
        fit_s = time.time() - t0

        y_pred = clf.predict(X_eval)
        y_pred_str = label_encoder.inverse_transform(y_pred)
        macro_f1 = f1_score(y_eval, y_pred, average="macro", zero_division=0)
        labels_sorted = sorted(set(y_train) | set(y_eval))
        report = classification_report(y_eval, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
        per_class_f1 = {label_encoder.classes_[c]: round(report[str(c)]["f1-score"], 4)
                         for c in labels_sorted if str(c) in report}
        fpr = benign_fpr(y_eval_str, y_pred_str)

        print(f"  {name}: macro-F1={macro_f1:.4f}  benign_FPR={fpr}  fit={fit_s:.0f}s"
              f"{'  [fell back to LogReg]' if fell_back else ''}", flush=True)
        for c, f1 in sorted(per_class_f1.items(), key=lambda kv: kv[1]):
            print(f"      {c:15s} F1={f1:.3f}")

        results["models"][name] = {
            "hyperparameters": params,
            "linearsvc_fell_back_to_logreg": fell_back if name == "LinearSVC" else None,
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "benign_fpr": fpr,
            "fit_seconds": round(fit_s, 1),
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"  [checkpoint] {name} saved to {OUT_PATH.name}")

    print(f"\n{'=' * 70}\nL6 EVALUATION COMPLETE. Written: {OUT_PATH}\n{'=' * 70}")
    print(f"{'model':10s} {'macro-F1':>10s} {'benign FPR':>11s}  vs targets (F1>=0.50, FPR<=40%)")
    for name in MODELS:
        m = results["models"][name]
        f1v = m["macro_f1"]
        fpr = m["benign_fpr"]
        verdict_f1 = "OK" if f1v >= 0.50 else "below"
        verdict_fpr = "OK" if (fpr is not None and fpr <= 0.40) else "above"
        print(f"{name:10s} {f1v:10.4f} {fpr if fpr is not None else 'n/a':>11}  F1:{verdict_f1}  FPR:{verdict_fpr}")

if __name__ == "__main__":
    main()
