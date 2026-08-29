"""
Leave-One-Domain-Out (LODO) Comprehensive Cross-Validation.

Evaluates all five benchmark classifiers (LR, LinearSVC, RF, XGBoost, LightGBM) across
four domains using frozen hyperparameters, testing strictly on held-out natural distributions.
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
from phase2_hpo import (  # noqa: E402 -- reuse the already-debugged Phase 2 machinery
    DOMAINS, NON_TRAINING_LABELS, PER_CLASS_CAP, SEED,
    build_capped_train_pool, fit_model_with_class_weight,
)

MODELS = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
# v2 (2026-08-18): points at the class_weight-tuned config; v1's own LODO
# result stays archived in reports/archive/2026-08-18_v1_run/ untouched.
FROZEN_CONFIG_NAME = "frozen_model_config_v2.json"
OUT_PATH = REPORT_DIR / "lodo_results_v2.json"

def load_frozen_config() -> dict:
    return json.loads((CONFIGS_DIR / FROZEN_CONFIG_NAME).read_text(encoding="utf-8"))

def load_domain_natural(domain: str, feature_cols: list[str]) -> pd.DataFrame:
    # FOUND 2026-08-18: the D04-held-out fold (train pool = D01+D02+D03,
    # 6,400,439 pooled rows -- the largest of the 4 fold combinations) hit
    # numpy._core._exceptions._ArrayMemoryError: "Unable to allocate 3.19 GiB
    # for an array with shape (76, 5627944)" inside stratified_cap()'s
    # df_class.drop(capped.index) call. Root cause: this function used to
    # return ALL of labelled.parquet's ~76-86 raw+metadata columns, so every
    # pandas operation downstream (groupby, concat, drop/reindex) had to
    # allocate/copy blocks sized by the FULL column count, not just the 25
    # frozen features actually used. pandas' BlockManager consolidates
    # same-dtype columns into one contiguous (n_cols, n_rows) block for
    # operations like .drop()/reindex -- a single ~3.2GB contiguous
    # allocation request failed even with 7.8GB nominally free (Windows
    # memory fragmentation after 3 completed folds' worth of large
    # intermediate DataFrames in the same long-running process, not true
    # exhaustion). Fixed by selecting only what's actually needed --
    # feature_cols (25) + label + session_id (stratified_cap's own grouping
    # key) -- immediately after reading, before any concat/cap operation,
    # cutting the per-row column count (and every subsequent block
    # allocation) by roughly 3x. Same "verify before trusting a `del`"
    # lesson as this project's own DOS_HULK OOM history -- freeing a
    # reference doesn't help if the live object was needlessly wide in the
    # first place.
    needed = feature_cols + ["label", "session_id"]
    df = pd.read_parquet(DATA_DIR / domain / "labelled.parquet", columns=needed)
    df = df[~df["label"].isin(NON_TRAINING_LABELS)].copy()
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{domain}: labelled.parquet missing frozen feature column(s): {missing}")
    return df

def run_fold(held_out: str, feature_cols: list[str], frozen_models: dict, results: dict) -> None:
    train_domains = [d for d in DOMAINS if d != held_out]
    print(f"\n{'=' * 70}\nLODO fold: held-out={held_out}  (train on {train_domains})\n{'=' * 70}", flush=True)

    print("  Loading + pooling natural data from the 3 training domains...", flush=True)
    train_natural = pd.concat([load_domain_natural(d, feature_cols) for d in train_domains], ignore_index=True)
    print(f"  Pooled natural train flows: {len(train_natural):,}")
    print(f"  Building session-stratified capped pool (cap={PER_CLASS_CAP:,}/class, seed={SEED})...", flush=True)
    train_pool = build_capped_train_pool(train_natural)
    print(f"  Capped train pool: {len(train_pool):,} flows")
    del train_natural

    print(f"  Loading held-out domain {held_out}'s full natural distribution for testing...", flush=True)
    test_df = load_domain_natural(held_out, feature_cols)
    print(f"  Test flows ({held_out}, natural): {len(test_df):,}")

    # Fit label encoder on the UNION of train-pool + held-out labels (see
    # module docstring) -- avoids a transform() crash if the held-out domain
    # has a class the other three don't.
    all_labels = sorted(set(train_pool["label"].unique()) | set(test_df["label"].unique()))
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    y_train = label_encoder.transform(train_pool["label"].to_numpy())
    X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float64))
    y_test = label_encoder.transform(test_df["label"].to_numpy())
    del train_pool, test_df

    for name in MODELS:
        key = f"{held_out}__{name}"
        if key in results["folds"]:
            print(f"  [skip] {key} already has a result -- not re-running.", flush=True)
            continue
        params = frozen_models[name]["best_hyperparameters"]
        print(f"\n  Fitting {name} (frozen params={params})...", flush=True)
        t0 = time.time()
        clf, fell_back = fit_model_with_class_weight(name, params, X_train, y_train)
        fit_s = time.time() - t0

        y_pred = clf.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        labels_sorted = sorted(set(y_train) | set(y_test))
        report = classification_report(y_test, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
        per_class_f1 = {label_encoder.classes_[c]: round(report[str(c)]["f1-score"], 4)
                         for c in labels_sorted if str(c) in report}

        print(f"    {name} ({held_out} held out): macro-F1={macro_f1:.4f}  fit={fit_s:.0f}s"
              f"{'  [fell back to LogReg]' if fell_back else ''}", flush=True)
        for c, f1 in sorted(per_class_f1.items(), key=lambda kv: kv[1]):
            print(f"        {c:15s} F1={f1:.3f}  support={int(report[str(label_encoder.transform([c])[0])]['support'])}")

        results["folds"][key] = {
            "held_out_domain": held_out,
            "model": name,
            "hyperparameters": params,
            "linearsvc_fell_back_to_logreg": fell_back if name == "LinearSVC" else None,
            "n_train_flows_capped": int(len(X_train)),
            "n_test_flows_natural": int(len(X_test)),
            "macro_f1": round(macro_f1, 4),
            "per_class_f1": per_class_f1,
            "fit_seconds": round(fit_s, 1),
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"    [checkpoint] {key} saved to {OUT_PATH.name}", flush=True)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen = load_frozen_config()
    feature_cols = frozen["feature_columns"]
    frozen_models = frozen["models"]
    missing_models = [m for m in MODELS if m not in frozen_models]
    if missing_models:
        raise RuntimeError(f"{FROZEN_CONFIG_NAME} is missing model(s) {missing_models} -- "
                            f"Phase 2 must be fully FROZEN before LODO can run.")

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"Resuming: {len(results.get('folds', {}))} (domain, model) result(s) already in {OUT_PATH.name}.")
    else:
        results = {
            "description": "Full Leave-One-Domain-Out (LODO) 4-fold cross-validation across all 5 models."
f"config (configs/{FROZEN_CONFIG_NAME}). Generated by "
                        "src/models/run_lodo_evaluation.py.",
            "feature_columns": feature_cols,
            "per_class_cap": PER_CLASS_CAP,
            "seed": SEED,
            "folds": {},
        }

    for held_out in DOMAINS:
        run_fold(held_out, feature_cols, frozen_models, results)

    print(f"\n{'=' * 70}\nLODO COMPLETE. Written: {OUT_PATH}\n{'=' * 70}")
    for name in MODELS:
        fold_scores = [results["folds"][f"{d}__{name}"]["macro_f1"] for d in DOMAINS
                        if f"{d}__{name}" in results["folds"]]
        if len(fold_scores) == len(DOMAINS):
            # Section 17: "For leave-one-domain-out (4 folds only), report mean +/- SD
            # across folds. Bootstrap is not applied (degenerate with 4 observations)."
            mean_f1 = float(np.mean(fold_scores))
            sd_f1 = float(np.std(fold_scores, ddof=1))
            verdict = "MEETS" if mean_f1 >= 0.60 else "DOES NOT MEET"
            print(f"{name:10s} LODO macro-F1 = {mean_f1:.4f} +/- {sd_f1:.4f}  ({verdict} Section 19's 0.60 threshold)")
            print(f"           per-fold: " + "  ".join(f"{d}={results['folds'][f'{d}__{name}']['macro_f1']:.4f}" for d in DOMAINS))
            results.setdefault("summary", {})[name] = {"mean_macro_f1": round(mean_f1, 4), "sd_macro_f1": round(sd_f1, 4)}

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
