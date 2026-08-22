"""
Feature Distribution Drift Diagnostics.

Computes Population Stability Index (PSI), Kolmogorov-Smirnov statistics, and Wasserstein distance
across domain pairs to identify features most vulnerable to network environment shifts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "reports"
CICIDS_DIR = DATA_DIR / "CICIDS2017_L6"
V1_TRAIN_PARQUET = REPO_ROOT / "OLD EXPIRED RESEARCH" / "data" / "processed" / "train.parquet"

sys.path.insert(0, str(REPO_ROOT / "src" / "models"))
from phase2_hpo import load_split, load_split_flows, build_capped_train_pool, SEED  # noqa: E402

OUT_PATH = REPORT_DIR / "drift_diagnostics_v2.json"
DAYS = ["tuesday", "wednesday", "friday"]
NON_TRAINING_LABELS = {"UNKNOWN", "CONFLICT"}

def psi(reference: np.ndarray, comparison: np.ndarray, bins: int = 10) -> float:
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=edges)
    cmp_counts, _ = np.histogram(comparison, bins=edges)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-6, None)
    cmp_pct = np.clip(cmp_counts / max(len(comparison), 1), 1e-6, None)
    return float(np.sum((cmp_pct - ref_pct) * np.log(cmp_pct / ref_pct)))

def feature_drift_table(ref_df: pd.DataFrame, cmp_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    rows = []
    for f in feature_cols:
        ref = ref_df[f].to_numpy(dtype=np.float64)
        cmp = cmp_df[f].to_numpy(dtype=np.float64)
        ref = ref[np.isfinite(ref)]
        cmp = cmp[np.isfinite(cmp)]
        if len(ref) < 2 or len(cmp) < 2:
            continue
        ks_stat, ks_p = ks_2samp(ref, cmp)
        rng = np.ptp(ref) or 1.0
        wd = wasserstein_distance(ref, cmp) / rng
        p = psi(ref, cmp)
        rows.append({
            "feature": f, "ks_statistic": round(float(ks_stat), 4), "ks_pvalue": float(ks_p),
            "wasserstein_normalised": round(float(wd), 4), "psi": round(float(p), 4),
            "psi_flag": "significant" if p > 0.2 else ("moderate" if p > 0.1 else "none"),
        })
    return sorted(rows, key=lambda r: -r["psi"])

def domain_classifier_auc(a_df: pd.DataFrame, b_df: pd.DataFrame, feature_cols: list[str],
                           max_per_side: int = 100_000) -> dict:
    # Subsample the larger side -- CICIDS2017's eval set is ~1.7M rows; combined
    # with the lab pool, cross_val_predict on the full set risks the same
    # memory-fragmentation failure mode already hit twice tonight (LODO's D04
    # fold, the C: drive near-crisis). The classifier's job is just to measure
    # distinguishability, not to see every flow -- 100k/side is ample for a
    # stable AUC estimate.
    if len(a_df) > max_per_side:
        a_df = a_df.sample(n=max_per_side, random_state=SEED)
    if len(b_df) > max_per_side:
        b_df = b_df.sample(n=max_per_side, random_state=SEED)
    X = pd.concat([a_df[feature_cols], b_df[feature_cols]], ignore_index=True).to_numpy(dtype=np.float64)
    y = np.array([0] * len(a_df) + [1] * len(b_df))
    finite = np.isfinite(X).all(axis=1)
    X, y = X[finite], y[finite]
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=SEED, n_jobs=-1, class_weight="balanced")
    proba = cross_val_predict(clf, X, y, cv=3, method="predict_proba", n_jobs=-1)[:, 1]
    auc = roc_auc_score(y, proba)
    clf.fit(X, y)
    importances = sorted(zip(feature_cols, clf.feature_importances_.tolist()), key=lambda kv: -kv[1])
    return {
        "auc": round(float(auc), 4),
        "distinguishable": auc > 0.70,
        "top_10_features_by_importance": [{"feature": f, "importance": round(v, 4)} for f, v in importances[:10]],
    }

def load_lab_capped_pool(feature_cols: list[str]) -> pd.DataFrame:
    split = load_split()
    train_natural = load_split_flows("train", split["sessions"])
    return build_capped_train_pool(train_natural)

def load_cicids_direct(feature_cols: list[str]) -> pd.DataFrame:
    needed = feature_cols + ["label"]
    frames = []
    for day in DAYS:
        df = pd.read_parquet(CICIDS_DIR / f"{day}_labelled.parquet", columns=needed)
        frames.append(df[df["label"] != "UNKNOWN"])
    return pd.concat(frames, ignore_index=True)

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    frozen = json.loads((CONFIGS_DIR / "frozen_model_config_v2.json").read_text(encoding="utf-8"))
    feature_cols = frozen["feature_columns"]

    print("Loading lab training pool (D01-D04 capped, deterministic replay)...")
    lab_pool = load_lab_capped_pool(feature_cols)
    print(f"  {len(lab_pool):,} flows")

    print("Loading CICIDS2017 direct-match eval set (Tue/Wed/Fri)...")
    cicids = load_cicids_direct(feature_cols)
    print(f"  {len(cicids):,} flows")

    results = {"description": "Feature distribution drift diagnostics: Population Stability Index and Wasserstein distance."
, "comparisons": {}}

    print("\n[1/5] Lab train vs CICIDS2017 -- primary comparison")
    results["comparisons"]["lab_vs_cicids"] = {
        "n_a": len(lab_pool), "n_b": len(cicids),
        "feature_drift": feature_drift_table(lab_pool, cicids, feature_cols),
    }
    top5 = results["comparisons"]["lab_vs_cicids"]["feature_drift"][:5]
    print("  Top-5 most-shifted features (by PSI):")
    for r in top5:
        print(f"    {r['feature']:25s} PSI={r['psi']:.3f} ({r['psi_flag']})  KS={r['ks_statistic']:.3f}  Wasserstein={r['wasserstein_normalised']:.3f}")

    print("\n[2/5] D01 vs D02 -- within-lab baseline")
    d01 = pd.read_parquet(DATA_DIR / "D01" / "labelled.parquet", columns=feature_cols + ["label"])
    d01 = d01[~d01["label"].isin(NON_TRAINING_LABELS)]
    d02 = pd.read_parquet(DATA_DIR / "D02" / "labelled.parquet", columns=feature_cols + ["label"])
    d02 = d02[~d02["label"].isin(NON_TRAINING_LABELS)]
    results["comparisons"]["d01_vs_d02"] = {
        "n_a": len(d01), "n_b": len(d02),
        "feature_drift": feature_drift_table(d01, d02, feature_cols),
    }
    print(f"  Median PSI (within-lab): {np.median([r['psi'] for r in results['comparisons']['d01_vs_d02']['feature_drift']]):.4f}")
    print(f"  Median PSI (lab-vs-CICIDS): {np.median([r['psi'] for r in results['comparisons']['lab_vs_cicids']['feature_drift']]):.4f}")

    print("\n[3/5] Domain classifier: lab (0) vs CICIDS2017 (1)")
    dc = domain_classifier_auc(lab_pool, cicids, feature_cols)
    results["comparisons"]["domain_classifier_lab_vs_cicids"] = dc
    print(f"  AUC={dc['auc']:.4f}  distinguishable={dc['distinguishable']}")
    print("  Top drivers:", ", ".join(f"{d['feature']}({d['importance']:.3f})" for d in dc["top_10_features_by_importance"][:5]))

    if V1_TRAIN_PARQUET.exists():
        v1 = pd.read_parquet(V1_TRAIN_PARQUET)
        overlap_cols = [c for c in feature_cols if c in v1.columns]
        missing_cols = [c for c in feature_cols if c not in v1.columns]
        print(f"\n[4/5] V2 vs V1 lab data -- {len(overlap_cols)}/{len(feature_cols)} features overlap "
              f"(V1 used NFStream, not CICFlowMeter; {len(missing_cols)} V2-only/raw-passthrough features have no V1 equivalent)")
        results["comparisons"]["v2_vs_v1_lab"] = {
            "overlap_feature_count": len(overlap_cols), "missing_from_v1": missing_cols,
            "n_a": len(lab_pool), "n_b": len(v1),
            "feature_drift": feature_drift_table(lab_pool, v1, overlap_cols),
        }

        print(f"\n[5/5] V1 lab data vs CICIDS2017 -- V1's own feature-drift gap, for comparison")
        results["comparisons"]["v1_vs_cicids"] = {
            "overlap_feature_count": len(overlap_cols), "missing_from_v1": missing_cols,
            "n_a": len(v1), "n_b": len(cicids),
            "feature_drift": feature_drift_table(v1, cicids, overlap_cols),
        }
        v1_median_psi = np.median([r["psi"] for r in results["comparisons"]["v1_vs_cicids"]["feature_drift"]])
        v2_median_psi_overlap = np.median([r["psi"] for r in feature_drift_table(lab_pool, cicids, overlap_cols)])
        print(f"  V1-vs-CICIDS median PSI: {v1_median_psi:.4f}")
        print(f"  V2-vs-CICIDS median PSI (same {len(overlap_cols)}-feature subset): {v2_median_psi_overlap:.4f}")
    else:
        print("\n[4/5, 5/5] SKIPPED -- V1 train.parquet not found at", V1_TRAIN_PARQUET)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten: {OUT_PATH}")

if __name__ == "__main__":
    main()
