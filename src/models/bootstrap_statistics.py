"""
Session-Clustered Paired Bootstrap Hypothesis Testing.

Computes 95% confidence intervals and p-values using 1,000 paired bootstrap iterations
clustered by recording session to assess statistical significance across evaluation tiers.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "reports"
CICIDS_DIR = DATA_DIR / "CICIDS2017_L6"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_hpo import (  # noqa: E402
    load_split, load_feature_groups, load_split_flows, build_capped_train_pool,
    fit_model_with_class_weight, SEED,
)

MODELS = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
CONFIG_VERSIONS = ["v1", "v2"]
DAYS = ["tuesday", "wednesday", "friday"]
B_RESAMPLES = 1000
OUT_PATH = REPORT_DIR / "bootstrap_statistics.json"

def _weighted_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    # sample_weight-weighted macro-F1 is mathematically equivalent to
    # duplicating each row by its bootstrap resample count, but O(n_flows)
    # per resample with no row-index concatenation -- see module docstring
    # note on why this replaced the naive row-expansion approach.
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0, sample_weight=weights))

def _cluster_codes(cluster_ids: np.ndarray) -> tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(cluster_ids, sort=False)
    return codes, len(uniques)

def cluster_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray, cluster_ids: np.ndarray,
                          n_boot: int = B_RESAMPLES, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    row_cluster_code, n_clusters = _cluster_codes(cluster_ids)
    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)  # cluster indices sampled w/ replacement
        cluster_weight = np.bincount(sampled, minlength=n_clusters)
        row_weight = cluster_weight[row_cluster_code]
        scores[b] = _weighted_macro_f1(y_true, y_pred, row_weight)
    point = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {
        "point_estimate": round(point, 4),
        "ci_lo": round(float(np.percentile(scores, 2.5)), 4),
        "ci_hi": round(float(np.percentile(scores, 97.5)), 4),
        "n_clusters": int(n_clusters), "n_flows": int(len(y_true)),
    }

def paired_bootstrap_ci(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray, cluster_ids: np.ndarray,
                         n_boot: int = B_RESAMPLES, seed: int = SEED) -> dict:
    """Execute internal routine."""
    rng = np.random.default_rng(seed)
    row_cluster_code, n_clusters = _cluster_codes(cluster_ids)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        cluster_weight = np.bincount(sampled, minlength=n_clusters)
        row_weight = cluster_weight[row_cluster_code]
        f1_a = _weighted_macro_f1(y_true, y_pred_a, row_weight)
        f1_b = _weighted_macro_f1(y_true, y_pred_b, row_weight)
        deltas[b] = f1_a - f1_b
    point = float(f1_score(y_true, y_pred_a, average="macro", zero_division=0)
                  - f1_score(y_true, y_pred_b, average="macro", zero_division=0))
    ci_lo, ci_hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        "delta_point_estimate": round(point, 4), "delta_ci_lo": round(ci_lo, 4), "delta_ci_hi": round(ci_hi, 4),
        "statistically_meaningful": not (ci_lo <= 0 <= ci_hi),
    }

def build_train_pool_and_fit(name: str, params: dict, feature_cols: list[str]):
    split = load_split()
    train_natural = load_split_flows("train", split["sessions"])
    train_pool = build_capped_train_pool(train_natural)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_pool["label"].to_numpy())
    clf, fell_back = fit_model_with_class_weight(name, params, X_train, y_train)
    return clf, scaler, label_encoder, fell_back

def load_l2_val(feature_cols: list[str]):
    split = load_split()
    val_natural = load_split_flows("val", split["sessions"])
    return val_natural

def load_l6_eval(feature_cols: list[str]):
    needed = feature_cols + ["label", "src_ip", "dst_ip"]
    frames = []
    for day in DAYS:
        df = pd.read_parquet(CICIDS_DIR / f"{day}_labelled.parquet", columns=needed)
        frames.append(df[df["label"] != "UNKNOWN"])
    df = pd.concat(frames, ignore_index=True)
    a = df["src_ip"].astype(str)
    b = df["dst_ip"].astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    df["host_pair"] = pd.Series(lo) + "__" + pd.Series(hi)
    return df

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if OUT_PATH.exists():
        results = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"Resuming: {len(results.get('runs', {}))} (config,model) run(s) already done.")
    else:
        results = {
            "description": "Session-clustered paired bootstrap testing (B=1000) for statistical significance."
,
            "b_resamples": B_RESAMPLES,
            "runs": {},
        }

    # Load eval sets ONCE per feature set (v1/v2 use the same 25 Group A+B
    # features -- confirmed by comparing feature_columns below before reuse).
    configs = {v: json.loads((CONFIGS_DIR / f"frozen_model_config_{v}.json").read_text(encoding="utf-8"))
               for v in CONFIG_VERSIONS}
    feat_v1, feat_v2 = configs["v1"]["feature_columns"], configs["v2"]["feature_columns"]
    if feat_v1 != feat_v2:
        raise RuntimeError("v1/v2 feature_columns differ -- eval-set caching below assumes they match; "
                            "fix before trusting these results.")
    feature_cols = feat_v1

    print("Loading L2 (val) and L6 (CICIDS2017) eval sets once...")
    val_df = load_l2_val(feature_cols)
    l6_df = load_l6_eval(feature_cols)
    print(f"  L2 val: {len(val_df):,} flows, {val_df['session_id'].nunique()} sessions")
    print(f"  L6 CICIDS2017: {len(l6_df):,} flows, {l6_df['host_pair'].nunique()} host-pairs")

    predictions = {}  # (version, model) -> dict(y_true_l2, y_pred_l2, sessions, y_true_l6, y_pred_l6, host_pairs)

    for version in CONFIG_VERSIONS:
        frozen = configs[version]
        for name in MODELS:
            key = f"{version}__{name}"
            if key in results["runs"]:
                print(f"[skip] {key} already has a result.")
                continue
            params = frozen["models"][name]["best_hyperparameters"]
            print(f"\n{'=' * 70}\n{key}: refitting, params={params}\n{'=' * 70}", flush=True)
            t0 = time.time()
            clf, scaler, label_encoder, fell_back = build_train_pool_and_fit(name, params, feature_cols)
            fit_s = time.time() - t0

            X_val = scaler.transform(val_df[feature_cols].to_numpy(dtype=np.float64))
            y_val_true = label_encoder.transform(val_df["label"].to_numpy())
            y_val_pred = clf.predict(X_val)
            l2_ci = cluster_bootstrap_ci(y_val_true, y_val_pred, val_df["session_id"].to_numpy())

            X_l6 = scaler.transform(l6_df[feature_cols].to_numpy(dtype=np.float64))
            y_l6_true = label_encoder.transform(l6_df["label"].to_numpy())
            y_l6_pred = clf.predict(X_l6)
            l6_ci = cluster_bootstrap_ci(y_l6_true, y_l6_pred, l6_df["host_pair"].to_numpy())

            print(f"  L2 macro-F1: {l2_ci['point_estimate']:.4f}  95% CI [{l2_ci['ci_lo']:.4f}, {l2_ci['ci_hi']:.4f}]  "
                  f"({l2_ci['n_clusters']} sessions)")
            print(f"  L6 macro-F1: {l6_ci['point_estimate']:.4f}  95% CI [{l6_ci['ci_lo']:.4f}, {l6_ci['ci_hi']:.4f}]  "
                  f"({l6_ci['n_clusters']} host-pairs)  fit={fit_s:.0f}s", flush=True)

            results["runs"][key] = {"config_version": version, "model": name, "fit_seconds": round(fit_s, 1),
                                     "l2_bootstrap": l2_ci, "l6_bootstrap": l6_ci}
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(f"  [checkpoint] {key} saved to {OUT_PATH.name}")

            predictions[(version, name)] = {
                "y_l2_true": y_val_true, "y_l2_pred": y_val_pred, "sessions": val_df["session_id"].to_numpy(),
                "y_l6_true": y_l6_true, "y_l6_pred": y_l6_pred, "host_pairs": l6_df["host_pair"].to_numpy(),
            }

    # Paired v1-vs-v2 comparison per model, on L6 (the metric that actually
    # changed meaningfully between configs) -- resolves whether the
    # differences observed in the raw evaluate_cicids2017.py runs are real
    # or within bootstrap noise.
    print(f"\n{'=' * 70}\nPaired v1-vs-v2 comparison (L6 macro-F1)\n{'=' * 70}")
    if "paired_v1_vs_v2_l6" not in results:
        results["paired_v1_vs_v2_l6"] = {}
    for name in MODELS:
        pkey = name
        if pkey in results["paired_v1_vs_v2_l6"]:
            print(f"[skip] {pkey} already compared.")
            continue
        if (("v1", name) not in predictions) or (("v2", name) not in predictions):
            print(f"[skip] {pkey} -- predictions not in memory this run (resumed from checkpoint); "
                  f"re-run without deleting {OUT_PATH.name} to compare in the same process.")
            continue
        p1, p2 = predictions[("v1", name)], predictions[("v2", name)]
        # Same eval rows/order for v1 and v2 (same feature_cols, same eval-set load) -- paired directly.
        cmp = paired_bootstrap_ci(p1["y_l6_true"], p1["y_l6_pred"], p2["y_l6_pred"], p1["host_pairs"])
        print(f"  {name}: v1-v2 delta={cmp['delta_point_estimate']:+.4f}  "
              f"95% CI [{cmp['delta_ci_lo']:+.4f}, {cmp['delta_ci_hi']:+.4f}]  "
              f"meaningful={cmp['statistically_meaningful']}")
        results["paired_v1_vs_v2_l6"][pkey] = cmp
        OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'=' * 70}\nBOOTSTRAP STATISTICS COMPLETE. Written: {OUT_PATH}\n{'=' * 70}")

if __name__ == "__main__":
    main()
