"""
Automated Publication Figure Rendering.

Generates publication-ready figures including ROC curves, confusion matrices,
feature drift heatmaps, and generalization ladder performance comparisons.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, RobustScaler, label_binarize

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
REPORT_DIR = REPO_ROOT / "reports"
CICIDS_DIR = DATA_DIR / "CICIDS2017_L6"
OUT_DIR = REPO_ROOT / "latex" / "img" / "generated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2_hpo import (  # noqa: E402
    DOMAINS, NON_TRAINING_LABELS, build_capped_train_pool, fit_model_with_class_weight,
)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.titlesize": 12, "axes.labelsize": 10, "figure.facecolor": "white",
})

BEST_MODEL = "XGB"

def load_json(name):
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))

def load_frozen():
    return load_json_configs("frozen_model_config_v2.json")

def load_json_configs(name):
    return json.loads((CONFIGS_DIR / name).read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# 1. Dataset class distribution (per domain, per family) -- no model needed
# ---------------------------------------------------------------------------
def fig_dataset_distribution():
    print("[1/10] Dataset class distribution...")
    rows = []
    for d in DOMAINS:
        df = pd.read_parquet(DATA_DIR / d / "labelled.parquet", columns=["label"])
        counts = df["label"].value_counts()
        for label, n in counts.items():
            if label in NON_TRAINING_LABELS:
                continue
            rows.append({"domain": d, "label": label, "count": n})
    dist = pd.DataFrame(rows)
    pivot = dist.pivot(index="label", columns="domain", values="count").fillna(0)
    order = pivot.sum(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_yscale("log")
    ax.set_ylabel("Flow count (log scale)")
    ax.set_xlabel("Attack family / class")
    ax.set_title("Figure: Class distribution by domain (D01-D03, natural distribution)")
    ax.legend(title="Domain")
    plt.xticks(rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_dataset_distribution.png")
    plt.close()

# ---------------------------------------------------------------------------
# 2. Generalisation ladder comparison: L1 vs L4(LODO) vs L6, all 5 models
# ---------------------------------------------------------------------------
def fig_ladder_comparison():
    print("[2/10] Generalisation ladder comparison (L1 vs L4 vs L6)...")
    l1 = load_json("l1_random_split_results.json")
    l4 = load_json("lodo_results_v2.json")
    l6 = load_json("cicids2017_l6_results_v2.json")

    models = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
    l1_vals = [l1["models"][m]["macro_f1"] for m in models]

    l4_vals = []
    for m in models:
        fold_scores = [l4["folds"][f"{d}__{m}"]["macro_f1"] for d in DOMAINS if f"{d}__{m}" in l4["folds"]]
        l4_vals.append(np.mean(fold_scores) if fold_scores else np.nan)

    l6_vals = [l6["models"][m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, l1_vals, w, label="L1 random split (sanity, not trusted)", color="#8C8C8C")
    ax.bar(x, l4_vals, w, label="L4 LODO (domain holdout, the real number)", color="#4C72B0")
    ax.bar(x + w, l6_vals, w, label="L6 CICIDS2017 (external, cross-dataset)", color="#C44E52")
    ax.axhline(0.60, color="#4C72B0", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(0.50, color="#C44E52", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure: Generalisation ladder -- L1 vs L4 (LODO) vs L6 (external)")
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ladder_comparison.png")
    plt.close()
    return {"models": models, "l1": l1_vals, "l4": l4_vals, "l6": l6_vals}

# ---------------------------------------------------------------------------
# 3. LODO per-fold breakdown
# ---------------------------------------------------------------------------
def fig_lodo_per_fold():
    print("[3/10] LODO per-fold breakdown...")
    l4 = load_json("lodo_results_v2.json")
    models = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(DOMAINS))
    w = 0.15
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    for i, m in enumerate(models):
        vals = [l4["folds"][f"{d}__{m}"]["macro_f1"] for d in DOMAINS]
        ax.bar(x + (i - 2) * w, vals, w, label=m, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([f"held out: {d}" for d in DOMAINS])
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Figure: LODO macro-F1 per held-out domain")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_lodo_per_fold.png")
    plt.close()

# ---------------------------------------------------------------------------
# Shared: refit BEST_MODEL and collect real predictions for LODO (aggregated
# across all 3 folds) and for L6 (CICIDS2017), reusing frozen hyperparameters.
# ---------------------------------------------------------------------------
def load_domain_natural(domain, feature_cols):
    needed = feature_cols + ["label", "session_id"]
    df = pd.read_parquet(DATA_DIR / domain / "labelled.parquet", columns=needed)
    return df[~df["label"].isin(NON_TRAINING_LABELS)].copy()

def collect_lodo_predictions(feature_cols, frozen_models):
    print("    Refitting XGB per LODO fold to capture real predictions (reuses frozen hyperparameters)...")
    all_true, all_pred, all_proba, classes_seen = [], [], [], set()
    for held_out in DOMAINS:
        train_domains = [d for d in DOMAINS if d != held_out]
        train_natural = pd.concat([load_domain_natural(d, feature_cols) for d in train_domains], ignore_index=True)
        train_pool = build_capped_train_pool(train_natural)
        test_df = load_domain_natural(held_out, feature_cols)

        all_labels = sorted(set(train_pool["label"].unique()) | set(test_df["label"].unique()))
        le = LabelEncoder(); le.fit(all_labels)
        scaler = RobustScaler()
        X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
        y_train = le.transform(train_pool["label"].to_numpy())
        X_test = scaler.transform(test_df[feature_cols].to_numpy(dtype=np.float64))
        y_test = le.transform(test_df["label"].to_numpy())

        params = frozen_models[BEST_MODEL]["best_hyperparameters"]
        clf, _ = fit_model_with_class_weight(BEST_MODEL, params, X_train, y_train)
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

        all_true.append(le.inverse_transform(y_test))
        all_pred.append(le.inverse_transform(y_pred))
        classes_seen |= set(all_labels)
        if proba is not None:
            # store as (labels, proba, class_order) tuples per fold -- class order differs per fold
            all_proba.append((le.inverse_transform(y_test), proba, list(le.classes_)))
        print(f"      fold {held_out}: done ({len(y_test):,} test flows)")
    return np.concatenate(all_true), np.concatenate(all_pred), all_proba, sorted(classes_seen)

def collect_l6_predictions(feature_cols, frozen_models):
    print("    Refitting XGB on full D01-D03 pool, predicting on CICIDS2017 L6 (reuses frozen hyperparameters)...")
    from phase2_hpo import load_split, load_split_flows
    split = load_split()
    train_natural = load_split_flows("train", split["sessions"])
    train_pool = build_capped_train_pool(train_natural)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    le = LabelEncoder()
    y_train = le.fit_transform(train_pool["label"].to_numpy())

    needed = feature_cols + ["label"]
    frames = []
    for day in ["Tuesday", "Wednesday", "Friday"]:
        p = CICIDS_DIR / f"{day}_labelled.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=needed)
            frames.append(df[df["label"] != "UNKNOWN"])
    eval_df = pd.concat(frames, ignore_index=True)
    eval_df = eval_df[eval_df["label"].isin(le.classes_)]  # only direct-match classes the model knows
    X_test = scaler.transform(eval_df[feature_cols].to_numpy(dtype=np.float64))
    y_test = le.transform(eval_df["label"].to_numpy())

    params = frozen_models[BEST_MODEL]["best_hyperparameters"]
    clf, _ = fit_model_with_class_weight(BEST_MODEL, params, X_train, y_train)
    y_pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None
    y_true_str = le.inverse_transform(y_test)
    y_pred_str = le.inverse_transform(y_pred)
    return y_true_str, y_pred_str, proba, list(le.classes_)

def plot_confusion(y_true, y_pred, classes, title, fname):
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = cm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, label="Row-normalised proportion")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.close()

def plot_roc(proba_data, classes, title, fname, single_fold=True):
    fig, ax = plt.subplots(figsize=(7, 6))
    if single_fold:
        y_true_str, proba, class_order = proba_data
        y_bin = label_binarize(y_true_str, classes=class_order)
        for i, c in enumerate(class_order):
            if y_bin[:, i].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
            ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
    else:
        # aggregate across folds: macro-average by concatenating per-class scores where the class appears
        for c in classes:
            y_true_all, score_all = [], []
            for y_true_str, proba, class_order in proba_data:
                if c not in class_order:
                    continue
                idx = class_order.index(c)
                y_true_all.append((y_true_str == c).astype(int))
                score_all.append(proba[:, idx])
            if not y_true_all:
                continue
            y_true_cat = np.concatenate(y_true_all)
            score_cat = np.concatenate(score_all)
            if y_true_cat.sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_true_cat, score_cat)
            ax.plot(fpr, tpr, label=f"{c} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / fname)
    plt.close()

def fig_confusion_and_roc():
    print("[4-7/10] Confusion matrices + ROC curves (LODO aggregate + L6)...")
    frozen = load_frozen()
    feature_cols = frozen["feature_columns"]
    frozen_models = frozen["models"]

    y_true_l4, y_pred_l4, proba_l4, classes_l4 = collect_lodo_predictions(feature_cols, frozen_models)
    plot_confusion(y_true_l4, y_pred_l4, classes_l4,
                    f"Figure: Confusion matrix -- {BEST_MODEL}, LODO (all 3 folds aggregated)",
                    "fig_confusion_lodo.png")
    plot_roc(proba_l4, classes_l4, f"Figure: ROC curves -- {BEST_MODEL}, LODO (all 3 folds aggregated)",
              "fig_roc_lodo.png", single_fold=False)

    y_true_l6, y_pred_l6, proba_l6, classes_l6 = collect_l6_predictions(feature_cols, frozen_models)
    plot_confusion(y_true_l6, y_pred_l6, classes_l6,
                    f"Figure: Confusion matrix -- {BEST_MODEL}, L6 (CICIDS2017 external)",
                    "fig_confusion_l6.png")
    plot_roc((y_true_l6, proba_l6, classes_l6), classes_l6,
              f"Figure: ROC curves -- {BEST_MODEL}, L6 (CICIDS2017 external)",
              "fig_roc_l6.png", single_fold=True)
    return frozen, feature_cols, frozen_models

# ---------------------------------------------------------------------------
# 8. Feature importance (from the L6-trained XGB model)
# ---------------------------------------------------------------------------
def fig_feature_importance(feature_cols, frozen_models):
    print("[8/10] Feature importance...")
    from phase2_hpo import load_split, load_split_flows
    split = load_split()
    train_natural = load_split_flows("train", split["sessions"])
    train_pool = build_capped_train_pool(train_natural)
    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    le = LabelEncoder()
    y_train = le.fit_transform(train_pool["label"].to_numpy())
    params = frozen_models[BEST_MODEL]["best_hyperparameters"]
    clf, _ = fit_model_with_class_weight(BEST_MODEL, params, X_train, y_train)

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:15]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh([feature_cols[i] for i in order][::-1], importances[order][::-1], color="#4C72B0")
    ax.set_xlabel("Feature importance (gain-based)")
    ax.set_title(f"Figure: Top 15 feature importances -- {BEST_MODEL}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_feature_importance.png")
    plt.close()

# ---------------------------------------------------------------------------
# 9. Drift diagnostics: top PSI-shifted features
# ---------------------------------------------------------------------------
def fig_drift_psi():
    print("[9/10] Drift diagnostics (PSI)...")
    d = load_json("drift_diagnostics_v2.json")
    feature_drift = d["comparisons"]["lab_vs_cicids"]["feature_drift"]
    top10 = sorted(feature_drift, key=lambda f: f["psi"], reverse=True)[:10]
    names = [f["feature"] for f in top10]
    psis = [f["psi"] for f in top10]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names[::-1], psis[::-1], color="#C44E52")
    ax.axvline(0.25, color="black", linestyle="--", linewidth=1, label="PSI=0.25 (significant shift threshold)")
    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Figure: Top 10 most-shifted features, lab train vs CICIDS2017")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_drift_psi.png")
    plt.close()

# ---------------------------------------------------------------------------
# 10. Ensemble vs single-model comparison
# ---------------------------------------------------------------------------
def fig_ensemble_comparison(ladder):
    print("[10/10] Ensemble vs single-model comparison...")
    ens = load_json("ensemble_results_v2.json")
    models = ladder["models"]
    l4_vals = ladder["l4"]
    l6_vals = ladder["l6"]

    lodo_soft = ens["levels"]["L4"]["soft_voting"]["mean_macro_f1"]
    l6_soft = ens["levels"]["L6"]["soft_voting"]["macro_f1"]
    labels = models + ["Ensemble\n(soft voting)"]
    l4_all = l4_vals + [lodo_soft if lodo_soft is not None else np.nan]
    l6_all = l6_vals + [l6_soft if l6_soft is not None else np.nan]

    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, l4_all, w, label="L4 LODO", color="#4C72B0")
    ax.bar(x + w/2, l6_all, w, label="L6 CICIDS2017", color="#C44E52")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Macro-F1"); ax.set_ylim(0, 1.05)
    ax.set_title("Figure: Single models vs soft-voting ensemble")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_ensemble_comparison.png")
    plt.close()

def main():
    fig_dataset_distribution()
    ladder = fig_ladder_comparison()
    fig_lodo_per_fold()
    frozen, feature_cols, frozen_models = fig_confusion_and_roc()
    fig_feature_importance(feature_cols, frozen_models)
    fig_drift_psi()
    fig_ensemble_comparison(ladder)
    print(f"\nAll figures written to {OUT_DIR}")

if __name__ == "__main__":
    main()
