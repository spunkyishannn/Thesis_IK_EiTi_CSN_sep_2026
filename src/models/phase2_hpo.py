"""
Hyperparameter Optimization and Configuration Freeze Pipeline.

Executes Optuna-driven 5-fold cross-validation across five classifier architectures
to identify optimal regularization and structural parameters on primary training data.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
LAB_SCRIPTS = REPO_ROOT / "lab" / "scripts"
sys.path.insert(0, str(LAB_SCRIPTS))
from build_training_set import stratified_cap  # noqa: E402 -- reuse, don't reimplement

DOMAINS = ["D01", "D02", "D03"]  # TEMP 2026-08-22: see build_primary_split.py's matching note
NON_TRAINING_LABELS = {"UNKNOWN", "CONFLICT"}
N_TRIALS = 50
N_FOLDS = 5
SEED = 42
PER_CLASS_CAP = 25000
LINEARSVC_ROW_LIMIT = 200_000  # Section 15's own stated constraint

# FOUND 2026-08-17: LR/LinearSVC fits are single-threaded per call (no
# internal n_jobs) -- confirmed by timing a single LR fit directly (121s,
# still hitting max_iter=1000 without converging even after removing the two
# exact-duplicate features). At that rate, sequential trials would take
# LR alone ~8+ hours (50 trials x 5 folds x ~24s/fold-equivalent). Running
# multiple Optuna TRIALS concurrently for these two models only is a free,
# legitimate speedup on a 6-core/12-thread machine (checked via
# Get-CimInstance Win32_Processor) -- it changes nothing about WHAT is
# computed (same 50 trials, same 5-fold GroupKFold, same search space, same
# solver/max_iter), only how they're scheduled. RF/XGB/LGBM already
# parallelize INTERNALLY per fit via their own n_jobs=-1 -- running multiple
# Optuna trials for those concurrently as well would oversubscribe the same
# 12 threads and likely lose time to contention, so those stay sequential
# (n_jobs=1).
OPTUNA_N_JOBS = {"LR": 6, "LinearSVC": 6, "RF": 1, "XGB": 1, "LGBM": 1}

# FOUND 2026-08-17, decided with Ishan (not a silent shortcut): direct
# diagnostic (fit LR at C=0.001, 1.0, 100.0 -- the full spec'd search range
# -- on the same train/val fold) showed BIT-IDENTICAL predictions at every
# C: same macro-F1 (0.4219), same n_iter_=[1000] (never converges), same
# per-class prediction counts down to the exact number. Not floating-point
# coincidence -- real evidence that at max_iter=1000, saga hasn't run long
# enough for different regularisation strengths to produce different
# optimisation trajectories, so C is non-informative for this model at this
# iteration budget. Confirmed independently by the first 12 real HPO trials
# (n_jobs=6, TPE-sampled C values spanning the range) all returning the
# identical CV score 0.2548. Running the remaining 38 trials would reproduce
# this with no new information. LR is frozen at C=1.0 (sklearn's own
# default) with a single fit instead of the full 50-trial search --
# documented here and in the frozen config's per-model "note" field, not
# hidden.
#
# UPDATE 2026-08-17, same session: LinearSVC hit the IDENTICAL failure mode,
# for a directly explainable reason -- Section 15's own documented fallback
# ("If convergence fails -> replace with LogisticRegression(solver='saga',
# max_iter=2000)") fired on every single real HPO trial (liblinear never
# converged), meaning "LinearSVC" was in practice ALWAYS actually running
# the LR-family fallback. Direct diagnostic confirmed it: fit at C=0.001,
# 1.0, 100.0 all produced fell_back=True and the identical macro-F1=0.4379.
# Same root cause as LR (saga not converging within the given max_iter,
# independent of C), just with max_iter=2000 instead of 1000 and still not
# enough. Frozen the same way, same reasoning, same precedent already
# agreed for LR -- not a new judgment call, applying the one already made.
# UPDATED 2026-08-18 (v2 run): C is still fixed at 1.0 for LR/LinearSVC (the
# 2026-08-17 evidence above -- C is provably non-informative at this
# max_iter budget -- is untouched by adding class_weight, a different axis
# entirely). What changed is MODEL_N_TRIALS: 1 -> 2, enqueuing BOTH
# class_weight=None (reproduces the v1 result exactly, as a built-in
# regression check) and class_weight='balanced' (the new v2 hypothesis,
# motivated by PORT_SCAN/SSH_BRUTE's genuine training-set rarity -- see
# CLASS_WEIGHT_OPTIONS comment above). This is a targeted 2-point
# comparison, not a full 50-trial re-search, because C is already known not
# to matter -- there is nothing else in this 2-dimensional (C, class_weight)
# space worth exploring beyond the one new dimension.
MODEL_N_TRIALS = {"LR": 2, "LinearSVC": 2, "RF": N_TRIALS, "XGB": N_TRIALS, "LGBM": N_TRIALS}
LR_FIXED_C = 1.0
LINEARSVC_FIXED_C = 1.0

def self_note(name: str) -> str | None:
    if name == "LR":
        return ("Full HPO search skipped (2026-08-17 evidence: saga does not converge "
                "within max_iter=1000 regardless of C -- C=0.001/1.0/100.0 produced "
                "bit-identical predictions). v2 (2026-08-18): C fixed at 1.0, but now "
                "compares class_weight=None vs 'balanced' (2 fits) -- motivated by "
                "PORT_SCAN (3,200 flows) and SSH_BRUTE (1,423 flows) being genuinely rare "
                "in the 99,670-flow capped training pool, and V1's own frozen models all "
                "having used class_weight='balanced' where V2's original Section 15 respec "
                "dropped it undocumented. Whichever class_weight wins on CV macro-F1 is "
                "kept. See problems_notes.txt 2026-08-18.")
    if name == "LinearSVC":
        return ("Full HPO search skipped for the same reason as LR (2026-08-17): Section "
                "15's own convergence fallback fired on every real trial, meaning this "
                "model was in practice always running the LR-family fallback (identical "
                "macro-F1=0.4379 at C=0.001/1.0/100.0). v2 (2026-08-18): same "
                "class_weight=None vs 'balanced' comparison as LR, same motivation "
                "(PORT_SCAN/SSH_BRUTE rarity), C still fixed at 1.0.")
    return None

optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_split() -> dict:
    return json.loads((CONFIGS_DIR / "primary_split_v1.json").read_text(encoding="utf-8"))

EXACT_DUPLICATE_THRESHOLD = 0.999  # stricter than Phase 1's 0.95 advisory flag -- see below

def load_feature_groups() -> list[str]:
    fd = yaml.safe_load((CONFIGS_DIR / "feature_dictionary_v2.yaml").read_text(encoding="utf-8"))
    feats = [name for name, info in fd["features"].items() if info["group"] in ("A_volumetric", "B_packet_size")]
    excluded = [f for f in feats if fd["features"][f]["category"] == "EXCLUDED"]
    if excluded:
        raise RuntimeError(f"Group A/B contains EXCLUDED feature(s), Phase 1 result changed "
                            f"unexpectedly: {excluded}")
    feats = set(feats)

    # FOUND 2026-08-17: a real, evidence-based fix, not a guess -- the first
    # full HPO attempt took 539s for a SINGLE LR trial (5-fold CV), which
    # would have made 50 trials take ~7.5 HOURS for LR alone. Traced to
    # Phase 1's own high_correlation_pairs: two pairs within Group A+B are
    # EXACT duplicates at r=1.0 (fwd_pkt_len_mean == Fwd Segment Size Avg;
    # bwd_pkt_len_mean == Bwd Segment Size Avg -- CICFlowMeter's renamed and
    # raw_passthrough columns for the same underlying quantity, see
    # lab/feature_schema_v2.yaml's own note about the two naming conventions
    # coexisting). Feeding a solver two perfectly collinear columns produces
    # a rank-deficient design matrix -- a well-known cause of exactly this
    # kind of saga non-convergence/slowness, and the most likely dominant
    # cause given every single trial hit max_iter without converging.
    # Deliberately using a STRICTER threshold (0.999) here than Phase 1's
    # 0.95 advisory-flag threshold: this drops only the two mathematically-
    # exact duplicates, not the merely-highly-correlated pairs (e.g.
    # flow_pkts_s vs fwd_pkts_s at r=0.9967, or Packet Length Max vs Std at
    # r=0.96) -- those remain distinct signal and dropping them would be a
    # separate, more aggressive feature-selection judgment call that
    # shouldn't be made silently while fixing a performance bug. When both
    # members of a >=0.999 pair are present, the RENAMED (canonical,
    # CICIDS_RENAME) column is kept and the raw_passthrough duplicate is
    # dropped, since the rest of the pipeline already treats the renamed
    # name as canonical.
    high_corr = fd.get("high_correlation_pairs", [])
    dropped = []
    for p in high_corr:
        a, b, r = p["feature_a"], p["feature_b"], abs(p["r"])
        if a in feats and b in feats and r >= EXACT_DUPLICATE_THRESHOLD:
            to_drop = b if "_" in a and a.islower() else a  # crude but correct here: keep the snake_case renamed name
            feats.discard(to_drop)
            dropped.append((a, b, r, to_drop))
    if dropped:
        print(f"  Dropping {len(dropped)} exact-duplicate feature(s) (r>={EXACT_DUPLICATE_THRESHOLD}, "
              f"see phase2_hpo.py comment for why):")
        for a, b, r, to_drop in dropped:
            print(f"    {a} <-> {b} (r={r}) -> dropped '{to_drop}'")

    return sorted(feats)

def load_split_flows(split_name: str, session_split: dict) -> pd.DataFrame:
    frames = []
    for domain in DOMAINS:
        df = pd.read_parquet(DATA_DIR / domain / "labelled.parquet")
        df = df[~df["label"].isin(NON_TRAINING_LABELS)]
        wanted = {sid for sid, meta in session_split.items() if meta["domain"] == domain and meta["split"] == split_name}
        df = df[df["session_id"].isin(wanted)].copy()
        frames.append(df)
        print(f"  [{domain}/{split_name}] {len(df):,} flows from {len(wanted)} session(s)")
    return pd.concat(frames, ignore_index=True)

def build_capped_train_pool(train_df: pd.DataFrame) -> pd.DataFrame:
    kept = []
    for label, g in train_df.groupby("label"):
        capped = stratified_cap(g, PER_CLASS_CAP, SEED)
        kept.append(capped)
    pool = pd.concat(kept, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return pool

# ADDED 2026-08-18 (v2 run, after Ishan flagged the L6 macro-F1 as not
# thesis-strength and asked for a real fix): checked BEFORE touching
# anything -- PORT_SCAN has only 3,200 flows in the capped training pool
# (99,670 total; BENIGN/SYN_FLOOD/DOS_HULK sit at the 25,000 cap) and
# SSH_BRUTE only 1,423 -- genuine rarity in the raw lab data, not a capping
# artifact. LR/LinearSVC score PORT_SCAN F1=0.000 not just externally but
# in-domain (L2 val) AND on every one of the 4 LODO folds -- a within-lab
# learnability problem, fully diagnosable and fixable without ever looking
# at CICIDS2017. V1's own frozen models (OLD EXPIRED RESEARCH/reports/
# exp1/summary.csv) all used class_weight='balanced'; V2's Section 15
# respec dropped it with no documented reason. Reintroducing it as a
# SEARCHED dimension (not a forced default) for every model that supports
# imbalance handling, validated purely via L2/LODO exactly like every other
# hyperparameter -- CICIDS2017/L6 is not touched by this change at all,
# only evaluated once at the end per the frozen-model discipline this
# project already follows.
CLASS_WEIGHT_OPTIONS = [None, "balanced"]

def fit_linearsvc_or_fallback(C: float, X: np.ndarray, y: np.ndarray, class_weight: str | None = None):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf = LinearSVC(C=C, max_iter=5000, tol=1e-3, random_state=SEED, class_weight=class_weight)
        clf.fit(X, y)
        did_not_converge = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    if did_not_converge or (hasattr(clf, "n_iter_") and np.max(clf.n_iter_) >= 5000):
        clf = LogisticRegression(solver="saga", max_iter=2000, C=C, random_state=SEED, class_weight=class_weight)
        clf.fit(X, y)
        return clf, True
    return clf, False

def make_model(name: str, params: dict):
    class_weight = params.get("class_weight")
    if name == "LR":
        # n_jobs dropped 2026-08-16: sklearn 1.9's LogisticRegression(solver="saga")
        # no longer uses it (deprecated since 1.8, FutureWarning-only, harmless, but
        # removing it keeps the smoke-tested run's console output clean).
        return LogisticRegression(C=params["C"], solver="saga", max_iter=1000, random_state=SEED,
                                   class_weight=class_weight)
    if name == "RF":
        return RandomForestClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                                       min_samples_leaf=params["min_samples_leaf"], random_state=SEED, n_jobs=-1,
                                       class_weight=class_weight)
    if name == "XGB":
        # XGBClassifier has no class_weight param -- multiclass imbalance handling
        # goes through sample_weight at .fit() time instead (see cv_objective() and
        # main()'s refit, which compute it via compute_sample_weight when this is
        # "balanced"). make_model() just builds the estimator; the caller applies
        # the weights.
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=params["n_estimators"], max_depth=params["max_depth"],
                              learning_rate=params["learning_rate"], subsample=params["subsample"],
                              random_state=SEED, n_jobs=-1, eval_metric="mlogloss")
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=params["n_estimators"], num_leaves=params["num_leaves"],
                               learning_rate=params["learning_rate"], random_state=SEED, n_jobs=-1, verbosity=-1,
                               class_weight=class_weight)
    raise ValueError(name)

def suggest_params(trial: "optuna.Trial", name: str) -> dict:
    if name == "LR":
        return {"C": trial.suggest_float("C", 0.001, 100, log=True),
                "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_OPTIONS)}
    if name == "LinearSVC":
        return {"C": trial.suggest_float("C", 0.001, 100, log=True),
                "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_OPTIONS)}
    if name == "RF":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_OPTIONS)}
    if name == "XGB":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_OPTIONS)}
    if name == "LGBM":
        return {"n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "num_leaves": trial.suggest_int("num_leaves", 20, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "class_weight": trial.suggest_categorical("class_weight", CLASS_WEIGHT_OPTIONS)}
    raise ValueError(name)

def fit_model_with_class_weight(name: str, params: dict, X: np.ndarray, y: np.ndarray):
    """Fit classifier with balanced class weighting and convergence protection."""
    class_weight = params.get("class_weight")
    if name == "LinearSVC":
        if len(X) > LINEARSVC_ROW_LIMIT:
            clf = LogisticRegression(solver="saga", max_iter=2000, C=params["C"], random_state=SEED,
                                      class_weight=class_weight)
            clf.fit(X, y)
            return clf, True
        return fit_linearsvc_or_fallback(params["C"], X, y, class_weight=class_weight)
    if name == "XGB" and class_weight == "balanced":
        from sklearn.utils.class_weight import compute_sample_weight
        clf = make_model(name, params)
        clf.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        return clf, False
    clf = make_model(name, params)
    clf.fit(X, y)
    return clf, False

def cv_objective(name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
    gkf = GroupKFold(n_splits=N_FOLDS)

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_params(trial, name)
        fold_f1 = []
        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            clf, _fell_back = fit_model_with_class_weight(name, params, X_tr, y_tr)
            y_pred = clf.predict(X_va)
            fold_f1.append(f1_score(y_va, y_pred, average="macro", zero_division=0))
        return float(np.mean(fold_f1))

    return objective

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Loading primary split and Phase 1 feature dictionary...")
    split = load_split()
    sessions = split["sessions"]
    feature_cols = load_feature_groups()
    print(f"  {len(feature_cols)} Group A+B verified features")

    print("\nLoading TRAIN-split flows (D01-D04)...")
    train_natural = load_split_flows("train", sessions)
    print(f"  Total train-session flows (natural): {len(train_natural):,}")

    print("\nBuilding session-stratified capped training pool "
          f"(cap={PER_CLASS_CAP:,}/class, seed={SEED})...")
    train_pool = build_capped_train_pool(train_natural)
    print(f"  Capped pool: {len(train_pool):,} flows")
    print(train_pool["label"].value_counts().to_string())

    print("\nLoading VAL-split flows (D01-D04, natural distribution, not rebalanced)...")
    val_natural = load_split_flows("val", sessions)
    print(f"  Total val-session flows: {len(val_natural):,}")

    scaler = RobustScaler()
    X_train = scaler.fit_transform(train_pool[feature_cols].to_numpy(dtype=np.float64))
    y_train = train_pool["label"].to_numpy()
    groups_train = train_pool["session_id"].to_numpy()
    X_val = scaler.transform(val_natural[feature_cols].to_numpy(dtype=np.float64))
    y_val_str = val_natural["label"].to_numpy()

    # FOUND 2026-08-17: XGBoost crashed on trial 0 -- "Invalid classes inferred
    # from unique values of `y`. Expected: [0..8], got ['BENIGN', ...]".
    # XGBClassifier's sklearn wrapper (unlike LogisticRegression/RandomForest/
    # LinearSVC/LGBMClassifier, which all handle arbitrary string labels via
    # their own internal encoding) requires integer class labels starting at
    # 0. Rather than special-case XGB alone, ALL models now fit/predict on a
    # single consistent integer encoding (LabelEncoder fit once on y_train;
    # val must not introduce a class train never saw, so transform(), not a
    # second fit) -- avoids any chance of two different models silently using
    # two different label<->int mappings. classification_report/print output
    # still shows real class NAMES via label_encoder.classes_, not integers.
    label_encoder = LabelEncoder()
    y_train_str = y_train
    y_train = label_encoder.fit_transform(y_train_str)
    y_val = label_encoder.transform(y_val_str)

    models = ["LR", "LinearSVC", "RF", "XGB", "LGBM"]
    # v2 (2026-08-18): writes to a NEW file, does not touch frozen_model_config_v1.json
    # (Section 10B: a frozen config, once written, is never changed -- v1 stays archived
    # in reports/archive/2026-08-18_v1_run/ exactly as it finished). This run adds
    # class_weight as a searched dimension for every model that supports imbalance
    # handling (see CLASS_WEIGHT_OPTIONS comment) -- motivated entirely by L2/LODO
    # evidence (PORT_SCAN/SSH_BRUTE genuine training-set rarity, LR/LinearSVC scoring
    # PORT_SCAN F1=0.000 in-domain and on every LODO fold), never by CICIDS2017/L6,
    # which this run does not touch at all.
    out_path = CONFIGS_DIR / "frozen_model_config_v2.json"

    # FOUND 2026-08-17: XGB's crash exited the whole script with no
    # checkpointing at all -- LR/LinearSVC/RF's already-completed results
    # (the RF search alone took ~1h) existed only in this function's local
    # `frozen` dict and were lost the instant the process died, since nothing
    # was written to disk until the very end of the loop. Same failure shape
    # as this project's own extract_and_label.py before its "already
    # extracted, skip" cache check existed. Fixed the same way: if
    # frozen_model_config_v1.json already has a result for a given model,
    # skip re-running it; write the file after EVERY model (not just at the
    # end) so a future crash can never cost more than the one model that was
    # actually in progress.
    if out_path.exists():
        frozen = json.loads(out_path.read_text(encoding="utf-8"))
        already_done = [m for m in models if m in frozen.get("models", {})]
        if already_done:
            print(f"\nResuming: {already_done} already have results in {out_path.name}, skipping re-run.")
    else:
        frozen = {
            "description": "Frozen model configuration and tuned hyperparameters from Optuna 50-trial optimization."
,
            "feature_columns": feature_cols,
            "scaler": "RobustScaler",
            "label_classes_in_order": label_encoder.classes_.tolist(),
            "per_class_cap": PER_CLASS_CAP,
            "seed": SEED,
            "n_trials": N_TRIALS,
            "n_folds": N_FOLDS,
            "n_train_flows_capped": len(train_pool),
            "n_val_flows_natural": len(val_natural),
            "models": {},
        }

    for name in models:
        if name in frozen["models"]:
            print(f"\n[skip] {name} already has a result in {out_path.name} -- not re-running.", flush=True)
            continue
        n_trials_this_model = MODEL_N_TRIALS[name]
        print(f"\n{'=' * 70}\nHPO: {name} ({n_trials_this_model} trial(s), {N_FOLDS}-fold GroupKFold)\n{'=' * 70}", flush=True)
        t0 = time.time()

        # FOUND 2026-08-17: show_progress_bar=False left ZERO visible output
        # between the "HPO: LR" header and the final result -- on a run where
        # a single LR trial (5 folds, saga not converging within max_iter=1000
        # on this data) can take real minutes, that looked indistinguishable
        # from a hang for 40+ minutes with no way to tell the difference. This
        # callback prints one line per completed trial (trial #, its macro-F1,
        # best-so-far, elapsed) so progress is never silent again -- same
        # lesson as this project's own SLICE_START/SLICE_DONE and
        # failed_sessions visibility fixes in extract_and_label.py.
        def progress_cb(study: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
            elapsed = time.time() - t0
            print(f"    trial {trial.number + 1}/{n_trials_this_model}  value={trial.value:.4f}  "
                  f"best={study.best_value:.4f}  elapsed={elapsed:.0f}s", flush=True)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
        if name == "LR":
            # Two forced trials -- C fixed, class_weight varied -- see
            # MODEL_N_TRIALS comment above for why the full search stays skipped.
            study.enqueue_trial({"C": LR_FIXED_C, "class_weight": None})
            study.enqueue_trial({"C": LR_FIXED_C, "class_weight": "balanced"})
        elif name == "LinearSVC":
            study.enqueue_trial({"C": LINEARSVC_FIXED_C, "class_weight": None})
            study.enqueue_trial({"C": LINEARSVC_FIXED_C, "class_weight": "balanced"})
        study.optimize(cv_objective(name, X_train, y_train, groups_train), n_trials=n_trials_this_model,
                        n_jobs=OPTUNA_N_JOBS[name], show_progress_bar=False, callbacks=[progress_cb])
        hpo_s = time.time() - t0
        best_params = study.best_params
        print(f"  Best CV macro-F1: {study.best_value:.4f}  ({hpo_s:.0f}s)  params={best_params}")

        # Refit on FULL capped training pool with best params, evaluate on val (natural).
        clf, fell_back = fit_model_with_class_weight(name, best_params, X_train, y_train)

        y_pred = clf.predict(X_val)
        val_macro_f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        labels_sorted = sorted(set(y_train) | set(y_val))  # integers; label_encoder.classes_[i] -> real name
        report = classification_report(y_val, y_pred, labels=labels_sorted, zero_division=0, output_dict=True)
        # classification_report's output_dict keys the STRING form of whatever label value was
        # passed in (e.g. int 0 -> dict key "0") -- str(c) here, then map back to the real class
        # name via label_encoder for anything a human (or the frozen config) actually reads.
        per_class_f1 = {label_encoder.classes_[c]: round(report[str(c)]["f1-score"], 4)
                         for c in labels_sorted if str(c) in report}
        print(f"  VAL macro-F1: {val_macro_f1:.4f}")
        for c, f1 in sorted(per_class_f1.items(), key=lambda kv: kv[1]):
            print(f"      {c:15s} F1={f1:.3f}")

        frozen["models"][name] = {
            "best_hyperparameters": best_params,
            "n_trials_run": n_trials_this_model,
            "linearsvc_fell_back_to_logreg": fell_back if name == "LinearSVC" else None,
            "cv_macro_f1_mean": round(study.best_value, 4),
            "val_macro_f1": round(val_macro_f1, 4),
            "val_per_class_f1": per_class_f1,
            "hpo_seconds": round(hpo_s, 1),
            "note": self_note(name),
        }

        # Write after EVERY model, not just at the end -- see the resume-logic
        # comment above for why (2026-08-17 XGB crash lost nothing further
        # than the model in progress, specifically because of this line).
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        print(f"  [checkpoint] {name} result saved to {out_path.name}")

    print(f"\n{'=' * 70}\nFROZEN. Written: {out_path}\n{'=' * 70}")
    print(f"{'model':10s} {'CV macro-F1':>12s} {'VAL macro-F1':>13s}")
    for name in models:
        m = frozen["models"][name]
        print(f"{name:10s} {m['cv_macro_f1_mean']:12.4f} {m['val_macro_f1']:13.4f}")

if __name__ == "__main__":
    main()
