#!/usr/bin/env python3
"""
Dataset Quality Gate and Integrity Verification Suite.

Performs rigorous post-extraction audits across all Parquet datasets, validating schema consistency,
verifying absence of null/infinite values, and checking label distributions against manifests.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT     = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# Gates that cause hard exit(1) if failed
HARD_GATES = {"QG-01", "QG-02", "QG-03", "QG-04", "QG-05", "QG-07", "QG-13", "QG-14"}
# Gates that cause exit(2) (warning) if failed
SOFT_GATES = {"QG-06", "QG-08", "QG-09", "QG-10", "QG-11", "QG-12"}

def banner(msg: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {msg}")
    print(f"{'─'*60}")

def result(gate: str, passed: bool, detail: str) -> bool:
    tag = "PASS" if passed else ("FAIL" if gate in HARD_GATES else "WARN")
    print(f"  [{gate} {tag}] {detail}")
    return passed

# ── Individual gates ──────────────────────────────────────────────────────────

def qg01_no_nan(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    num = df.select_dtypes(include=[np.number])
    count = num.isna().sum().sum()
    return result("QG-01", count == 0,
                  f"NaN count = {count}" + (f"\n    Columns: {list(num.columns[num.isna().any()])}" if count else ""))

def qg02_no_inf(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    num = df.select_dtypes(include=[np.number])
    count = np.isinf(num).sum().sum()
    return result("QG-02", count == 0, f"Inf count = {count}")

def qg03_unknown_rate(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    pooled_rate = (df["label"] == "UNKNOWN").mean()
    if "session_id" not in df.columns:
        return result("QG-03", pooled_rate < 0.05,
                       f"UNKNOWN rate = {pooled_rate:.2%} (threshold < 5%) "
                       f"[no session_id column — checked pooled rate only]")
    per_session = df.groupby("session_id").apply(
        lambda g: (g["label"] == "UNKNOWN").mean(), include_groups=False
    )
    worst_session = per_session.idxmax()
    worst_rate = per_session.max()
    detail = (f"Worst-session UNKNOWN rate = {worst_rate:.2%} in {worst_session} "
              f"(threshold < 5% per session; pooled rate = {pooled_rate:.2%})")
    if worst_rate >= 0.05:
        failing = per_session[per_session >= 0.05]
        detail += f"\n    Sessions over threshold: {failing.round(4).to_dict()}"
    return result("QG-03", worst_rate < 0.05, detail)

def qg04_no_conflict(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    count = (df["label"] == "CONFLICT").sum()
    return result("QG-04", count == 0, f"CONFLICT count = {count}")

def qg05_label_column(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    if "label" not in df.columns:
        return result("QG-05", False, "Missing 'label' column")
    valid = {"BENIGN", "UNKNOWN", "CONFLICT"}
    # Add any attack family labels (anything not a meta-label)
    attack_labels = set(df["label"].unique()) - valid
    all_labels = valid | attack_labels
    unexpected = set(df["label"].unique()) - all_labels
    return result("QG-05", len(unexpected) == 0,
                  f"Labels present: {sorted(df['label'].unique())}")

def qg06_zero_variance(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    num = df.select_dtypes(include=[np.number])
    zero_var = [c for c in num.columns if num[c].std() == 0]
    passed = len(zero_var) == 0
    detail = f"Zero-variance columns: {zero_var}" if zero_var else "None"
    return result("QG-06", passed, detail)

def qg07_session_count(df: pd.DataFrame, min_sessions: int = 10) -> bool:
    """Execute internal routine."""
    if "session_id" not in df.columns:
        return result("QG-07", False, "Missing 'session_id' column")
    attack_labels = [l for l in df["label"].unique()
                     if l not in {"BENIGN", "UNKNOWN", "CONFLICT"}]
    failing = []
    for label in attack_labels:
        count = df[df["label"] == label]["session_id"].nunique()
        if count < min_sessions:
            failing.append(f"{label}: {count} sessions (need {min_sessions})")
    passed = len(failing) == 0
    detail = "; ".join(failing) if failing else f"All attack classes ≥ {min_sessions} sessions"
    return result("QG-07", passed, detail)

def qg08_class_balance(df: pd.DataFrame, threshold: float = 0.50) -> bool:
    """Execute internal routine."""
    fracs = df["label"].value_counts(normalize=True)
    dominant = fracs[fracs > threshold]
    passed = dominant.empty
    detail = (f"Dominant classes: {dominant.to_dict()}" if not passed
              else f"Max class fraction = {fracs.max():.2%}")
    return result("QG-08", passed, detail)

def qg09_benign_present(df: pd.DataFrame) -> bool:
    """Execute internal routine."""
    present = "BENIGN" in df["label"].values
    return result("QG-09", present, "BENIGN flows present" if present else "No BENIGN flows!")

def qg10_attack_classes(df: pd.DataFrame, min_classes: int = 2) -> bool:
    """Execute internal routine."""
    attack_labels = [l for l in df["label"].unique()
                     if l not in {"BENIGN", "UNKNOWN", "CONFLICT"}]
    passed = len(attack_labels) >= min_classes
    return result("QG-10", passed,
                  f"Attack classes = {sorted(attack_labels)} ({len(attack_labels)} found, need {min_classes})")

def qg11_flow_count(df: pd.DataFrame, min_flows: int = 1000) -> bool:
    """Execute internal routine."""
    passed = len(df) >= min_flows
    return result("QG-11", passed,
                  f"Total flows = {len(df):,} (min {min_flows:,})")

def qg12_feature_count(df: pd.DataFrame, min_features: int = 20) -> bool:
    """Execute internal routine."""
    meta_cols = {"label", "session_id", "domain_id", "timestamp",
                 "src_ip", "dst_ip", "src_port", "dst_port", "protocol"}
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in meta_cols]
    passed = len(feature_cols) >= min_features
    return result("QG-12", passed,
                  f"Feature columns = {len(feature_cols)} (min {min_features})")

def qg13_domain_plan_coverage(df: pd.DataFrame, domain: str, pilot: bool) -> bool:
    """Execute internal routine."""
    if pilot:
        return result("QG-13", True, "Skipped in --pilot mode (uses PILOT_PLAN, not DOMAIN_PLANS)")
    try:
        from orchestrate import DOMAIN_PLANS
    except ImportError as e:
        return result("QG-13", False, f"Could not import DOMAIN_PLANS from orchestrate.py to check against: {e}")
    if domain not in DOMAIN_PLANS:
        return result("QG-13", True, f"No DOMAIN_PLANS entry for {domain!r} -- nothing to check")

    expected = {}
    for family, variant_id, generator, config in DOMAIN_PLANS[domain]:
        if family is None:  # benign-only sessions
            continue
        expected[family] = expected.get(family, 0) + 1

    problems = []
    for family, expected_count in sorted(expected.items()):
        actual_count = df[df["label"] == family]["session_id"].nunique()
        if actual_count < expected_count:
            problems.append(f"{family}: {actual_count}/{expected_count} configured session(s) present in data")

    passed = len(problems) == 0
    detail = ("; ".join(problems) if problems
              else f"All {len(expected)} configured attack families fully represented "
                   f"({sum(expected.values())} sessions total)")
    return result("QG-13", passed, detail)

def qg14_flow_floor(df: pd.DataFrame, pilot: bool,
                    min_flows: int = 100, target_flows: int = 500) -> bool:
    """Execute internal routine."""
    if pilot:
        return result("QG-14", True,
                      "Skipped in --pilot mode (short single-variant sessions "
                      "legitimately produce few flows — e.g. ICMP_FLOOD's 2)")
    attack_labels = [l for l in df["label"].unique()
                     if l not in {"BENIGN", "UNKNOWN", "CONFLICT"}]
    counts = {l: int((df["label"] == l).sum()) for l in attack_labels}
    below    = {l: n for l, n in sorted(counts.items()) if n < min_flows}
    marginal = {l: n for l, n in sorted(counts.items()) if min_flows <= n < target_flows}
    passed = len(below) == 0
    if below:
        detail = (f"Attack families below the {min_flows}-flow floor — unusable "
                  f"for per-class metrics under session splits: {below}")
    else:
        detail = f"All {len(attack_labels)} attack families ≥ {min_flows} flows"
    if marginal:
        detail += (f"\n    [advisory] marginal (< {target_flows} flows — aim higher "
                   f"for stable per-class F1): {marginal}")
    return result("QG-14", passed, detail)

# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # FOUND 2026-08-16 (see extract_and_label.py's identical guard): the
    # 2026-08-13 UnicodeEncodeError fix only covers this script when it's
    # launched THROUGH run_full_generation.py's subprocess wrapper -- a
    # direct invocation with redirected stdout still hits the same crash.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True)
    parser.add_argument("--strict", action="store_true",
                        help="Treat soft-gate warnings as hard failures")
    parser.add_argument("--min-sessions", type=int, default=10,
                        help="Minimum sessions per attack class (QG-07)")
    parser.add_argument("--min-flows-per-family", type=int, default=100,
                        help="QG-14 hard floor: min total flows per attack family "
                             "(advisory target 500). Below this a class can't be "
                             "trained/scored under session-level splits.")
    parser.add_argument("--pilot", action="store_true",
                        help="Pilot mode: lower thresholds (min-sessions=1, min-flows=100)")
    args = parser.parse_args()

    if args.pilot:
        args.min_sessions = 1
        min_flows = 100
    else:
        min_flows = 1000

    parquet = PROCESSED_DIR / args.domain / "labelled.parquet"
    if not parquet.exists():
        print(f"[ERROR] Parquet not found: {parquet}", file=sys.stderr)
        print("Run extract_and_label.py first.", file=sys.stderr)
        sys.exit(1)

    banner(f"V2 Dataset Validator — {args.domain}")
    df = pd.read_parquet(parquet)
    print(f"  Loaded: {len(df):,} flows, {df.shape[1]} columns")
    print(f"  Labels: {df['label'].value_counts().to_dict()}")

    banner("Running Quality Gates")

    gates = [
        qg01_no_nan(df),
        qg02_no_inf(df),
        qg03_unknown_rate(df),
        qg04_no_conflict(df),
        qg05_label_column(df),
        qg06_zero_variance(df),
        qg07_session_count(df, min_sessions=args.min_sessions),
        qg08_class_balance(df),
        qg09_benign_present(df),
        qg10_attack_classes(df),
        qg11_flow_count(df, min_flows=min_flows),
        qg12_feature_count(df),
        qg13_domain_plan_coverage(df, args.domain, args.pilot),
        qg14_flow_floor(df, args.pilot, min_flows=args.min_flows_per_family),
    ]

    n_pass = sum(gates)
    n_fail = len(gates) - n_pass

    banner(f"Summary: {n_pass}/{len(gates)} gates passed")

    if n_fail == 0:
        print("  ✓ Dataset passed all quality gates. Safe for ML training.\n")
        sys.exit(0)

    # Classify failures by severity
    hard_failures = []
    soft_failures = []
    gate_names = ["QG-01","QG-02","QG-03","QG-04","QG-05","QG-06",
                  "QG-07","QG-08","QG-09","QG-10","QG-11","QG-12","QG-13","QG-14"]
    for i, (passed, name) in enumerate(zip(gates, gate_names)):
        if not passed:
            if name in HARD_GATES:
                hard_failures.append(name)
            else:
                soft_failures.append(name)

    if hard_failures:
        print(f"  ✗ Hard failures (MUST fix before training): {hard_failures}")
        print("  Append findings to problems_notes.txt\n")
        sys.exit(1)

    if soft_failures:
        print(f"  ⚠ Soft warnings (investigate): {soft_failures}")
        if args.strict:
            sys.exit(1)
        sys.exit(2)

if __name__ == "__main__":
    main()
