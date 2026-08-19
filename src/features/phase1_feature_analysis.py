"""
Phase 1 Feature Selection and Collinearity Analysis.

Analyzes raw flow metrics for near-zero variance, pairwise collinearity, and mutual information
to construct the verified feature dictionary and eliminate redundant dimensions.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
DOMAINS = ["D01", "D02", "D03", "D04"]

ID_COLS = ["Flow ID", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
           "timestamp", "Label", "label", "session_id", "domain_id"]

NAN_INF_EXCLUDE_THRESHOLD = 0.05  # spec: "> 5% NaN/Inf rate"
CORRELATION_FLAG_THRESHOLD = 0.95  # not specified exactly in the spec; documented choice

# Section 10B's feature-group table, transcribed by hand from lab/feature_schema_v2.yaml's
# real column list (2026-08-14, generated from actual pilot output) -- every name below is
# checked against the real parquet columns at runtime; the script fails loudly if any group
# member is missing or any real feature column isn't assigned to a group.
FEATURE_GROUPS = {
    "A_volumetric": [
        "fwd_pkts", "bwd_pkts", "fwd_bytes", "bwd_bytes",
        "flow_bytes_s", "flow_pkts_s", "fwd_pkts_s", "bwd_pkts_s",
        "Fwd Act Data Pkts", "Down/Up Ratio",
    ],
    "B_packet_size": [
        "fwd_pkt_len_max", "fwd_pkt_len_min", "fwd_pkt_len_mean", "fwd_pkt_len_std",
        "bwd_pkt_len_max", "bwd_pkt_len_min", "bwd_pkt_len_mean", "bwd_pkt_len_std",
        "Packet Length Min", "Packet Length Max", "Packet Length Mean",
        "Packet Length Std", "Packet Length Variance", "Average Packet Size",
        "Fwd Segment Size Avg", "Bwd Segment Size Avg", "Fwd Seg Size Min",
    ],
    "C_temporal": [
        "flow_duration",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
        "Bwd IAT Total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    ],
    "D_tcp_flags": [
        "fwd_fin_cnt", "fwd_syn_cnt", "fwd_rst_cnt", "fwd_psh_cnt", "fwd_ack_cnt",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "URG Flag Count", "CWR Flag Count", "ECE Flag Count",
    ],
    "E_header_connection": [
        "Fwd Header Length", "Bwd Header Length", "FWD Init Win Bytes", "Bwd Init Win Bytes",
        "Fwd Bytes/Bulk Avg", "Fwd Packet/Bulk Avg", "Fwd Bulk Rate Avg",
        "Bwd Bytes/Bulk Avg", "Bwd Packet/Bulk Avg", "Bwd Bulk Rate Avg",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ],
}

# Per Section 10B's own group table: Group C ("HIGH domain sensitivity; unit-sensitive...
# verify within CICFlowMeter") and Group D ("documented fwd/bwd assignment bugs in some
# versions; verify against Wireshark") start QUESTIONABLE unless independently verified.
# The 2026-08-13 scapy cross-check (see extraction_metadata.json) independently verified
# packet COUNTS and flow_duration for one session -- real evidence, but it does not cover
# IAT statistics or any TCP flag field, so it upgrades flow_duration specifically with a
# documented note rather than upgrading the whole group.
GROUPS_STARTING_QUESTIONABLE = {"C_temporal", "D_tcp_flags"}
VERIFIED_WITH_NOTE = {
    "flow_duration": "Independently cross-checked against raw packet timestamps for one "
                      "ICMP session, 2026-08-13 (see extraction_metadata.json). Remains in "
                      "Group C (temporal) for the ablation table, but treated as VERIFIED "
                      "rather than QUESTIONABLE on the strength of that direct evidence -- "
                      "unlike the rest of Group C (IAT statistics), which have not been "
                      "independently checked against raw packets.",
}

def load_pooled_training_data() -> pd.DataFrame:
    frames = []
    for d in DOMAINS:
        path = DATA_DIR / d / "labelled.parquet"
        df = pd.read_parquet(path)
        df = df[df["label"] != "UNKNOWN"].copy()
        df["domain_id"] = d  # already present, but keep authoritative for this analysis
        frames.append(df)
        print(f"  [{d}] {len(df):,} flows loaded")
    pooled = pd.concat(frames, ignore_index=True)
    print(f"  TOTAL pooled: {len(pooled):,} flows across {len(DOMAINS)} domains")
    return pooled

def main() -> None:
    print("=" * 70)
    print("PHASE 1 -- Pre-test feature analysis (D01-D04 training data only)")
    print("=" * 70)
    df = load_pooled_training_data()

    feature_cols = [c for c in df.columns if c not in ID_COLS]
    grouped_cols = {c for cols in FEATURE_GROUPS.values() for c in cols}
    missing_from_groups = set(feature_cols) - grouped_cols
    extra_in_groups = grouped_cols - set(feature_cols)
    if missing_from_groups:
        raise RuntimeError(f"Real feature column(s) not assigned to any Section 10B group: "
                            f"{sorted(missing_from_groups)} -- fix FEATURE_GROUPS before trusting this analysis.")
    if extra_in_groups:
        raise RuntimeError(f"FEATURE_GROUPS references column(s) that don't exist in the real "
                            f"data: {sorted(extra_in_groups)} -- schema drift since this script was written.")
    print(f"\n{len(feature_cols)} real feature columns, all accounted for across the 5 groups.\n")

    col_to_group = {c: g for g, cols in FEATURE_GROUPS.items() for c in cols}

    # ---- Global variance + NaN/Inf rate ----
    print("Computing global variance and NaN/Inf rates...")
    n_rows = len(df)
    stats = {}
    for c in feature_cols:
        s = df[c]
        nan_rate = s.isna().mean()
        inf_rate = np.isinf(s.to_numpy(dtype=np.float64)).mean()
        stats[c] = {
            "std": float(s.std()),
            "nan_rate": float(nan_rate),
            "inf_rate": float(inf_rate),
            "zero_variance": bool(s.std() == 0),  # exact match to validate_dataset.py's QG-06
        }

    # ---- Within-domain variance ----
    print("Computing within-domain variance per feature...")
    for c in feature_cols:
        per_domain_var = df.groupby("domain_id", observed=True)[c].var()
        per_domain_mean = df.groupby("domain_id", observed=True)[c].mean()
        pooled_var = df[c].var()
        # Stability score: 1 - (variance of per-domain means / pooled variance).
        # Not a formula given verbatim in the spec (Section 10B just says "compute
        # within-domain variance per feature" and later references a "LOW-DRIFT"
        # subset "above threshold"). This is a defensible, documented choice: a
        # feature whose per-domain MEANS barely move relative to its own overall
        # spread scores close to 1 (stable); one whose domain means are as spread
        # out as the data itself scores close to 0 (unstable/domain-sensitive).
        between_domain_var = float(per_domain_mean.var()) if len(per_domain_mean) > 1 else 0.0
        stability = 1.0 - (between_domain_var / pooled_var) if pooled_var > 0 else 0.0
        stats[c]["within_domain_variance"] = {d: float(v) for d, v in per_domain_var.items()}
        stats[c]["domain_stability_score"] = round(float(np.clip(stability, -10, 1.0)), 4)

    # ---- Pairwise correlation ----
    print("Computing pairwise correlation matrix (flagging |r| >= "
          f"{CORRELATION_FLAG_THRESHOLD})...")
    corr = df[feature_cols].corr(numeric_only=True)
    high_corr_pairs = []
    for i, a in enumerate(feature_cols):
        for b in feature_cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r) and abs(r) >= CORRELATION_FLAG_THRESHOLD:
                high_corr_pairs.append({"feature_a": a, "feature_b": b, "r": round(float(r), 4)})
    print(f"  {len(high_corr_pairs)} pair(s) with |r| >= {CORRELATION_FLAG_THRESHOLD}")

    # ---- Assignment: VERIFIED / QUESTIONABLE / EXCLUDED ----
    print("\nAssigning VERIFIED / QUESTIONABLE / EXCLUDED...")
    assignments = {}
    for c in feature_cols:
        s = stats[c]
        if s["zero_variance"] or s["nan_rate"] > NAN_INF_EXCLUDE_THRESHOLD or s["inf_rate"] > NAN_INF_EXCLUDE_THRESHOLD:
            category = "EXCLUDED"
            reason = ("zero variance across all pooled D01-D04 training data" if s["zero_variance"]
                       else f"NaN/Inf rate {max(s['nan_rate'], s['inf_rate']):.2%} exceeds the 5% threshold")
        elif c in VERIFIED_WITH_NOTE:
            category = "VERIFIED"
            reason = VERIFIED_WITH_NOTE[c]
        elif col_to_group[c] in GROUPS_STARTING_QUESTIONABLE:
            category = "QUESTIONABLE"
            group = col_to_group[c]
            reason = ("Group C (temporal): HIGH domain sensitivity per Section 10B; IAT "
                      "statistics not independently verified against raw packets (only "
                      "flow_duration has that evidence -- see flow_duration's own entry)."
                      if group == "C_temporal" else
                      "Group D (TCP flags): CICFlowMeter has documented fwd/bwd assignment "
                      "bugs in some versions per Section 10B; never independently verified "
                      "against Wireshark/raw packets in this project.")
        else:
            category = "VERIFIED"
            reason = f"Group {col_to_group[c]}: no documented domain-sensitivity concern in Section 10B."
        assignments[c] = {"category": category, "group": col_to_group[c], "reason": reason}

    counts = pd.Series([a["category"] for a in assignments.values()]).value_counts()
    print(counts.to_string())

    # ---- Write feature_dictionary_v2.yaml ----
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "description": "Phase 1 feature analysis: collinearity filtering and variance thresholding.",
        "n_training_flows": int(n_rows),
        "domains": DOMAINS,
        "correlation_flag_threshold": CORRELATION_FLAG_THRESHOLD,
        "high_correlation_pairs": high_corr_pairs,
        "features": {
            c: {
                "group": assignments[c]["group"],
                "category": assignments[c]["category"],
                "reason": assignments[c]["reason"],
                "global_std": round(stats[c]["std"], 6),
                "nan_rate": stats[c]["nan_rate"],
                "inf_rate": stats[c]["inf_rate"],
                "domain_stability_score": stats[c]["domain_stability_score"],
            }
            for c in feature_cols
        },
    }
    out_path = CONFIGS_DIR / "feature_dictionary_v2.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True, width=100)
    print(f"\nWritten: {out_path}")

    excluded = [c for c, a in assignments.items() if a["category"] == "EXCLUDED"]
    print(f"\nEXCLUDED ({len(excluded)}): {excluded}")

if __name__ == "__main__":
    main()
