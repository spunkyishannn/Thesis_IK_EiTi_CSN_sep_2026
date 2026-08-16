"""
Primary Session-Level Train/Validation/Test Split Constructor.

Partitions recording sessions into stratified splits by attack family across domains (D01-D04)
to ensure zero flow leakage between training, validation, and test subsets.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_DIR = REPO_ROOT / "lab" / "manifests"
DATA_DIR = REPO_ROOT / "data" / "processed"
CONFIGS_DIR = REPO_ROOT / "configs"
DOMAINS = ["D01", "D02", "D03"]  # TEMP 2026-08-22: D04 still generating (netem fix + retry in
# progress after a brute-force-timeout bug). Running v2-interim on D01-D03 now to hit the
# thesis deadline; revert to D01-D04 and re-run once D04 finishes. See problems_notes.txt.
SEED = 42

# (train_ratio, val_ratio, test_ratio) -- must sum to 1.0
SPLIT_RATIOS = {
    "D01": (0.70, 0.15, 0.15),
    "D02": (0.70, 0.15, 0.15),
    "D03": (0.70, 0.15, 0.15),
    "D04": (0.70, 0.30, 0.00),
}

def load_real_sessions(domain: str) -> list[dict]:
    # FOUND 2026-08-16 while building this split: D01's manifest has 11
    # different experiment_ids (full generation + multiple regen patches --
    # DOS_HULK OOM fix, the ICMP/UDP multi-source regen, merge_regenerated.py
    # patches) accumulated over its 3-day history. Filtering to just the
    # LAST experiment_id (the pattern every other pipeline script uses,
    # correctly, for a single fresh run) silently picked up only the final
    # 2-session UDP_DDOS patch here, not the real merged 19-session dataset
    # -- because D01's final labelled.parquet is itself a MERGE across
    # several experiment_ids, something merge_regenerated.py did to the
    # data but never had a corresponding "authoritative session list" for
    # anything reading the manifest alone to reconstruct. Fixed by treating
    # the domain's actual labelled.parquet (data/processed/D0N/labelled.parquet)
    # as the ground truth for "which sessions are really in this dataset"
    # instead of re-deriving it from manifest experiment_id bookkeeping --
    # same principle this project has hit repeatedly: trust the real data
    # file over an inferred/assumed selection rule.
    import pandas as pd
    real_df = pd.read_parquet(DATA_DIR / domain / "labelled.parquet", columns=["session_id"])
    real_session_ids = set(real_df["session_id"].unique())

    path = MANIFEST_DIR / f"{domain}_manifest.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    all_recs = [json.loads(l) for l in lines if l.strip()]
    # A session can appear multiple times across experiment_ids (superseded
    # regen attempts) -- keep the LAST manifest record for each session_id
    # that's actually present in the final data, so metadata (attack_family
    # etc.) reflects whatever generated the version that's really there.
    by_session = {}
    for r in all_recs:
        if r["session_id"] in real_session_ids:
            by_session[r["session_id"]] = r
    missing = real_session_ids - set(by_session)
    if missing:
        raise RuntimeError(f"{domain}: {len(missing)} session_id(s) in labelled.parquet have "
                            f"NO manifest record at all: {sorted(missing)} -- data with no "
                            f"traceable ground truth, must not be used for training .")
    return list(by_session.values())

def split_family_sessions(sids: list[str], ratios: tuple[float, float, float], rng: random.Random) -> dict[str, list[str]]:
    train_r, val_r, test_r = ratios
    sids = sorted(sids)
    rng.shuffle(sids)
    n = len(sids)

    if n == 1:
        return {"train": sids, "val": [], "test": []}

    if n == 2:
        # See module docstring: prefer test if this domain has a test slice
        # at all (Section 8C's own reasoning), otherwise val.
        holdout_split = "test" if test_r > 0 else ("val" if val_r > 0 else "train")
        if holdout_split == "train":
            return {"train": sids, "val": [], "test": []}
        result = {"train": [sids[0]], "val": [], "test": []}
        result[holdout_split] = [sids[1]]
        return result

    # n >= 3: proportional rounding, floored at 1 for any non-zero ratio.
    n_val = max(1, round(n * val_r)) if val_r > 0 else 0
    n_test = max(1, round(n * test_r)) if test_r > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        # Ratios + flooring pushed train to zero (only possible at very
        # small n) -- claw back one session from whichever of val/test is
        # larger, since train must never be empty.
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        n_train = n - n_val - n_test

    return {
        "train": sids[:n_train],
        "val": sids[n_train:n_train + n_val],
        "test": sids[n_train + n_val:n_train + n_val + n_test],
    }

def main() -> None:
    rng = random.Random(SEED)
    assignment: dict[str, dict] = {}
    summary_rows = []

    for domain in DOMAINS:
        recs = load_real_sessions(domain)
        by_family = defaultdict(list)
        for r in recs:
            fam = r["attack_family"] or "BENIGN"
            by_family[fam].append(r["session_id"])

        for fam, sids in sorted(by_family.items()):
            split = split_family_sessions(sids, SPLIT_RATIOS[domain], rng)
            for slice_name, slice_ids in split.items():
                for sid in slice_ids:
                    assignment[sid] = {"split": slice_name, "domain": domain, "family": fam}
            summary_rows.append((domain, fam, len(sids), len(split["train"]), len(split["val"]), len(split["test"])))

    print(f"{'domain':6s} {'family':16s} {'n':>3s} {'train':>5s} {'val':>5s} {'test':>5s}")
    for row in summary_rows:
        print(f"{row[0]:6s} {row[1]:16s} {row[2]:3d} {row[3]:5d} {row[4]:5d} {row[5]:5d}")

    n_train = sum(1 for a in assignment.values() if a["split"] == "train")
    n_val = sum(1 for a in assignment.values() if a["split"] == "val")
    n_test = sum(1 for a in assignment.values() if a["split"] == "test")
    print(f"\nTotal sessions: {len(assignment)}  (train={n_train}, val={n_val}, test={n_test})")

    families_no_val = sorted({f"{d}/{fam}" for (d, fam, n, tr, va, te) in summary_rows if va == 0})
    families_no_test = sorted({f"{d}/{fam}" for (d, fam, n, tr, va, te) in summary_rows if te == 0 and SPLIT_RATIOS[d][2] > 0})
    if families_no_val:
        print(f"\nFamilies with ZERO val sessions (n was too small): {families_no_val}")
    if families_no_test:
        print(f"Families with ZERO test sessions despite domain having a test slice (n was too small): {families_no_test}")

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONFIGS_DIR / "primary_split_v1.json"
    out_path.write_text(json.dumps({
        "description": "Primary session-level train, validation, and test split across domains D01-D04."
,
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "sessions": assignment,
    }, indent=2), encoding="utf-8")
    print(f"\nWritten: {out_path}")

if __name__ == "__main__":
    main()
