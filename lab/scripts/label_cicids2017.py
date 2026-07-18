#!/usr/bin/env python3
"""
CICIDS2017 Flow Label Assignment from Ground-Truth Timeframes.

Applies official attack timeframe tables and IP mappings to re-extracted CICIDS2017 flows,
standardizing attack taxonomy across internal lab captures and external benchmark data.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FLOWS_DIR = REPO_ROOT / "data" / "flows" / "CICIDS2017_L6"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "CICIDS2017_L6"
LABELLING_DIR = REPO_ROOT / "data" / "raw" / "cicids2017" / "TrafficLabelling"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_and_label import load_csv, CICIDS_RENAME  # noqa: E402 -- reuse, don't reimplement

DAY_EXTRACTED_PCAP = {
    "Tuesday": "Tuesday-WorkingHours.pcap",
    "Wednesday": "Wednesday-workingHours.pcap",
    "Friday": "Friday-WorkingHours.pcap",
}
# Published label source file(s) per day -- Friday's single PCAP corresponds
# to 3 separate published sub-files (Morning/PortScan/DDoS); confirmed by
# directory listing, not assumed.
DAY_LABEL_FILES = {
    "Tuesday": ["Tuesday-WorkingHours.pcap_ISCX.csv"],
    "Wednesday": ["Wednesday-workingHours.pcap_ISCX.csv"],
    "Friday": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
}

# Section 13's class-mapping table, transcribed exactly.
DIRECT_MAP = {
    "BENIGN": "BENIGN",
    "DoS Hulk": "DOS_HULK",
    "DoS Slowloris": "DOS_SLOWLORIS",
    "DoS slowloris": "DOS_SLOWLORIS",  # published CSV uses lowercase 'l' -- confirmed by direct inspection
    "FTP-Patator": "FTP_BRUTE",
    "SSH-Patator": "SSH_BRUTE",
    "PortScan": "PORT_SCAN",
}
APPROX_MAP = {
    "DDoS": "SYN_FLOOD",
    "DoS GoldenEye": "DOS_HULK",
    "DoS Slowhttptest": "DOS_SLOWLORIS",
    "DoS slowhttptest": "DOS_SLOWLORIS",  # case variant, same reasoning as above
}
# Everything else (Web Attack - Brute Force, Infiltration, Bot, Heartbleed --
# not in Section 13's table at all, 11 flows in Wednesday, negligible) has no
# V2 equivalent and is left unmapped -> UNKNOWN below.

def canonical_key(ip_a: pd.Series, port_a: pd.Series, ip_b: pd.Series, port_b: pd.Series, proto: pd.Series) -> pd.Series:
    """Execute internal routine."""
    port_a = port_a.astype("int64")
    port_b = port_b.astype("int64")
    proto = proto.astype("int64")
    a = ip_a.astype(str) + ":" + port_a.astype(str)
    b = ip_b.astype(str) + ":" + port_b.astype(str)
    lo = np.where(a <= b, a, b)
    hi = np.where(a <= b, b, a)
    return pd.Series(lo, index=ip_a.index) + "__" + pd.Series(hi, index=ip_a.index) + "__" + proto.astype(str)

def load_published_labels(day: str) -> pd.DataFrame:
    frames = []
    for fname in DAY_LABEL_FILES[day]:
        path = LABELLING_DIR / fname
        df = pd.read_csv(path, encoding="latin1")
        df.columns = df.columns.str.strip()
        frames.append(df[["Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp", "Label"]])
        print(f"  [labels] {fname}: {len(df):,} rows, {dict(df['Label'].value_counts())}")
    pub = pd.concat(frames, ignore_index=True)
    pub["minute"] = pub["Timestamp"].astype(str).str.strip()  # already minute-precision, e.g. "7/7/2017 3:30"
    pub["key"] = canonical_key(pub["Source IP"], pub["Source Port"], pub["Destination IP"], pub["Destination Port"], pub["Protocol"])
    # Majority-vote label per (key, minute) -- see module docstring.
    grouped = pub.groupby(["key", "minute"])["Label"].agg(lambda s: s.value_counts().idxmax())
    print(f"  [labels] {len(pub):,} published rows -> {len(grouped):,} unique (key, minute) groups")
    return grouped

def label_day(day: str) -> None:
    pcap_name = DAY_EXTRACTED_PCAP[day]
    csv_path = FLOWS_DIR / f"{pcap_name}_ISCX.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found -- run extract_cicids2017.py --day {day} first")

    print(f"[label] Loading fresh extraction: {csv_path.name}")
    df = load_csv(csv_path)
    print(f"  {len(df):,} flows after load_csv() numeric/embedded-header cleanup")

    print(f"[label] Loading published ground truth for {day}...")
    label_lookup = load_published_labels(day)

    # Our re-extraction's Timestamp is already renamed to lowercase "timestamp"
    # by load_csv() (via CICIDS_RENAME), format "%d/%m/%Y %I:%M:%S %p" -- parse
    # then floor to the minute and re-render to match the published CSV's own
    # format.
    #
    # VERIFIED 2026-08-18 (not assumed): checked the published Friday-DDoS
    # file's own DDoS-labelled rows directly against the well-documented
    # CICIDS2017 attack window (afternoon) -- they read "3:56" .. "4:16",
    # i.e. the published Timestamp is 12-HOUR clock with the AM/PM marker
    # DROPPED, not 24-hour as first assumed (that assumption would have
    # silently produced zero matches for every PM flow -- caught here before
    # it corrupted every label, not after). This is safe to reproduce exactly
    # (no AM/PM collision risk) because every CICIDS2017 day is a single
    # "working hours" 9am-5pm capture: hours 9/10/11 only ever occur as AM
    # and hours 12/1/2/3/4 only ever occur as PM within that window, so a
    # bare hour number is unambiguous in practice for this dataset.
    # FOUND 2026-08-18 (Friday's second real join attempt -- 0.3% match after
    # fixing the int/float key bug, still clearly broken): compared the exact
    # same flow (identical 5-tuple: 192.168.10.5:51345 <-> 173.231.178.117:443)
    # on both sides directly -- our extraction reads "03:14:09 PM", the
    # published CSV reads "12:14" for that SAME real event. A clean, exact
    # 3-hour offset, not a coincidence: CICIDS2017 was captured at UNB
    # (Fredericton, New Brunswick, Canada) and its published timestamps are in
    # local wall-clock time -- Atlantic Daylight Time, UTC-3 in July under DST.
    # Our CICFlowMeter runs inside the monitor container, which renders in UTC
    # (its own system clock, unrelated to where the packets were originally
    # captured -- pcap files store timestamps as timezone-agnostic epoch
    # values, and CICFlowMeter formats them using whatever local zone the
    # process it's running in is set to). Fix: shift our parsed timestamp back
    # 3 hours before deriving the minute key, so both sides describe the same
    # wall-clock moment. Subtracting from the full datetime (not just the hour
    # field) so a day/month rollover near midnight is handled correctly by
    # pandas automatically, though in practice CICIDS2017's 9am-5pm ADT
    # capture window sits mid-day in UTC too, nowhere near a rollover.
    # FOUND 2026-08-18 (Tuesday's first join attempt after the two fixes
    # above -- 0/392,671 matched, 0.0% again): Friday's 99.2% "confirmation"
    # of the date field order was accidentally uninformative -- July 7 is a
    # palindrome (day==month), so "7/7/2017" reads identically whether the
    # format is D/M or M/D, and the bug hid behind it. Direct same-flow check
    # on Tuesday (identical 5-tuple 192.168.10.5:49182<->192.168.10.3:88):
    # published reads "4/7/2017", which is July 4 (day=4, month=7) -- matches
    # the file being Tuesday's data (July 4, 2017 really was a Tuesday) and
    # the day-first order this project already uses elsewhere (our own
    # Timestamp format is %d/%m/%Y). "4/7/2017" as M/D would be April 7 (a
    # Friday), inconsistent with the filename. Fixed: emit day before month.
    ts = pd.to_datetime(df["timestamp"], format="%d/%m/%Y %I:%M:%S %p") - pd.Timedelta(hours=3)
    hour12 = ts.dt.hour % 12
    hour12 = hour12.mask(hour12 == 0, 12)
    df["minute"] = (
        ts.dt.day.astype(str) + "/" + ts.dt.month.astype(str) + "/" + ts.dt.year.astype(str) + " "
        + hour12.astype(str) + ":" + ts.dt.minute.astype(str).str.zfill(2)
    )
    df["key"] = canonical_key(df["src_ip"], df["src_port"], df["dst_ip"], df["dst_port"], df["protocol"])

    matched = df.set_index(["key", "minute"]).index.map(label_lookup)
    df["cicids_label"] = matched.to_numpy()
    n_matched = df["cicids_label"].notna().sum()
    print(f"[label] {n_matched:,}/{len(df):,} flows matched to a published label "
          f"({100 * n_matched / max(len(df), 1):.1f}%)")

    df["label"] = df["cicids_label"].map(DIRECT_MAP)
    df["label_approx"] = df["cicids_label"].map(APPROX_MAP)
    unmapped = df["cicids_label"].notna() & df["label"].isna() & df["label_approx"].isna()
    print(f"[label] {int(unmapped.sum()):,} matched flow(s) have a published label with no V2 mapping "
          f"(Web Attack/Infiltration/Botnet/Heartbleed etc. -- excluded per Section 13's table)")
    df["label"] = df["label"].fillna("UNKNOWN")

    print(f"[label] Final primary label distribution:\n{df['label'].value_counts().to_string()}")
    approx_counts = df["label_approx"].dropna().value_counts()
    if len(approx_counts):
        print(f"[label] Approximate-mapping flows (secondary analysis only, NOT in 'label'):\n{approx_counts.to_string()}")

    df["domain_id"] = "CICIDS2017_L6"
    df["session_id"] = f"CICIDS2017-{day}"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{day.lower()}_labelled.parquet"
    df.drop(columns=["cicids_label", "minute", "key"]).to_parquet(out_path, index=False)
    print(f"[label] Written: {out_path} ({len(df):,} flows)")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True, choices=sorted(DAY_EXTRACTED_PCAP))
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    label_day(args.day)

if __name__ == "__main__":
    main()
