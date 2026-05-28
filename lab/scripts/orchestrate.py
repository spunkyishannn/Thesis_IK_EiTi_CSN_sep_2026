#!/usr/bin/env python3
"""
Docker Testbed Orchestrator for Controlled Network Traffic Simulation.

Coordinates multi-container testbed topologies (attacker, target, client, monitor),
applies dynamic network impairments via Linux tc/netem, and triggers automated attack scenarios.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # ML IDS THESIS/
LAB_DIR   = REPO_ROOT / "lab"
PCAP_DIR  = REPO_ROOT / "data" / "raw" / "pcaps"
MANIFEST_DIR = LAB_DIR / "manifests"

ATTACKER  = "ids_v2_attacker"
CLIENT    = "ids_v2_client"
TARGET    = "ids_v2_target"
TARGET_IP = "172.20.0.30"
MONITOR   = "ids_v2_monitor"

# CICIDS2017 Monday PCAP replayed as benign baseline.
# Path inside monitor container (mounted from data/raw/pcaps/ on host).
BENIGN_TEMPLATE_IN_CONTAINER = "/data/pcaps/benign_template.pcap"

# ── Domain session plans ──────────────────────────────────────────────────────
# Format: (attack_family, variant_id, generator, config_dict)
DOMAIN_PLANS = {
    # ── D01: Moderate intensity, medium-duration sessions ─────────────────────
    # Calibrated against CICIDS2017 timing: DoS attacks ran 5-15 min windows,
    # brute-force ran until wordlist exhausted (~10-20 min), scans ~5 min.
    # Duration unit: seconds. Rates: packets-per-second (pps) or Mbps.
    "D01": [
        ("SYN_FLOOD",     "sf-med-1",   "hping3",  {"rate_pps": 250, "pkt_size": 40,  "duration": 300, "port": 80}),
        ("SYN_FLOOD",     "sf-med-2",   "hping3",  {"rate_pps": 300, "pkt_size": 64,  "duration": 300, "port": 8080}),
        # 2026-08-19 (VOLUME SCALE-UP -- see problems_notes.txt): pooled natural
        # flow counts across D01-D04 were far below the 25,000/class target this
        # round requires -- ICMP_FLOOD 2,802 and UDP_DDOS 11,022 total (verified
        # directly from data/processed/*/labelled.parquet, not estimated). Both
        # scale as num_sources * ceil(duration/120) (confirmed from
        # icmp_flood.sh/udp_ddos.sh's own documented formula) -- num_sources
        # added explicitly (150, near the safe ceiling: BASE=100 + up to 154
        # stays inside the /24 subnet's .254 max) instead of relying on the
        # scripts' own default of 100. A 3rd session per domain is also added
        # for every under-target family below. This alone will not certainly hit
        # 25k -- see the mandatory pilot-measurement step before the full run.
        ("ICMP_FLOOD",    "ic-med-1",   "hping3",  {"rate_pps": 150, "pkt_size": 64,  "duration": 300, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-med-2",   "hping3",  {"rate_pps": 200, "pkt_size": 256, "duration": 300, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-med-3",   "hping3",  {"rate_pps": 175, "pkt_size": 128, "duration": 600, "num_sources": 150}),
        # 2026-08-19 SECOND SCALE-UP (see problems_notes.txt "D01 REAL DATA
        # SHOWS SEVERAL FAMILIES STILL SHORT"): D01's first-round fix (3
        # variants, num_sources=150) measured at 3,000 flows/domain -- pooled
        # projection ~12,000, still well under the 25,000 target. num_sources
        # is already near the /24 alias-space ceiling, so duration is the
        # remaining lever (flows scale with num_sources*ceil(duration/120)).
        # Two long-duration variants added to roughly triple this domain's
        # total ICMP attack-time rather than stretching one session to an
        # extreme length.
        ("ICMP_FLOOD",    "ic-med-4",   "hping3",  {"rate_pps": 180, "pkt_size": 96,  "duration": 1800, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-med-5",   "hping3",  {"rate_pps": 160, "pkt_size": 192, "duration": 1800, "num_sources": 150}),
        ("UDP_DDOS",      "ud-med-1",   "iperf3",  {"rate_mbps": 10, "pkt_size": 1024, "duration": 300, "port": 5001, "num_sources": 150}),
        # FOUND 2026-08-13 (see problems_notes.txt): DOMAIN_PLANS originally had
        # only 1 session variant for the 6 families below, violating Section 8C's
        # "≥2 independent sessions per attack family per domain" minimum (needed
        # for the L2 session-holdout / L3 config-holdout generalisation ladder to
        # even be constructible). Second variants added with genuine parameter
        # diversity below; SSH_BRUTE's 2nd variant also gets genuine tool
        # diversity (ncrack) now that ssh_brute.sh actually dispatches on `tool`.
        ("UDP_DDOS",      "ud-med-2",   "iperf3",  {"rate_mbps": 20, "pkt_size": 512,  "duration": 300, "port": 5002, "num_sources": 150}),
        ("UDP_DDOS",      "ud-med-3",   "iperf3",  {"rate_mbps": 15, "pkt_size": 768,  "duration": 600, "port": 5000, "num_sources": 150}),
        # FTP_BRUTE/SSH_BRUTE: hydra has no early-exit flag here (-f not passed),
        # so it should try the FULL wordlist (currently 1000 entries) regardless
        # of thread count -- observed flow counts (FTP_BRUTE 6,567, SSH_BRUTE
        # 2,822 pooled) are well under that, most likely target-side connection
        # throttling (vsftpd/sshd defaults), not confirmed. 3rd session variants
        # added for volume; the pilot-measurement step should confirm whether
        # this alone is enough or whether the wordlists also need expanding.
        # 2026-08-19 WORDLIST EXPANSION + THREAD BUMP (see problems_notes.txt):
        # D01 measured FTP_BRUTE at ~815-835 flows/session regardless of
        # thread count (4, 6, and 8 threads all landed in that same narrow
        # band) and SSH_BRUTE hydra at ~376-426 flows/session (4 vs 6
        # threads, same result) -- direct evidence that flow count here is
        # bounded by wordlist size (both were 1000 entries) divided by
        # however many attempts each TCP connection sustains before the
        # server disconnects it (sshd's MaxAuthTries, default 6), NOT by
        # thread count. Thread count only changed wall-clock time (4 threads:
        # 14m13s for FTP, 38m53s for SSH; 8 threads: 7m3s for FTP) --
        # confirmed via the manifest's own recorded session windows, not
        # guessed. Fix: both wordlists expanded 1000 -> 8000 entries
        # (lab/wordlists/*.txt, all original entries preserved, common
        # mutation-rule expansion) for an ~8x volume lever that actually
        # works, PLUS a moderate thread bump (not extreme -- sshd's
        # MaxStartups default 10:30:100 starts probabilistically dropping
        # connections past 10 concurrent, so staying in the ~15-25 range
        # trades some parallelism for stability) purely to keep the now
        # 8x-longer sessions from taking 8x longer in wall-clock too, since
        # thread count doesn't cost anything in flow yield either way.
        ("FTP_BRUTE",     "fb-hydra-1", "hydra",   {"threads": 15, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-2", "hydra",   {"threads": 20, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-3", "hydra",   {"threads": 18, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("SSH_BRUTE",     "sb-hydra-1", "hydra",   {"threads": 12, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("SSH_BRUTE",     "sb-ncrack-1","ncrack",  {"threads": 15,  "wordlist": "/wordlists/ssh_passwords.txt", "tool": "ncrack"}),
        ("SSH_BRUTE",     "sb-hydra-2", "hydra",   {"threads": 18, "wordlist": "/wordlists/ssh_passwords.txt"}),
        # PORT_SCAN: flow count scales with port-range width. ps-full-* uses the
        # entire port space (-p-, same technique D05's ps-held-1 already used)
        # instead of --top-ports N, which should scale flows by roughly 65x
        # over --top-ports 1000.
        ("PORT_SCAN",     "ps-syn-1",   "nmap",    {"type": "-sS", "speed": "-T3", "ports": "--top-ports 1000"}),
        ("PORT_SCAN",     "ps-conn-1",  "nmap",    {"type": "-sT", "speed": "-T4", "ports": "--top-ports 500"}),
        ("PORT_SCAN",     "ps-full-1",  "nmap",    {"type": "-sS", "speed": "-T4", "ports": "-p-"}),
        ("DOS_HULK",      "dh-1",       "hulk",    {"threads": 500, "url": f"http://{TARGET_IP}/index.html",   "duration": 300}),
        ("DOS_HULK",      "dh-2",       "hulk",    {"threads": 250, "url": f"http://{TARGET_IP}/large_file.txt", "duration": 300}),
        # DOS_SLOWLORIS: flow count scales roughly with socket count. 19,106
        # pooled -- already close to 25k, a modest bump + 3rd session is enough
        # (not scaled as aggressively as the other under-target families).
        ("DOS_SLOWLORIS", "ds-1",       "slowloris",{"sockets": 150, "interval": 15, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-2",       "slowloris",{"sockets": 100, "interval": 20, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-3",       "slowloris",{"sockets": 250, "interval": 10, "duration": 300}),
        # 2026-08-19: D01 measured 6,559 flows/domain -- pooled projection
        # ~26,236, only ~105% of the 25,000 target (the task explicitly wants
        # a COMFORTABLE margin, not a razor's edge that one slightly-lower
        # domain could tip under). Small 4th variant added for headroom;
        # socket-count scaling already confirmed to work as expected here
        # (unlike the brute-force families), so no wordlist-style rework
        # needed, just more of the same lever.
        ("DOS_SLOWLORIS", "ds-4",       "slowloris",{"sockets": 200, "interval": 12, "duration": 450}),
        # Benign-only sessions: longer windows → realistic flow_duration distribution
        (None, "benign-only-1", None, {"duration": 900}),
        (None, "benign-only-2", None, {"duration": 900}),
        (None, "benign-only-3", None, {"duration": 900}),
    ],
    # ── D02: High intensity, shorter bursts (stress variants) ─────────────────
    "D02": [
        ("SYN_FLOOD",     "sf-high-1",  "hping3",  {"rate_pps": 500, "pkt_size": 40,   "duration": 240, "port": 80}),
        ("SYN_FLOOD",     "sf-high-2",  "hping3",  {"rate_pps": 600, "pkt_size": 40,   "duration": 240, "port": 443}),
        # 2026-08-19 VOLUME SCALE-UP -- see D01's block above for the full
        # reasoning (num_sources near the safe /24 ceiling, 3rd session per
        # under-target family, pilot-measure before the full run).
        ("ICMP_FLOOD",    "ic-high-1",  "hping3",  {"rate_pps": 400, "pkt_size": 64,   "duration": 240, "num_sources": 150}),
        # FOUND 2026-08-13 (Section 8C ≥2-sessions-per-family audit, see
        # problems_notes.txt): second variants added below for the 7 families
        # that only had 1 session in this domain.
        ("ICMP_FLOOD",    "ic-high-2",  "hping3",  {"rate_pps": 350, "pkt_size": 128,  "duration": 240, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-high-3",  "hping3",  {"rate_pps": 375, "pkt_size": 96,   "duration": 480, "num_sources": 150}),
        # 2026-08-19 SECOND SCALE-UP -- see D01's block for the full
        # measured-evidence reasoning (num_sources maxed, duration is the
        # remaining lever).
        ("ICMP_FLOOD",    "ic-high-4",  "hping3",  {"rate_pps": 400, "pkt_size": 96,  "duration": 1500, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-high-5",  "hping3",  {"rate_pps": 380, "pkt_size": 160, "duration": 1500, "num_sources": 150}),
        ("UDP_DDOS",      "ud-high-1",  "iperf3",  {"rate_mbps": 50, "pkt_size": 1500, "duration": 240, "port": 5002, "num_sources": 150}),
        ("UDP_DDOS",      "ud-high-2",  "iperf3",  {"rate_mbps": 40, "pkt_size": 1024, "duration": 240, "port": 5000, "num_sources": 150}),
        ("UDP_DDOS",      "ud-high-3",  "iperf3",  {"rate_mbps": 45, "pkt_size": 1280, "duration": 480, "port": 5001, "num_sources": 150}),
        # 2026-08-19 WORDLIST EXPANSION + THREAD BUMP -- see D01's block for
        # the full measured-evidence reasoning (wordlists now 8000 entries,
        # shared across all domains via the same mounted files; thread
        # counts bumped to keep wall-clock time reasonable, capped ~15-25).
        ("FTP_BRUTE",     "fb-hydra-2", "hydra",   {"threads": 20, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-3", "hydra",   {"threads": 24, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-4", "hydra",   {"threads": 22, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("SSH_BRUTE",     "sb-hydra-2", "hydra",   {"threads": 18, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("SSH_BRUTE",     "sb-ncrack-2","ncrack",  {"threads": 20, "wordlist": "/wordlists/ssh_passwords.txt", "tool": "ncrack"}),
        ("SSH_BRUTE",     "sb-hydra-3", "hydra",   {"threads": 20, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("PORT_SCAN",     "ps-aggr-1",  "nmap",    {"type": "-A",  "speed": "-T4", "ports": "--top-ports 100"}),
        ("PORT_SCAN",     "ps-aggr-2",  "nmap",    {"type": "-sS", "speed": "-T5", "ports": "--top-ports 200"}),
        ("PORT_SCAN",     "ps-full-1",  "nmap",    {"type": "-sT", "speed": "-T5", "ports": "-p-"}),
        ("DOS_HULK",      "dh-2",       "hulk",    {"threads": 200, "url": f"http://{TARGET_IP}/large_file.txt", "duration": 300}),
        ("DOS_HULK",      "dh-3",       "hulk",    {"threads": 400, "url": f"http://{TARGET_IP}/index.html",     "duration": 240}),
        ("DOS_SLOWLORIS", "ds-2",       "slowloris",{"sockets": 200, "interval": 10, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-3",       "slowloris",{"sockets": 300, "interval": 8,  "duration": 240}),
        ("DOS_SLOWLORIS", "ds-4",       "slowloris",{"sockets": 350, "interval": 8,  "duration": 300}),
        # 2026-08-19: small extra-margin variant, see D01's identical note.
        ("DOS_SLOWLORIS", "ds-5",       "slowloris",{"sockets": 250, "interval": 10, "duration": 400}),
        (None, "benign-only-1", None, {"duration": 900}),
        (None, "benign-only-2", None, {"duration": 900}),
        (None, "benign-only-3", None, {"duration": 900}),
    ],
    # ── D03: Low intensity, extended sessions (stealthy variants) ─────────────
    "D03": [
        ("SYN_FLOOD",     "sf-low-1",   "hping3",  {"rate_pps": 50,  "pkt_size": 40,  "duration": 600, "port": 80}),
        ("SYN_FLOOD",     "sf-low-2",   "hping3",  {"rate_pps": 100, "pkt_size": 40,  "duration": 600, "port": 443}),
        # 2026-08-19 VOLUME SCALE-UP -- see D01's block above for the full
        # reasoning.
        ("ICMP_FLOOD",    "ic-low-1",   "hping3",  {"rate_pps": 30,  "pkt_size": 64,  "duration": 600, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-low-2",   "hping3",  {"rate_pps": 60,  "pkt_size": 128, "duration": 600, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-low-3",   "hping3",  {"rate_pps": 45,  "pkt_size": 96,  "duration": 900, "num_sources": 150}),
        # 2026-08-19 SECOND SCALE-UP -- see D01's block for the full
        # measured-evidence reasoning. D03 already had the longest base
        # durations of any domain (2100s total across 3 variants), so the
        # added variants here are slightly shorter than D01/D02's to land at
        # a similar ~3x total-duration multiplier rather than compounding
        # on an already-larger base.
        ("ICMP_FLOOD",    "ic-low-4",   "hping3",  {"rate_pps": 40,  "pkt_size": 80,  "duration": 1500, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-low-5",   "hping3",  {"rate_pps": 50,  "pkt_size": 112, "duration": 1500, "num_sources": 150}),
        ("UDP_DDOS",      "ud-low-1",   "iperf3",  {"rate_mbps": 1,  "pkt_size": 1024, "duration": 600, "port": 5000, "num_sources": 150}),
        ("UDP_DDOS",      "ud-low-2",   "iperf3",  {"rate_mbps": 2,  "pkt_size": 512,  "duration": 600, "port": 5000, "num_sources": 150}),
        ("UDP_DDOS",      "ud-low-3",   "iperf3",  {"rate_mbps": 1.5,"pkt_size": 768,  "duration": 900, "port": 5001, "num_sources": 150}),
        # 2026-08-20 (found while D02 was finishing -- see problems_notes.txt
        # "UDP_DDOS RAZOR'S-EDGE FOUND MID-RUN"): D01 (1200s total) yielded
        # 6,863 flows, D02 (960s total) yielded 5,573 -- a remarkably
        # consistent ~5.7-5.8 flows/sec at num_sources=150, but averaged
        # across D01+D02 that pooled-projects to only ~24,872, under the
        # 25,000 target. Same lever as ICMP_FLOOD (num_sources already
        # maxed, duration is what's left). D01/D02 are already validated and
        # not being repatched -- their real combined total (12,436) plus a
        # comfortably-boosted D03+D04 clears 25,000 pooled without touching
        # them. D03 already had the longest base (2100s) of any domain, so a
        # smaller relative add here than D04 gets.
        ("UDP_DDOS",      "ud-low-4",   "iperf3",  {"rate_mbps": 1.2,"pkt_size": 640,  "duration": 1500, "port": 5000, "num_sources": 150}),
        # FOUND 2026-08-13 (see problems_notes.txt "MEDUSA/NCRACK TOOL-DISPATCH
        # MISLABELING"): "medusa" was never a real tool here — run_attack() only
        # ever calls /attacks/{family}.sh regardless of the `generator` field,
        # ftp_brute.sh unconditionally ran hydra, and medusa isn't even installed
        # in the attacker image (confirmed via Dockerfile). Relabelled honestly
        # to hydra rather than doing a same-night Dockerfile+rebuild+debug cycle
        # to add real medusa support. sb-ncrack-1 gets "tool": "ncrack" added so
        # it genuinely dispatches to ncrack now that ssh_brute.sh supports it
        # (ncrack IS installed). Second variants also added per family below to
        # satisfy Section 8C's ≥2-sessions-per-family minimum.
        # 2026-08-19 WORDLIST EXPANSION + THREAD BUMP -- see D01's block for
        # the full measured-evidence reasoning.
        ("FTP_BRUTE",     "fb-medusa-1","hydra",   {"threads": 16, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-2", "hydra",   {"threads": 18, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-medusa-2","hydra",   {"threads": 17, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("SSH_BRUTE",     "sb-ncrack-1","ncrack",  {"threads": 15, "wordlist": "/wordlists/ssh_passwords.txt", "tool": "ncrack"}),
        ("SSH_BRUTE",     "sb-hydra-1", "hydra",   {"threads": 12, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("SSH_BRUTE",     "sb-ncrack-2","ncrack",  {"threads": 16, "wordlist": "/wordlists/ssh_passwords.txt", "tool": "ncrack"}),
        ("PORT_SCAN",     "ps-syn-2",   "nmap",    {"type": "-sS", "speed": "-T2", "ports": "--top-ports 1000"}),
        ("PORT_SCAN",     "ps-conn-1",  "nmap",    {"type": "-sT", "speed": "-T3", "ports": "--top-ports 1000"}),
        ("PORT_SCAN",     "ps-full-1",  "nmap",    {"type": "-sS", "speed": "-T2", "ports": "-p-"}),
        ("DOS_HULK",      "dh-1",       "hulk",    {"threads": 500, "url": f"http://{TARGET_IP}/index.html",   "duration": 300}),
        ("DOS_HULK",      "dh-2",       "hulk",    {"threads": 100, "url": f"http://{TARGET_IP}/large_file.txt", "duration": 600}),
        ("DOS_SLOWLORIS", "ds-1",       "slowloris",{"sockets": 150, "interval": 15, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-2",       "slowloris",{"sockets": 80,  "interval": 25, "duration": 600}),
        ("DOS_SLOWLORIS", "ds-3",       "slowloris",{"sockets": 200, "interval": 18, "duration": 450}),
        # 2026-08-19: small extra-margin variant, see D01's identical note.
        ("DOS_SLOWLORIS", "ds-4",       "slowloris",{"sockets": 220, "interval": 15, "duration": 400}),
        (None, "benign-only-1", None, {"duration": 900}),
        (None, "benign-only-2", None, {"duration": 900}),
        (None, "benign-only-3", None, {"duration": 900}),
    ],
    # ── D04: Mixed intensity (full diversity for LODO cross-validation) ────────
    "D04": [
        ("SYN_FLOOD",     "sf-low-1",   "hping3",  {"rate_pps": 50,  "pkt_size": 40, "duration": 600, "port": 80}),
        ("SYN_FLOOD",     "sf-med-1",   "hping3",  {"rate_pps": 250, "pkt_size": 40, "duration": 300, "port": 80}),
        ("SYN_FLOOD",     "sf-high-1",  "hping3",  {"rate_pps": 500, "pkt_size": 40, "duration": 240, "port": 80}),
        # 2026-08-19 VOLUME SCALE-UP -- see D01's block above for the full
        # reasoning. D04 originally only had 1 ICMP_FLOOD session -- bumped
        # to 3 (low/med/high, matching D04's own "mixed intensity" theme)
        # like the other under-target families below.
        ("ICMP_FLOOD",    "ic-low-1",   "hping3",  {"rate_pps": 30,  "pkt_size": 64, "duration": 600, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-med-1",   "hping3",  {"rate_pps": 150, "pkt_size": 64, "duration": 300, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-high-1",  "hping3",  {"rate_pps": 350, "pkt_size": 96, "duration": 240, "num_sources": 150}),
        # 2026-08-19 SECOND SCALE-UP -- see D01's block for the full
        # measured-evidence reasoning.
        ("ICMP_FLOOD",    "ic-vol-1",   "hping3",  {"rate_pps": 200, "pkt_size": 96, "duration": 1500, "num_sources": 150}),
        ("ICMP_FLOOD",    "ic-vol-2",   "hping3",  {"rate_pps": 220, "pkt_size": 128, "duration": 1500, "num_sources": 150}),
        # FOUND 2026-08-16: relabelled from "iperf3" -- stale since the
        # 2026-08-14 rewrite moved UDP_DDOS to the same multi-source hping3
        # alias emulation ICMP_FLOOD uses (iperf3 is no longer invoked at
        # all). Functionally harmless either way (run_attack() always
        # dispatches on `family`, never on this label -- same as the
        # medusa/ncrack finding below), but this label IS written verbatim
        # into the manifest's "generator" field, which is meant to be an
        # auditable record  -- fixing it before D04 generates rather than
        # after, same principle as the medusa relabel.
        ("UDP_DDOS",      "ud-low-1",   "hping3",  {"rate_mbps": 1,  "pkt_size": 1024, "duration": 600, "port": 5000, "num_sources": 150}),
        ("UDP_DDOS",      "ud-high-1",  "hping3",  {"rate_mbps": 50, "pkt_size": 1500, "duration": 240, "port": 5002, "num_sources": 150}),
        ("UDP_DDOS",      "ud-med-1",   "hping3",  {"rate_mbps": 15, "pkt_size": 900,  "duration": 480, "port": 5001, "num_sources": 150}),
        # 2026-08-20: see D03's identical note on this same fix.
        ("UDP_DDOS",      "ud-vol-1",   "hping3",  {"rate_mbps": 10, "pkt_size": 800,  "duration": 1500, "port": 5000, "num_sources": 150}),
        # 2026-08-19 WORDLIST EXPANSION + THREAD BUMP -- see D01's block for
        # the full measured-evidence reasoning.
        ("FTP_BRUTE",     "fb-hydra-1", "hydra",   {"threads": 16, "wordlist": "/wordlists/ftp_passwords.txt"}),
        # FOUND 2026-08-13: relabelled from "medusa" (never a real dispatch,
        # medusa isn't installed) to "hydra" — see D03's identical comment and
        # problems_notes.txt. sb-ncrack-1 gets "tool": "ncrack" so it genuinely
        # dispatches to ncrack now that ssh_brute.sh supports it.
        ("FTP_BRUTE",     "fb-medusa-1","hydra",   {"threads": 16, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("FTP_BRUTE",     "fb-hydra-2", "hydra",   {"threads": 19, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("SSH_BRUTE",     "sb-hydra-1", "hydra",   {"threads": 12, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("SSH_BRUTE",     "sb-ncrack-1","ncrack",  {"threads": 15, "wordlist": "/wordlists/ssh_passwords.txt", "tool": "ncrack"}),
        ("SSH_BRUTE",     "sb-hydra-2", "hydra",   {"threads": 19, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("PORT_SCAN",     "ps-syn-1",   "nmap",    {"type": "-sS", "speed": "-T3", "ports": "--top-ports 1000"}),
        ("PORT_SCAN",     "ps-conn-1",  "nmap",    {"type": "-sT", "speed": "-T3", "ports": "--top-ports 1000"}),
        ("PORT_SCAN",     "ps-full-1",  "nmap",    {"type": "-sS", "speed": "-T3", "ports": "-p-"}),
        ("DOS_HULK",      "dh-1",       "hulk",    {"threads": 500, "url": f"http://{TARGET_IP}/index.html",        "duration": 300}),
        ("DOS_HULK",      "dh-2",       "hulk",    {"threads": 200, "url": f"http://{TARGET_IP}/large_file.txt",    "duration": 300}),
        ("DOS_SLOWLORIS", "ds-1",       "slowloris",{"sockets": 150, "interval": 15, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-2",       "slowloris",{"sockets": 200, "interval": 10, "duration": 300}),
        ("DOS_SLOWLORIS", "ds-3",       "slowloris",{"sockets": 275, "interval": 9,  "duration": 350}),
        # 2026-08-19: small extra-margin variant, see D01's identical note.
        ("DOS_SLOWLORIS", "ds-4",       "slowloris",{"sockets": 230, "interval": 12, "duration": 400}),
        (None, "benign-only-1", None, {"duration": 900}),
        (None, "benign-only-2", None, {"duration": 900}),
        (None, "benign-only-3", None, {"duration": 900}),
        (None, "benign-only-4", None, {"duration": 900}),
    ],
    # ── D05: Compound holdout — unseen configs + new operating condition ───────
    "D05": [
        ("SYN_FLOOD",     "sf-held-1",  "hping3",  {"rate_pps": 150, "pkt_size": 40,  "duration": 300, "port": 80}),
        ("SYN_FLOOD",     "sf-held-2",  "hping3",  {"rate_pps": 750, "pkt_size": 40,  "duration": 240, "port": 80}),
        ("ICMP_FLOOD",    "ic-held-1",  "hping3",  {"rate_pps": 100, "pkt_size": 64,  "duration": 300}),
        ("UDP_DDOS",      "ud-held-1",  "iperf3",  {"rate_mbps": 5,  "pkt_size": 768, "duration": 300, "port": 5000}),
        ("FTP_BRUTE",     "fb-held-1",  "hydra",   {"threads": 6, "wordlist": "/wordlists/ftp_passwords.txt"}),
        ("SSH_BRUTE",     "sb-held-1",  "hydra",   {"threads": 6, "wordlist": "/wordlists/ssh_passwords.txt"}),
        ("PORT_SCAN",     "ps-held-1",  "nmap",    {"type": "-sS", "speed": "-T4", "ports": "-p-"}),
        ("DOS_HULK",      "dh-held-1",  "hulk",    {"threads": 1000, "url": f"http://{TARGET_IP}/index.html", "duration": 300}),
        ("DOS_SLOWLORIS", "ds-held-1",  "slowloris",{"sockets": 300, "interval": 20, "duration": 300}),
        (None, "benign-only-1", None, {"duration": 900}),
        (None, "benign-only-2", None, {"duration": 900}),
        (None, "benign-only-3", None, {"duration": 900}),
    ],
}

PILOT_PLAN = [
    # Pilot: short enough to finish quickly, long enough to verify realistic flow features
    # ICMP_FLOOD chosen because it's fast to verify flow counts in Wireshark/CICFlowMeter
    #
    # EXPANDED 2026-08-13: V2.1 Section 8B mandates all 6 core attack families in the
    # pilot's verification scope (SYN_FLOOD, ICMP_FLOOD, UDP_DDOS, FTP_BRUTE, SSH_BRUTE,
    # PORT_SCAN) but this plan originally only ran 3 of them — found during the Section
    # 21/22 pre-generation review, after the first 3 families had already passed cleanly.
    # Added UDP_DDOS/SSH_BRUTE/PORT_SCAN below with config values copied verbatim from
    # DOMAIN_PLANS["D01"], cross-checked against the actual attack scripts' argument
    # names (not guessed): udp_ddos.sh reads rate_mbps/pkt_size/duration/port,
    # ssh_brute.sh reads threads/wordlist, port_scan.sh reads type/speed/ports — all
    # three confirmed to match by reading the scripts directly. UDP_DDOS's duration
    # shortened from D01's 300s to 180s to match the other timed pilot attacks; hydra
    # (SSH_BRUTE) and nmap (PORT_SCAN) have no duration config since they run until
    # wordlist-exhaustion/scan-completion respectively, same pattern already validated
    # by FTP_BRUTE.
    ("SYN_FLOOD",   "sf-med-1",   "hping3", {"rate_pps": 250, "pkt_size": 40, "duration": 180, "port": 80}),
    ("ICMP_FLOOD",  "ic-med-1",   "hping3", {"rate_pps": 150, "pkt_size": 64, "duration": 180}),
    ("UDP_DDOS",    "ud-med-1",   "iperf3", {"rate_mbps": 10, "pkt_size": 1024, "duration": 180, "port": 5001}),
    ("FTP_BRUTE",   "fb-hydra-1", "hydra",  {"threads": 4, "wordlist": "/wordlists/ftp_passwords.txt"}),
    ("SSH_BRUTE",   "sb-hydra-1", "hydra",  {"threads": 4, "wordlist": "/wordlists/ssh_passwords.txt"}),
    ("PORT_SCAN",   "ps-syn-1",   "nmap",   {"type": "-sS", "speed": "-T3", "ports": "--top-ports 1000"}),
    (None, "benign-only-1", None, {"duration": 600}),
]

# ── Benign traffic profiles by domain ────────────────────────────────────────
BENIGN_PROFILES = {
    "D01": ["http", "dns", "icmp"],
    "D02": ["http", "ftp", "dns", "ssh"],
    "D03": ["https", "ssh", "icmp", "ftp"],
    "D04": ["http", "https", "dns", "ftp", "ssh", "icmp"],
    "D05": ["http", "dns", "ssh"],
    "PILOT": ["http", "dns"],
}

# ── Helpers ──────────────────────────────────────────────────────────────────

def dexec(container: str, cmd: list[str], detach: bool = False,
          timeout: float | None = None) -> subprocess.CompletedProcess:
    """Execute internal routine."""
    args = ["docker", "exec"]
    if detach:
        args.append("-d")
    args += [container] + cmd
    if detach:
        return subprocess.Popen(args)
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # FOUND 2026-08-22: FTP_BRUTE/SSH_BRUTE/PORT_SCAN sessions have no
        # 'duration' config at all -- hydra/ncrack/medusa/nmap just run until
        # they exhaust their own wordlist/target space, with no external
        # bound. Fine pre-netem; with per-session network delay/jitter/loss
        # now applied (2026-08-19), each interactive auth attempt costs a
        # full extra round-trip, and D04's sb-hydra-2 (8000-entry wordlist,
        # busy_lan profile) measured ~164 tries/min -- hydra's own [STATUS]
        # line projected ~48 MORE HOURS to exhaust the wordlist. Flood-type
        # attacks (SYN/ICMP/UDP/DOS_HULK/SLOWLORIS) are unaffected -- they
        # self-terminate via their own internal 'duration' sleep loop and
        # return in seconds, well under any sane cap. Return a synthetic
        # CompletedProcess (returncode 124, the standard shell timeout exit
        # code) instead of letting TimeoutExpired propagate and kill the
        # whole domain run over one oversized wordlist.
        return subprocess.CompletedProcess(
            args, 124,
            stdout=(e.stdout or ""),
            stderr=(e.stderr or "") + f"\n[dexec] TIMEOUT after {timeout}s -- killed, moving on."
        )

def ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

# FOUND 2026-08-19 (Section 14 drift diagnostics, autonomous overnight
# continuation): setup_netem() applied ONE FIXED profile (1ms/0.3ms/0.01%)
# to every session in every domain -- confirmed by reading the code, not
# assumed. A domain classifier (lab V2 data vs CICIDS2017) scored AUC=0.9846,
# driven almost entirely by rate/timing features (bwd_pkts_s, flow_pkts_s,
# fwd_pkts_s) rather than volumetric ones -- a uniform, artificial network
# condition across 100% of lab traffic is a direct, plausible mechanism for
# exactly that signature (real captures have variable network conditions;
# ours had one constant one). These 5 profiles span a realistic range of
# real-world LAN/WAN paths (clean local LAN through a congested/wireless-like
# path) -- deliberately NOT extreme/synthetic values, staying inside what a
# real corporate network would plausibly show, so this varies the SAME kind
# of realism CICIDS2017's physical capture had, rather than inventing an
# unrealistic new artifact. Selected per-session (not per-domain) via a
# session-index-seeded deterministic RNG so the exact sequence is
# reproducible from the same seed, and recorded in the manifest by
# write_manifest() for auditability  -- see NETEM_PROFILES/
# pick_netem_profile() below and the write_manifest()/run_session() call
# sites that thread the choice through.
NETEM_PROFILES = [
    {"name": "clean_lan",        "delay": "0.5ms", "jitter": "0.1ms", "loss": "0.001%"},
    {"name": "typical_lan",      "delay": "1ms",   "jitter": "0.3ms", "loss": "0.01%"},   # the old fixed default
    {"name": "busy_lan",         "delay": "2ms",   "jitter": "0.5ms", "loss": "0.05%"},
    {"name": "wan_like",         "delay": "5ms",   "jitter": "1ms",   "loss": "0.1%"},
    {"name": "congested_wireless","delay": "8ms",   "jitter": "2ms",  "loss": "0.2%"},
]

NETEM_SEED = 42  # matches this project's standard seed used everywhere else

def pick_netem_profile(session_index: int, seed: int = NETEM_SEED) -> dict:
    """Execute internal routine."""
    import random
    rng = random.Random(seed + session_index)
    return rng.choice(NETEM_PROFILES)

def setup_netem(profile: dict | None = None) -> None:
    """Execute internal routine."""
    p = profile or {"delay": "1ms", "jitter": "0.3ms", "loss": "0.01%"}
    DELAY, JITTER, LOSS = p["delay"], p["jitter"], p["loss"]

    for container in [ATTACKER, CLIENT, TARGET]:
        # Delete existing qdisc first (idempotent)
        subprocess.run(
            ["docker", "exec", container,
             "tc", "qdisc", "del", "dev", "eth0", "root"],
            capture_output=True
        )
        r = subprocess.run(
            ["docker", "exec", container,
             "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
             "delay", DELAY, JITTER, "loss", LOSS],
            capture_output=True, text=True
        )
        if r.returncode == 0:
            print(f"  [netem] {container}: delay={DELAY} jitter={JITTER} loss={LOSS}")
        else:
            # Non-fatal: Docker Desktop on Windows may block tc in some kernels
            print(f"  [netem] {container}: skipped ({r.stderr.strip()[:60]})")

def find_bridge_interface() -> str:
    """Execute internal routine."""
    result = subprocess.run(
        ["docker", "network", "ls", "--filter", "name=lab_net", "--format", "{{.ID}}"],
        capture_output=True, text=True
    )
    net_id = result.stdout.strip()[:12]
    if not net_id:
        raise RuntimeError("Docker network 'lab_net' not found. Is docker compose up?")
    return f"br-{net_id}"

def start_capture(pcap_path: str, bridge_if: str) -> subprocess.Popen:
    """Execute internal routine."""
    proc = subprocess.Popen([
        "docker", "exec", MONITOR,
        "tcpdump", "-i", bridge_if, "-s", "0", "-n", "-w", f"/data/pcaps/{pcap_path}"
    ])
    time.sleep(1.5)  # let tcpdump initialise
    print(f"  [capture] Started → /data/pcaps/{pcap_path}")
    return proc

def stop_capture(proc: subprocess.Popen) -> None:
    """Execute internal routine."""
    # BUG FOUND 2026-08-13: this used to call proc.terminate()/proc.wait() on
    # the LOCAL `docker exec` client Popen handle. On Windows, Popen.terminate()
    # calls the Win32 TerminateProcess() API — an unconditional, immediate kill
    # of the local docker-cli process, with NO signal actually delivered to it.
    # Docker's exec signal-forwarding (client receives SIGTERM -> forwards a
    # stop to the container-side process) depends on the client genuinely
    # receiving a POSIX signal to trigger that forwarding; TerminateProcess
    # bypasses that entirely. Net effect: the local docker.exe was killed, but
    # the actual tcpdump process INSIDE the container was never told to stop —
    # it kept capturing indefinitely, for as long as the container stayed
    # alive. Confirmed 2026-08-13: after a pilot run, all 4 session PCAPs
    # (which orchestrate.py itself verified at 16-940MB right after each
    # session) had silently grown to 12-14GB overnight, all sharing the exact
    # same final mtime — the moment the host machine lost power, not any of
    # this function's own stop calls. Fix: explicitly ask the container to
    # kill the process by name via a fresh `docker exec ... pkill`, which
    # doesn't depend on host-OS signal semantics working through the docker
    # CLI's local process at all.
    subprocess.run(["docker", "exec", MONITOR, "pkill", "-INT", "-f", "tcpdump"],
                    capture_output=True)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    # Belt and suspenders: if pkill -INT didn't finish it in time, force it.
    subprocess.run(["docker", "exec", MONITOR, "pkill", "-KILL", "-f", "tcpdump"],
                    capture_output=True)
    time.sleep(1)
    print("  [capture] Stopped and flushed.")

def start_benign(domain: str) -> list[subprocess.Popen]:
    """Execute internal routine."""
    bridge_if = find_bridge_interface()

    # Verify template exists
    r = subprocess.run(
        ["docker", "exec", MONITOR, "test", "-f", BENIGN_TEMPLATE_IN_CONTAINER],
        capture_output=True
    )
    if r.returncode != 0:
        raise FileNotFoundError(
            f"Benign template not found in monitor container: {BENIGN_TEMPLATE_IN_CONTAINER}\n"
            f"Run first: docker exec {MONITOR} python3 /scripts/prep_cicids_benign.py"
        )

    # NOTE (2026-08-12): --mtu-trunc / --mtu are NOT supported by the tcpreplay
    # build in this image ("illegal option -- mtu-trunc") — passing them made
    # tcpreplay exit immediately on every invocation, with the error swallowed
    # by stderr=DEVNULL below. Every prior session's benign injection silently
    # sent zero packets. Removed until we confirm a tcpreplay version that
    # supports them (or move MTU truncation to the tcprewrite prep step).
    proc = subprocess.Popen([
        "docker", "exec", MONITOR,
        "tcpreplay",
        "--loop=0",                          # loop indefinitely
        "--timer=gtod",                      # gettimeofday: highest precision timing
        "--quiet",                           # suppress flow.c DLT warnings (cosmetic only)
        f"--intf1={bridge_if}",              # inject onto Docker bridge
        BENIGN_TEMPLATE_IN_CONTAINER,
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2)  # let tcpreplay start and inject first packets

    # Fail fast and loud if tcpreplay exited immediately (bad args, bad pcap,
    # missing interface, etc.) instead of silently proceeding with no benign
    # traffic on the wire.
    if proc.poll() is not None:
        out, err = proc.communicate()
        raise RuntimeError(
            f"tcpreplay exited immediately (code {proc.returncode}) — benign "
            f"injection did not start.\nstdout: {out}\nstderr: {err}"
        )

    # Verify traffic is actually flowing on the bridge. Best-effort only: we
    # intentionally replay at the source pcap's original relative timing (not
    # --topspeed) to preserve human-scale IATs, so a quiet stretch near the
    # current replay position can legitimately take >10s to yield 5 packets.
    # A timeout here is NOT proof of failure — the fail-fast check above
    # already ruled out the "tcpreplay never started" case.
    try:
        check = subprocess.run(
            ["docker", "exec", MONITOR,
             "tcpdump", "-i", bridge_if, "-c", "5", "-q", "--immediate-mode"],
            capture_output=True, text=True, timeout=10
        )
        if "packets captured" in check.stderr or check.returncode == 0:
            print(f"  [benign:tcpreplay] Active on {bridge_if} — traffic verified ✓")
        else:
            print(f"  [benign:tcpreplay] Started on {bridge_if} (unverified — check PCAP size after session)")
    except subprocess.TimeoutExpired:
        print(f"  [benign:tcpreplay] Started on {bridge_if} (process still running; verification "
              f"check timed out — likely a quiet stretch in the source pcap, not a failure; "
              f"check PCAP size after session)")
    return [proc]

def stop_benign(procs: list[subprocess.Popen]) -> None:
    """Execute internal routine."""
    # Same bug and same fix as stop_capture() above: p.terminate() only ever
    # killed the local `docker exec` client process on Windows, never the
    # actual tcpreplay process inside the container. Every tcpreplay this
    # script ever started was almost certainly still running, all looping the
    # benign template concurrently, for as long as the monitor container
    # stayed up — a real explanation for both the runaway PCAP sizes and the
    # earlier open question in problems_notes.txt about whether benign volume
    # was compounding session over session (it was).
    subprocess.run(["docker", "exec", MONITOR, "pkill", "-INT", "-f", "tcpreplay"],
                    capture_output=True)
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass
    subprocess.run(["docker", "exec", MONITOR, "pkill", "-KILL", "-f", "tcpreplay"],
                    capture_output=True)
    time.sleep(1)

def run_attack(family: str, variant: str, generator: str, config: dict) -> None:
    """Trigger automated attack scripts inside attacker container."""
    script = f"/attacks/{family.lower()}.sh"
    env_args = [f"{k}={v}" for k, v in config.items()]
    cmd = ["bash", script] + env_args
    print(f"  [attack] {family}/{variant} via {generator}: {config}")
    # FOUND 2026-08-22: see dexec()'s matching comment -- FTP_BRUTE/SSH_BRUTE/
    # PORT_SCAN have no 'duration' in their config (interactive-attempt-bound,
    # not time-bound), so cap them externally. 300s comfortably matches the
    # scale of every other short session in the plan while still exercising a
    # real chunk of the wordlist/port range. Flood families all carry
    # 'duration' up to 1500s -- give those a generous 1800s ceiling as a
    # belt-and-suspenders safety net, not an expected trigger.
    attack_timeout = 300 if "duration" not in config else 1800
    # FOUND 2026-08-13: dexec()'s CompletedProcess was always discarded here,
    # so an attack script that fails instantly and silently (both the
    # tcpreplay --mtu-trunc bug on 2026-08-12 and the UDP_DDOS
    # iperf3-vs-plain-netcat bug found today did exactly this) looked
    # identical in the orchestrator's own log to a real, successful attack —
    # "[attack] ... via iperf3: {...}" printed either way, no matter whether
    # the target ever received a single flood packet. Every attack script
    # here also ends with `|| true`, so a non-zero exit code alone wasn't
    # even a reliable signal. Print the script's actual output so a
    # near-instant failure is visible in the run itself, not discoverable
    # only after the fact by noticing a suspiciously tiny/short flow count.
    result = dexec(ATTACKER, cmd, timeout=attack_timeout)
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"    [attack:stdout] {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"    [attack:stderr] {line}")
    if result.returncode == 124:
        print(f"  [TIMEOUT] {family}/{variant} exceeded {attack_timeout}s -- "
              f"stopped early, session keeps whatever attempts landed so far")
    elif result.returncode != 0:
        print(f"  [WARNING] Attack script exited {result.returncode} — "
              f"check output above; this attack session's flow data may be unreliable")
    # Belt-and-suspenders: subprocess timeout kills the local "docker exec"
    # client, which does not reliably kill the remote in-container process
    # (confirmed 2026-08-22 -- hydra kept running server-side after its host
    # subprocess was gone). The next session's own pkill sweep would catch
    # this eventually, but that leaves an orphaned brute-force run
    # contaminating THIS session's remaining warmup/drain window too.
    if result.returncode == 124:
        dexec(ATTACKER, ["pkill", "-9", "hydra"], detach=False)
        dexec(ATTACKER, ["pkill", "-9", "ncrack"], detach=False)
        dexec(ATTACKER, ["pkill", "-9", "-f", "medusa"], detach=False)
        dexec(ATTACKER, ["pkill", "-9", "nmap"], detach=False)

# First alias used by multi-source floods (172.20.0.100..). MUST match
# icmp_flood.sh's and udp_ddos.sh's BASE. Chosen above the CICIDS-Monday
# benign IP space (.3-.51) and the lab container IPs (.1/.5/.10/.20/.30) so
# the alias set is dedicated. Shared between ICMP_FLOOD and UDP_DDOS (added
# 2026-08-14, see the note below) — safe because sessions run strictly
# sequentially and each script fully removes its own aliases before
# returning, so the two families never hold aliases at the same time.
MULTI_SRC_ALIAS_BASE = 100

def write_manifest(
    domain: str, session_id: str, pcap_file: str,
    family: str | None, variant: str | None, generator: str | None,
    config: dict, start_utc: str, end_utc: str,
    capture_start_utc: str, capture_end_utc: str,
    concurrent_benign: list[str], experiment_id: str,
    netem_profile: dict | None = None,
) -> None:
    """Execute internal routine."""
    # NOTE (2026-08-13): start_time_utc/end_time_utc bracket only the
    # attack's own execution window (matches the V2.1 spec's Section 9
    # sample record), NOT the full PCAP. But the same section's labelling
    # LOGIC says concurrent benign should be labelled BENIGN for any flow
    # "within an ATTACK SESSION" whose IPs don't match the attack tuple —
    # i.e. the whole session, not just the narrow attack window. Without a
    # field for the actual capture bounds, label_flows() had no way to tell
    # "benign traffic during warmup/drain" apart from "outside this PCAP
    # entirely", and was labelling the former UNKNOWN. Caught via the V2.1
    # Section 21/22 approval review: FTP_BRUTE and ICMP_FLOOD pilot sessions
    # were at 7.4% and 51.6% per-session UNKNOWN respectively — both fail
    # spec's own <5%-per-session QG-03 threshold, even though the pooled
    # rate across all sessions looked fine at 2.5%. Added these two fields
    # so label_flows() can use the true capture window for the "concurrent
    # benign" branch while still using the narrower attack window for actual
    # attack_family matches.
    record = {
        "manifest_version": "2.1",
        "experiment_id": experiment_id,
        "domain_id": domain,
        "session_id": session_id,
        "pcap_file": pcap_file,
        "traffic_type": "ATTACK" if family else "BENIGN",
        "attack_family": family,
        "variant_id": variant,
        "generator": generator,
        "generator_version": None,  # filled in post-hoc from extraction_metadata.json
        "configuration": config,
        "src_ip": "172.20.0.10" if family else "172.20.0.20",
        "dst_ip": "172.20.0.30",
        "start_time_utc": start_utc,
        "end_time_utc": end_utc,
        "capture_start_utc": capture_start_utc,
        "capture_end_utc": capture_end_utc,
        "concurrent_benign": concurrent_benign,
        "netem_profile": netem_profile,  # 2026-08-19: per-session varied network condition, see NETEM_PROFILES
        "notes": ""
    }
    # Multi-source flood: record the exact REAL-alias set used so
    # label_flows() can match those flows by source-set membership instead of
    # a single src_ip. Originally ICMP-only (ICMP has no ports, so it is
    # flooded from N real aliases — see icmp_flood.sh). EXTENDED 2026-08-14
    # to UDP_DDOS too: the first udp_ddos.sh rewrite (hping3 --udp with a
    # per-packet source-PORT sweep) tested clean locally but came back with
    # ZERO extracted attacker<->target flows in the real environment —
    # confirmed by pulling the actual CICFlowMeter CSV, not guessed: the
    # port-sweep pcap (real, 46.7MB, hping3 itself reported ~29,764 packets
    # transmitted) produced a CSV with no row touching either the attacker or
    # target IP, while the OLD single-5-tuple iperf3 approach's CSV (same
    # CICFlowMeter binary) DID show correctly-labelled attacker<->target
    # rows, chunked every ~120s by CICFlowMeter's flow timeout. So CICFlowMeter's
    # UDP path — unlike its TCP path, which SYN_FLOOD's equivalent port sweep
    # fragments correctly into 162,121 flows — does not turn a rapid stream of
    # brand-new UDP 5-tuples into flow records; it drops them. Switched
    # udp_ddos.sh to the same multi-source-IP emulation as ICMP_FLOOD instead
    # (proven: 600 real flows), with each alias keeping a CONSTANT source
    # port (hping3 -k) so every alias reproduces the OLD single-stream
    # pattern already confirmed to extract cleanly — just replicated across
    # N aliases. num_sources defaults to 100, matching both scripts' default.
    if family in ("ICMP_FLOOD", "UDP_DDOS"):
        n_src = int(config.get("num_sources", 100))
        record["attack_src_ips"] = [f"172.20.0.{MULTI_SRC_ALIAS_BASE + k}" for k in range(n_src)]
    manifest_file = MANIFEST_DIR / f"{domain}_manifest.jsonl"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_file, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"  [manifest] Written → {manifest_file.name}")

# ── Session runner ────────────────────────────────────────────────────────────

def run_session(
    domain: str,
    family: str | None, variant: str, generator: str | None, config: dict,
    bridge_if: str, experiment_id: str,
    netem_profile: dict | None = None,
    # ── Timing calibration ────────────────────────────────────────────────────
    # warmup: benign runs before capture begins.
    #   120s ensures multiple full HTTP/FTP/SSH cycles are established
    #   so the capture starts with realistic concurrent-flow background,
    #   matching CICIDS2017's continuous-day benign baseline.
    benign_warmup_s: int = 120,
    # drain: benign continues after attack ends, then capture stops.
    #   60s ensures in-flight flows (esp. TCP FIN/RST) complete their
    #   handshakes and CICFlowMeter records them as properly terminated flows.
    benign_drain_s: int = 60,
) -> None:
    session_id = f"{domain}-{family or 'BENIGN'}-{variant}-{datetime.now(timezone.utc).strftime('%H%M%S')}"
    pcap_filename = f"{session_id}.pcap"
    benign_profile = BENIGN_PROFILES.get(domain, ["http", "dns"])

    print(f"\n{'='*60}")
    print(f"SESSION: {session_id}")
    print(f"{'='*60}")

    # 1. Reset attacker state (flush any lingering connections)
    # FOUND 2026-08-19 (D01 ds-3): this sweep only ever covered hping3/hydra,
    # so when DOS_SLOWLORIS's slowloris process outlived dexec()'s own return
    # (see dos_slowloris.sh's timeout -k comment for the full incident),
    # nothing here would have caught it -- it would have kept leaking into
    # every subsequent session indefinitely instead of being cleaned up at
    # the very next session boundary. Extended to every attack tool used
    # anywhere in DOMAIN_PLANS as a second, independent line of defence
    # (belt-and-suspenders alongside each script's own timeout -k), so any
    # future leak of this kind is bounded to at most one contaminated
    # session instead of accumulating across the rest of the domain. hulk.py
    # runs as a bare "python3" process (too generic a name to pkill safely by
    # short name -- would also match unrelated helper processes), so it's
    # matched by full cmdline via -f instead.
    dexec(ATTACKER, ["pkill", "-9", "hping3"], detach=False)
    dexec(ATTACKER, ["pkill", "-9", "hydra"], detach=False)
    dexec(ATTACKER, ["pkill", "-9", "slowloris"], detach=False)
    dexec(ATTACKER, ["pkill", "-9", "nmap"], detach=False)
    dexec(ATTACKER, ["pkill", "-9", "ncrack"], detach=False)
    dexec(ATTACKER, ["pkill", "-9", "-f", "hulk.py"], detach=False)
    time.sleep(1)

    # 2. Start benign traffic
    benign_procs = start_benign(domain)
    print(f"  [warmup] Benign warming up for {benign_warmup_s}s …")
    time.sleep(benign_warmup_s)

    # 3. Start capture
    capture_start_utc = datetime.now(timezone.utc).isoformat()
    cap_proc = start_capture(pcap_filename, bridge_if)

    # 4. Start attack (if this is an attack session)
    start_utc = datetime.now(timezone.utc).isoformat()
    if family is not None:
        run_attack(family, variant, generator, config)
    else:
        duration = config.get("duration", 300)
        print(f"  [benign-only] Running for {duration}s …")
        time.sleep(duration)

    end_utc = datetime.now(timezone.utc).isoformat()

    # 5. Drain benign
    print(f"  [drain] Benign draining for {benign_drain_s}s …")
    time.sleep(benign_drain_s)
    capture_end_utc = datetime.now(timezone.utc).isoformat()

    # 6. Stop capture
    stop_capture(cap_proc)

    # 7. Stop benign generators
    stop_benign(benign_procs)

    # 8. Write manifest
    write_manifest(
        domain=domain, session_id=session_id, pcap_file=pcap_filename,
        family=family, variant=variant, generator=generator, config=config,
        start_utc=start_utc, end_utc=end_utc,
        capture_start_utc=capture_start_utc, capture_end_utc=capture_end_utc,
        concurrent_benign=benign_profile, experiment_id=experiment_id,
        netem_profile=netem_profile,
    )

    # 9. Verify PCAP was written
    pcap_full = PCAP_DIR / pcap_filename
    if pcap_full.exists() and pcap_full.stat().st_size > 1000:
        print(f"  [verify] PCAP OK: {pcap_full.stat().st_size / 1024:.0f} KB")
    else:
        print(f"  [WARNING] PCAP missing or very small: {pcap_full}", file=sys.stderr)

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

    parser = argparse.ArgumentParser(description="V2 Lab Session Orchestrator")
    parser.add_argument("--domain", required=True, choices=["D01","D02","D03","D04","D05"])
    parser.add_argument("--pilot", action="store_true",
                        help="Run pilot plan (2 attack sessions only) instead of full domain plan")
    parser.add_argument("--experiment-id", default=None,
                        help="Override experiment ID (default: auto-generated)")
    parser.add_argument("--only-families", default=None,
                        help="Comma-separated attack families to run (e.g. "
                             "'ICMP_FLOOD,UDP_DDOS'). Runs ONLY those sessions from "
                             "the domain plan — used to cheaply REGENERATE specific "
                             "families without re-running the whole domain (esp. the "
                             "slow DOS_HULK). Benign-only sessions are skipped. The "
                             "resulting sessions get a fresh experiment_id; extract "
                             "them with extract_and_label.py --out <patch>.parquet and "
                             "splice them in with merge_regenerated.py.")
    args = parser.parse_args()

    experiment_id = args.experiment_id or f"v2-{args.domain}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    plan = PILOT_PLAN if args.pilot else DOMAIN_PLANS[args.domain]
    benign_profile = BENIGN_PROFILES["PILOT" if args.pilot else args.domain]

    if args.only_families:
        wanted = {f.strip() for f in args.only_families.split(",") if f.strip()}
        unknown = wanted - {fam for fam, *_ in plan if fam}
        if unknown:
            print(f"[ERROR] --only-families names not in the {args.domain} plan: {sorted(unknown)}")
            print(f"        Available: {sorted({fam for fam, *_ in plan if fam})}")
            sys.exit(1)
        plan = [row for row in plan if row[0] in wanted]
        print(f"[only-families] Restricted to {sorted(wanted)}: {len(plan)} session(s)")

    print(f"V2 Lab Orchestrator")
    print(f"Experiment : {experiment_id}")
    print(f"Domain     : {args.domain}{'  [PILOT]' if args.pilot else ''}")
    print(f"Sessions   : {len(plan)}")
    print(f"Benign     : {benign_profile}")
    print(f"PCAP dir   : {PCAP_DIR}")
    print(f"Manifest   : {MANIFEST_DIR}")
    print()

    # Pre-flight checks
    PCAP_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    # NOTE (2026-08-12): TARGET was missing from this list. ids_v2_target had
    # crashed (Apache2 SSL configtest failure -> set -e in start.sh -> whole
    # container exit) hours before the D01 pilot ran, and because this
    # pre-flight loop never checked it, every session ran with zero container
    # actually listening at 172.20.0.30 — silently, for the entire run. Every
    # attack session's flow data came back with 0 flows involving the target
    # IP at all. Don't let that recur: TARGET is now checked like every other
    # container.
    for container in [ATTACKER, TARGET, CLIENT, MONITOR]:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}}", container],
            capture_output=True, text=True
        )
        status = result.stdout.strip()
        if status != "running":
            print(f"[ERROR] Container {container} not running (status: {status!r}).")
            print("        Run: docker compose -f lab/docker-compose.yml up -d --build")
            sys.exit(1)
        print(f"[pre-flight] {container}: {status}")

    # Kill any tcpdump/tcpreplay orphaned inside the monitor container by a
    # previous run that didn't shut down cleanly (crash, Ctrl+C, machine
    # sleep/power loss). Necessary now that we know stop_capture()/
    # stop_benign() previously failed to reach these processes at all on
    # Windows — a leaked capture from a prior run would otherwise keep
    # writing into monitor's PCAP dir and/or keep replaying benign traffic
    # underneath a brand new session, corrupting it the same way last night's
    # run was corrupted.
    leaked = subprocess.run(
        ["docker", "exec", MONITOR, "sh", "-c", "pgrep -f 'tcpdump|tcpreplay' || true"],
        capture_output=True, text=True
    ).stdout.strip()
    if leaked:
        print(f"[pre-flight] Killing leaked capture process(es) in {MONITOR}: {leaked.splitlines()}")
        subprocess.run(["docker", "exec", MONITOR, "pkill", "-KILL", "-f", "tcpdump|tcpreplay"],
                        capture_output=True)
        time.sleep(1)
    else:
        print(f"[pre-flight] No leaked tcpdump/tcpreplay in {MONITOR}")

    bridge_if = find_bridge_interface()
    print(f"[pre-flight] Bridge interface: {bridge_if}")

    print("\n[pre-flight] Network emulation (netem) will be applied PER-SESSION "
          f"(varied across {len(NETEM_PROFILES)} realistic profiles, seed={NETEM_SEED}) "
          "-- see NETEM_PROFILES comment. Applying the first profile now as a baseline "
          "so pre-flight checks below run under realistic conditions too.")
    setup_netem(pick_netem_profile(0, NETEM_SEED))

    # ── Inter-session gap ──────────────────────────────────────────────────────
    # 45s between sessions lets the OS fully close lingering TCP connections
    # from the previous session. Without this, CICFlowMeter may merge flows
    # across sessions or record artificially long flow_duration values.
    INTERSESSION_GAP_S = 45

    # Run all sessions
    for i, (family, variant, generator, config) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] " + (f"{family}/{variant}" if family else f"BENIGN/{variant}"))
        netem_profile = pick_netem_profile(i, NETEM_SEED)
        print(f"  [netem] session profile: {netem_profile['name']} "
              f"(delay={netem_profile['delay']} jitter={netem_profile['jitter']} loss={netem_profile['loss']})")
        setup_netem(netem_profile)
        run_session(
            domain=args.domain,
            family=family, variant=variant, generator=generator, config=config,
            bridge_if=bridge_if, experiment_id=experiment_id,
            netem_profile=netem_profile,
        )
        if i < len(plan):
            print(f"  [gap] Inter-session cool-down: {INTERSESSION_GAP_S}s …")
            time.sleep(INTERSESSION_GAP_S)

    print(f"\n{'='*60}")
    print(f"DOMAIN {args.domain} COMPLETE")
    print(f"Experiment: {experiment_id}")
    print(f"Manifest:   {MANIFEST_DIR}/{args.domain}_manifest.jsonl")
    print(f"PCAPs:      {PCAP_DIR}/")
    print(f"{'='*60}")
    print("\nNext step: python scripts/extract_and_label.py --domain", args.domain)

if __name__ == "__main__":
    main()
