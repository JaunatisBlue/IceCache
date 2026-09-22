#!/usr/bin/env python3
"""Parse run_15/run_16 early-stop experiment logs.

Extracts DCI_PROFILE json + F1 from result.json for each config and prints
a compact comparison table + saves a markdown summary.
"""
import json
import glob
import os
import re
import sys

LOGS_DIR = "/home/yx/IceCache/experiment/logs/dci_opt"
PRED_DIR = "/home/yx/IceCache/IceCache/benchmark/pred/llama-3.1"

CONFIGS = [
    # (run, prop, name)
    ("run15", "1.0", "earlystop_1p0_qasper20"),
    ("run15", "0.5", "earlystop_0p5_qasper20"),
    ("run16", "1.0", "earlystop_trunc_1p0_qasper20"),
    ("run16", "0.5", "earlystop_trunc_0p5_qasper20"),
    ("run16", "0.25", "earlystop_trunc_0p25_qasper20"),
    ("run16", "0.125", "earlystop_trunc_0p125_qasper20"),
]

KEYS = [
    "decode_tpot_ms",
    "native_query_ms_per_token",
    "dci_select_ms_per_token",
    "recall_pages_per_token",
    "recall_wait_ms_per_token",
    "recall_gather_ms_per_token",
    "decode_steps_measured",
    "dci_select_calls",
    "index_update_calls",
]


def load_profile(name):
    log = os.path.join(LOGS_DIR, f"{name}.log")
    if not os.path.exists(log):
        return None
    txt = open(log).read()
    m = re.search(r"DCI_PROFILE (\{.*?\})\n", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def load_f1(name):
    rj = os.path.join(PRED_DIR, name, "result.json")
    if not os.path.exists(rj):
        return None
    try:
        return json.load(open(rj)).get("qasper")
    except Exception:
        return None


def main():
    rows = []
    for run, prop, name in CONFIGS:
        prof = load_profile(name)
        f1 = load_f1(name)
        if prof is None:
            print(f"[{name}] NO PROFILE (log missing/empty)")
            continue
        row = {"run": run, "prop": prop, "name": name, "f1": f1}
        for k in KEYS:
            row[k] = prof.get(k)
        rows.append(row)

    if not rows:
        print("no rows")
        return

    # header
    hdr = f"{'cfg':<38} {'F1':>6} {'TPOT':>8} {'nq/t':>7} {'sel/t':>8} {'pages':>7} {'wait':>8} {'gather':>8} {'steps':>6}"
    print(hdr)
    print("-" * len(hdr))
    baseline = None
    for r in rows:
        if r["prop"] == "1.0" and baseline is None:
            baseline = r
        f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "  -  "
        tpot = f"{r['decode_tpot_ms']:.2f}" if r["decode_tpot_ms"] is not None else "  -  "
        nq = f"{r['native_query_ms_per_token']:.2f}" if r["native_query_ms_per_token"] else "  -  "
        sel = f"{r['dci_select_ms_per_token']:.2f}" if r["dci_select_ms_per_token"] else "  -  "
        pg = f"{r['recall_pages_per_token']:.0f}" if r["recall_pages_per_token"] else "  -  "
        wt = f"{r['recall_wait_ms_per_token']:.2f}" if r["recall_wait_ms_per_token"] else "  -  "
        gt = f"{r['recall_gather_ms_per_token']:.2f}" if r["recall_gather_ms_per_token"] else "  -  "
        st = f"{r['decode_steps_measured']}" if r["decode_steps_measured"] else "  -  "
        print(f"{r['name']:<38} {f1:>6} {tpot:>8} {nq:>7} {sel:>8} {pg:>7} {wt:>8} {gt:>8} {st:>6}")

    # write markdown
    md = ["## Early-stop experiments (run15/run16)", ""]
    md.append("| config | F1 | TPOT(ms) | native_query(ms/t) | dci_select(ms/t) | pages/t | wait(ms/t) | gather(ms/t) | steps |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "-"
        tpot = f"{r['decode_tpot_ms']:.2f}" if r["decode_tpot_ms"] is not None else "-"
        nq = f"{r['native_query_ms_per_token']:.2f}" if r["native_query_ms_per_token"] else "-"
        sel = f"{r['dci_select_ms_per_token']:.2f}" if r["dci_select_ms_per_token"] else "-"
        pg = f"{r['recall_pages_per_token']:.0f}" if r["recall_pages_per_token"] else "-"
        wt = f"{r['recall_wait_ms_per_token']:.2f}" if r["recall_wait_ms_per_token"] else "-"
        gt = f"{r['recall_gather_ms_per_token']:.2f}" if r["recall_gather_ms_per_token"] else "-"
        st = f"{r['decode_steps_measured']}" if r["decode_steps_measured"] else "-"
        md.append(f"| {r['name']} | {f1} | {tpot} | {nq} | {sel} | {pg} | {wt} | {gt} | {st} |")
    md.append("")
    if baseline:
        md.append(f"Baseline (prop=1.0): F1={baseline['f1']}, TPOT={baseline['decode_tpot_ms']:.2f}ms")

    out = "/home/yx/IceCache/docs/earlystop_run15_16_summary.md"
    open(out, "w").write("\n".join(md))
    print(f"\nsummary -> {out}")


if __name__ == "__main__":
    main()