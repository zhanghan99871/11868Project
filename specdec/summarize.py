"""Summarize bench JSONL files into a markdown table.

Usage: python -m specdec.summarize results/*.jsonl
"""
import json
import sys
from collections import defaultdict

import numpy as np


def main(paths):
    rows, micro = defaultdict(list), []
    for p in paths:
        for line in open(p):
            r = json.loads(line)
            if r["config"] == "micro":
                micro.append(r)
                continue
            # ar_target does not depend on the draft; everything else is labelled with it
            key = r["config"] if r["config"] == "ar_target" else f'{r["config"]} [{r.get("draft", "gpt2")}]'
            rows[key].append(r)

    if micro:
        print("| model | new tokens T | forward latency (ms) | vs T=1 |\n|---|---|---|---|")
        base = {}
        for r in micro:
            base.setdefault(r["model"], r["median_s"])
            print(f"| {r['model']} | {r['T']} | {r['median_s'] * 1e3:.1f} | {r['median_s'] / base[r['model']]:.2f}x |")
        print()

    ref = {r["prompt"]: r for r in rows.get("ar_target", [])}
    ref_tps = None
    if ref:
        ref_tps = sum(r["n_new"] for r in ref.values()) / sum(r["decode_time"] for r in ref.values())
    print("| config | prompts | decode tok/s | speedup vs target AR | acceptance | tokens/iter "
          "| draft time share | identical to target greedy |\n|---|---|---|---|---|---|---|---|")
    for cfg, rs in rows.items():
        tps = sum(r["n_new"] for r in rs) / sum(r["decode_time"] for r in rs)
        sp = f"{tps / ref_tps:.2f}x" if ref_tps else "-"
        if cfg.startswith("spec"):
            acc = sum(r["accepted"] for r in rs) / max(1, sum(r["drafted"] for r in rs))
            tpi = sum(r["accepted"] + r["iterations"] for r in rs) / sum(r["iterations"] for r in rs)
            share = sum(r["draft_time"] for r in rs) / sum(r["draft_time"] + r["target_time"] for r in rs)
            same = [r["tokens"] == ref[r["prompt"]]["tokens"] for r in rs if r["prompt"] in ref]
            ident = (f"{sum(same)}/{len(same)}" if rs[0]["temperature"] == 0 and same else "n/a (sampling)")
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp} | {acc:.3f} | {tpi:.2f} | {share:.2f} | {ident} |")
        else:
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp} | - | 1.00 | - | - |")


if __name__ == "__main__":
    main(sys.argv[1:])
