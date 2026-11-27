"""Summarize bench JSONL files into markdown tables.

Usage: python -m specdec.summarize results/*.jsonl

The first table has one line per config and drafter ("config [draft]"); the second, for files with
speculative configs, the fastest config of each drafter and temperature (eval.sh's ablation view).
"""
import json
import sys
from collections import defaultdict


def spec_stats(rs, ref):
    """Acceptance, tokens per verifier call, draft time share and greedy identity of one config's rows."""
    acc = sum(r["accepted"] for r in rs) / max(1, sum(r["drafted"] for r in rs))
    tpi = sum(r["accepted"] + r["iterations"] for r in rs) / sum(r["iterations"] for r in rs)
    share = sum(r["draft_time"] for r in rs) / sum(r["draft_time"] + r["target_time"] for r in rs)
    same = [r["tokens"] == ref[r["prompt"]]["tokens"] for r in rs if r["prompt"] in ref]
    ident = f"{sum(same)}/{len(same)}" if rs[0]["temperature"] == 0 and same else "n/a (sampling)"
    return acc, tpi, share, ident


def main(paths):
    rows, micro, targets = defaultdict(list), [], set()
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["config"] == "micro":
                micro.append(r)
                continue
            targets.add(r.get("target", "gpt2-medium"))  # rows before --target existed used gpt2-medium
            if len(targets) > 1:
                sys.exit(f"{p}: results mix targets {sorted(targets)}; speedups would use the wrong "
                         "baseline, so summarize each target separately")
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
    best = {}  # (draft, temperature) -> (tok/s, line)
    for cfg, rs in rows.items():
        tps = sum(r["n_new"] for r in rs) / sum(r["decode_time"] for r in rs)
        sp = f"{tps / ref_tps:.2f}x" if ref_tps else "-"
        if cfg.startswith(("spec", "hspec")):  # classic and H-Spec speculative decoding
            acc, tpi, share, ident = spec_stats(rs, ref)
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp} | {acc:.3f} | {tpi:.2f} | {share:.2f} | {ident} |")
            draft, temp = rs[0].get("draft", "gpt2"), rs[0]["temperature"]
            line = (f"| {draft} | {temp:g} | {rs[0]['config']} | {tps:.2f} | {sp} | {acc:.3f} | {tpi:.2f} "
                    f"| {ident} |")
            if (draft, temp) not in best or tps > best[draft, temp][0]:
                best[draft, temp] = (tps, line)
        else:
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp} | - | 1.00 | - | - |")

    if best:
        print("\nFastest config per drafter:\n")
        print("| draft | temperature | best config | decode tok/s | speedup vs target AR | acceptance "
              "| tokens/iter | identical to target greedy |\n|---|---|---|---|---|---|---|---|")
        for (_, temp), (tps, line) in sorted(best.items(), key=lambda kv: (kv[0][1], -kv[1][0])):
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
