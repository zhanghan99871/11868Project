"""Summarize bench JSONL files into markdown tables.

Usage: python -m specdec.summarize results/*.jsonl

The first table has one line per config and drafter ("config [draft]"); the second, for files with
speculative configs, the fastest config of each drafter and temperature (eval.sh's ablation view).

Speedups divide by the ar_target rows of the same bench run ("run_id") when there are some, so a
config and its baseline are timed under the same conditions; rows written before bench recorded
run_id fall back to every ar_target row. A note is printed when verifier calls ran at clearly
different speeds in different runs while some speedups had no same-run baseline.
"""
import json
import sys
from collections import defaultdict


def load(paths):
    """(rows grouped by "config [draft]" (ar_target alone), the micro and memory rows) of bench
    JSONL files."""
    rows, micro, targets = defaultdict(list), [], set()
    for p in paths:
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["config"] in ("micro", "memory"):
                micro.append(r)
                continue
            targets.add(r.get("target", "gpt2-medium"))  # rows before --target existed used gpt2-medium
            if len(targets) > 1:
                sys.exit(f"{p}: results mix targets {sorted(targets)}; speedups would use the wrong "
                         "baseline, so summarize each target separately")
            # ar_target does not depend on the draft; everything else is labelled with it
            key = r["config"] if r["config"] == "ar_target" else f'{r["config"]} [{r.get("draft", "gpt2")}]'
            rows[key].append(r)
    return rows, micro


def is_spec(rs):
    return rs[0]["config"].startswith(("spec", "hspec"))


def tok_per_s(rs):
    return sum(r["n_new"] for r in rs) / sum(r["decode_time"] for r in rs)


def baseline(rs, ar_rows):
    """(ar_target rows to divide by, paired?): those of the same bench run as `rs`, else all."""
    runs = {r.get("run_id") for r in rs} - {None}
    paired = [r for r in ar_rows if r.get("run_id") in runs]
    return (paired, True) if paired else (ar_rows, False)


def speedup(rs, ar_rows):
    """(decode speedup vs target AR or None, paired?)."""
    base, paired = baseline(rs, ar_rows)
    return (tok_per_s(rs) / tok_per_s(base), paired) if base else (None, False)


def verifier_ms_per_call(rs):
    return 1e3 * sum(r["target_time"] for r in rs) / sum(r["target_calls"] for r in rs)


def timing_check(rows, tolerance=0.15):
    """Warning text if verifier calls were timed at clearly different speeds across drafters while
    some speedups lack a same-run baseline, else None. A verifier call scores k + 1 <= 8 tokens and
    costs about as much as a one-token call, so a large spread means the machine's speed changed."""
    ar = rows.get("ar_target", [])
    if not ar or all(speedup(rs, ar)[1] for rs in rows.values() if is_spec(rs)):
        return None
    per_draft = defaultdict(list)
    for rs in rows.values():
        if is_spec(rs):
            per_draft[rs[0].get("draft", "gpt2")].extend(rs)
    ms = {"ar_target": verifier_ms_per_call(ar), **{d: verifier_ms_per_call(rs) for d, rs in per_draft.items()}}
    lo, hi = min(ms, key=ms.get), max(ms, key=ms.get)
    if ms[hi] <= (1 + tolerance) * ms[lo]:
        return None
    listing = ", ".join(f"{d} {v:.0f}" for d, v in ms.items())
    return (f"verifier ms per call differ across bench runs ({listing}; {hi} is {ms[hi] / ms[lo]:.2f}x {lo}) "
            "and some speedups have no same-run ar_target baseline: speedups from different runs are not "
            "comparable (re-run them with a paired ar_target; acceptance and tokens/iter are unaffected)")


def spec_stats(rs, ref):
    """Acceptance, tokens per verifier call, draft time share and greedy identity of one config's rows."""
    acc = sum(r["accepted"] for r in rs) / max(1, sum(r["drafted"] for r in rs))
    tpi = sum(r["accepted"] + r["iterations"] for r in rs) / sum(r["iterations"] for r in rs)
    share = sum(r["draft_time"] for r in rs) / sum(r["draft_time"] + r["target_time"] for r in rs)
    same = [r["tokens"] == ref[r["prompt"]]["tokens"] for r in rs if r["prompt"] in ref]
    ident = f"{sum(same)}/{len(same)}" if rs[0]["temperature"] == 0 and same else "n/a (sampling)"
    return acc, tpi, share, ident


def memory_table(memory):
    """Markdown table of bench `memory` rows: one line per model (the latest measurement)."""
    latest = {}
    for r in memory:
        latest[r["model"], r["role"]] = r
    mb = 1 / 2**20
    lines = ["| model | role | parameters | measured GPU (MB) | measured host (MB) | cache per token "
             "| cache allocated (MB) | reads verifier head |", "|---|---|---|---|---|---|---|---|"]
    for r in latest.values():
        dev = "-" if r["device_bytes"] is None else f"{r['device_bytes'] * mb:.0f}"
        host = "-" if r["host_bytes"] is None else f"{r['host_bytes'] * mb:.0f}"
        lines.append(f"| {r['model']} | {r['role']} | {r['params'] / 1e6:.1f}M | {dev} | {host} | "
                     f"{r['cache_bytes_per_token'] / 1024:.1f} KB | {r['cache_bytes_allocated'] * mb:.0f} | "
                     f"{'yes' if r.get('shares_head') else '-'} |")
    return "\n".join(lines)


def main(paths):
    rows, aux = load(paths)
    micro = [r for r in aux if r["config"] == "micro"]
    memory = [r for r in aux if r["config"] == "memory"]

    if micro:
        print("| model | new tokens T | forward latency (ms) | vs T=1 |\n|---|---|---|---|")
        base = {}
        for r in micro:
            base.setdefault(r["model"], r["median_s"])
            print(f"| {r['model']} | {r['T']} | {r['median_s'] * 1e3:.1f} | {r['median_s'] / base[r['model']]:.2f}x |")
        print()
    if memory:
        print("Memory (bench `memory`: GPU/host bytes measured around loading each model):\n")
        print(memory_table(memory))
        print()

    ar = rows.get("ar_target", [])
    ref = {r["prompt"]: r for r in ar}
    print("| config | prompts | decode tok/s | speedup vs target AR | acceptance | tokens/iter "
          "| draft time share | identical to target greedy |\n|---|---|---|---|---|---|---|---|")
    best, unpaired = {}, 0  # (draft, temperature) -> (speedup or tok/s, line)
    for cfg, rs in rows.items():
        tps = tok_per_s(rs)
        if cfg == "ar_target":
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {'1.00x' if ar else '-'} | - | 1.00 | - | - |")
            continue
        sp, paired = speedup(rs, ar)
        unpaired += sp is not None and not paired
        sp_text = f"{sp:.2f}x" if sp is not None else "-"
        if is_spec(rs):  # classic and H-Spec speculative decoding
            acc, tpi, share, ident = spec_stats(rs, ref)
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp_text} | {acc:.3f} | {tpi:.2f} | {share:.2f} | {ident} |")
            draft, temp = rs[0].get("draft", "gpt2"), rs[0]["temperature"]
            line = (f"| {draft} | {temp:g} | {rs[0]['config']} | {tps:.2f} | {sp_text} | {acc:.3f} | {tpi:.2f} "
                    f"| {ident} |")
            score = sp if sp is not None else tps
            if (draft, temp) not in best or score > best[draft, temp][0]:
                best[draft, temp] = (score, line)
        else:
            print(f"| {cfg} | {len(rs)} | {tps:.2f} | {sp_text} | - | 1.00 | - | - |")

    if best:
        print("\nFastest config per drafter:\n")
        print("| draft | temperature | best config | decode tok/s | speedup vs target AR | acceptance "
              "| tokens/iter | identical to target greedy |\n|---|---|---|---|---|---|---|---|")
        for (_, temp), (_, line) in sorted(best.items(), key=lambda kv: (kv[0][1], -kv[1][0])):
            print(line)
    if unpaired and any(r.get("run_id") for rs in rows.values() for r in rs):
        print(f"\nNote: {unpaired} configs have no ar_target from their own bench run; their speedups use all "
              "ar_target rows.")
    warning = timing_check(rows)
    if warning:
        print(f"\nWarning: {warning}.")


if __name__ == "__main__":
    main(sys.argv[1:])
