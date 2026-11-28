"""Figures for eval.sh's drafter ablation: bench rows (results/eval.jsonl) and the drafters'
training histories (runs/*/loss_history.json).

  python plot.py                                        # results/eval.jsonl -> results/figures/
  python plot.py --results results/x.jsonl --out results/x_figures --format pdf
  python plot.py --run hspec-gen=runs/my-hspec-run      # training run behind a drafter label

Figures (greedy decoding, --temperature to change):
  speedup_tau          decode speedup and tokens per verifier step vs draft length k
  ablation             drafter x training data: fastest config's speedup; tokens/step at --ablation-k
  acceptance           share of steps accepting the first j drafts, and P(draft j | draft j-1), at k
  step_time            verifier and drafter ms per decoding step at each drafter's fastest config
  training             validation tau_greedy (teacher-forced, k = 7) and first-draft top-1 vs epoch
  estimate_vs_measured training-time tau_greedy estimate vs tokens/step measured in decoding
  memory               weights + caches of each setup at --memory-context tokens, memory each
                       drafter adds vs cached tokens (batch x context), and the GPU memory bench's
                       `memory` config measured (from --memory, written by specdec.memory)
plus ablation.md, the numbers behind the ablation figure.

Speedups divide by the ar_target of the same bench run when the rows have one (specdec.summarize);
otherwise, if verifier calls ran at clearly different speeds in different runs, the speedup figures
carry the warning. Tokens per verifier step (tau) does not depend on timing.
"""
import argparse
import json
import math
import os
import re
import sys
import textwrap
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from specdec import summarize  # noqa: E402

# drafter label -> (architecture, training data); eval.sh's labels
LABELS = {
    "distilgpt2-pretrained": ("distilgpt2", "pretrained"),
    "distil-wikitext": ("distilgpt2", "WikiText-2"),
    "distil-gen": ("distilgpt2", "verifier-generated"),
    "hspec-wikitext": ("H-Spec", "WikiText-2"),
    "hspec-gen": ("H-Spec", "verifier-generated"),
}
RUNS = {
    "distil-wikitext": "runs/distil-drafter-wikitext",
    "distil-gen": "runs/distil-drafter-gen-s32",
    "hspec-wikitext": "runs/hspec-drafter-wikitext",
    "hspec-gen": "runs/hspec-drafter-gen-s32",
}
DATA_ORDER = ("WikiText-2", "verifier-generated")
ARCH_ORDER = ("distilgpt2", "H-Spec")

# reference palette (light): categorical slots in fixed order, a gray reference, chart inks
SLOTS = ("#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647")
ARCH_COLOR = {"distilgpt2": SLOTS[0], "H-Spec": SLOTS[1]}
REFERENCE = "#9c9c9c"
INK, INK_2, GRID, AXIS, SURFACE = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"

SPEC = re.compile(r"spec_k(\d+)(?:_t([\d.]+))?")
HSPEC = re.compile(r"hspec(?:_k(\d+))?(?:_t([\d.]+))?")


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", nargs="+", default=[os.path.join(ROOT, "results", "eval.jsonl")])
    p.add_argument("--out", default=os.path.join(ROOT, "results", "figures"))
    p.add_argument("--run", action="append", default=[], metavar="LABEL=DIR",
                   help="training run (loss_history.json) of a drafter label; default: eval.sh's runs")
    p.add_argument("--temperature", type=float, default=0.0, help="decoding temperature to plot")
    p.add_argument("--ablation-k", type=int, default=5, help="draft length for the ablation tau panel")
    p.add_argument("--acceptance-k", type=int, default=5, help="draft length of the acceptance figure")
    p.add_argument("--block-size", type=int, default=7, help="k of the H-Spec config 'hspec' (no _k)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--memory", default=os.path.join(ROOT, "results", "memory.json"),
                   help="specdec.memory output (parameters and caches of the verifier and drafters)")
    p.add_argument("--memory-context", type=int, default=1024, help="cached tokens for the memory bars")
    p.add_argument("--memory-max-tokens", type=int, default=16384,
                   help="x range of the added-memory panel (batch x context)")
    args = p.parse_args(argv)
    runs = dict(RUNS)
    for item in args.run:
        label, sep, path = item.partition("=")
        if not sep or not label or not path:
            p.error(f"--run takes LABEL=DIR, got {item!r}")
        runs[label] = path
    args.runs = runs
    return args


# --- data ----------------------------------------------------------------------------------------

def describe(label):
    """(architecture, training data) of a drafter label."""
    if label in LABELS:
        return LABELS[label]
    arch = "H-Spec" if label.startswith("hspec") else "distilgpt2" if label.startswith("distil") else label
    data = ("WikiText-2" if "wiki" in label else "verifier-generated" if label.endswith(("gen", "-gen-s32"))
            else "pretrained" if "pretrained" in label else "")
    return arch, data


def display(label):
    arch, data = describe(label)
    return f"{arch} · {data}" if data else arch


def config_k(config, block_size=7):
    """(drafts per step, temperature) of a speculative config name."""
    m = SPEC.fullmatch(config)
    if m:
        return int(m.group(1)), float(m.group(2) or 0)
    m = HSPEC.fullmatch(config)
    if m:
        return int(m.group(1)) if m.group(1) else block_size, float(m.group(2) or 0)
    raise ValueError(f"not a speculative config: {config!r}")


def tau(rs):
    """Tokens per verifier step (accepted drafts + the verifier's own token)."""
    return sum(r["accepted"] + r["iterations"] for r in rs) / sum(r["iterations"] for r in rs)


def collect(rows, temperature, block_size):
    """{label: [(k, config, rows)] sorted by k} of the speculative configs at `temperature`."""
    out = defaultdict(list)
    for rs in rows.values():
        if not summarize.is_spec(rs):
            continue
        k, temp = config_k(rs[0]["config"], block_size)
        if math.isclose(temp, temperature):
            out[rs[0].get("draft", "gpt2")].append((k, rs[0]["config"], rs))
    for series in out.values():
        series.sort(key=lambda e: e[0])
    return dict(sorted(out.items(), key=lambda kv: order_key(kv[0])))


def order_key(label):
    arch, data = describe(label)
    return (data != "pretrained", ARCH_ORDER.index(arch) if arch in ARCH_ORDER else 9,
            DATA_ORDER.index(data) if data in DATA_ORDER else 9, label)


def survival(rs, k):
    """[share of steps whose first j drafts were all accepted, j = 1..k]."""
    accepted = [a for r in rs for a in r["accepted_per_iter"]]
    return [sum(a >= j for a in accepted) / len(accepted) for j in range(1, k + 1)]


def fastest(series, ar):
    """(k, config, rows, speedup, paired) of the config with the highest speedup (tok/s if no baseline)."""
    def score(entry):
        sp, _ = summarize.speedup(entry[2], ar)
        return sp if sp is not None else summarize.tok_per_s(entry[2])
    k, cfg, rs = max(series, key=score)
    sp, paired = summarize.speedup(rs, ar)
    return k, cfg, rs, sp, paired


def load_history(run_dir):
    path = os.path.join(run_dir, "loss_history.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        history = json.load(f)
    if not history.get("val") or not history.get("train"):
        return None
    with open(os.path.join(run_dir, "arguments.json")) as f:
        history["arguments"] = json.load(f)
    return history


def val_tau(entry, k=None):
    """tau_greedy at k drafts of a validation entry: a list (k = 1..) for finetune_distilgpt2.py, a
    number at the block size for train_drafter.py (only k = block size or None)."""
    t = entry["tau_greedy"]
    if isinstance(t, list):
        return t[(k or len(t)) - 1] if (k or len(t)) <= len(t) else None
    return t


def first_top1(entry):
    t = entry["top1_agree"]
    return t[0] if isinstance(t, list) else t


def best_entry(history):
    """The validation entry of the saved checkpoint (lowest validation loss)."""
    return min(history["val"], key=lambda e: e["loss"])


def epochs_of(history):
    """Validation steps as (fractional) epochs."""
    last = history["train"][-1]
    per_epoch = last["step"] / last["epoch"] if last.get("epoch") else None
    return [e["step"] / per_epoch if per_epoch else e["step"] for e in history["val"]]


# --- drawing ------------------------------------------------------------------------------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE, "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "grid.linestyle": "-", "legend.frameon": False, "legend.fontsize": 8.5, "lines.linewidth": 1.6,
        "lines.markersize": 6.5, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
    return plt


def style(label, slot_of):
    """Line/marker style: color = architecture, filled marker = verifier-generated data, hollow =
    WikiText-2, gray x = the pretrained reference; unknown labels take the next free slot."""
    arch, data = describe(label)
    if data == "pretrained":
        return dict(color=REFERENCE, marker="x", markerfacecolor=REFERENCE, linestyle="-", zorder=2)
    color = ARCH_COLOR.get(arch) or SLOTS[slot_of(label)]
    face = SURFACE if data == "WikiText-2" else color
    return dict(color=color, marker="o", markerfacecolor=face, markeredgecolor=color, markeredgewidth=1.4,
                linestyle="-", zorder=3)


def slot_allocator():
    used = {}

    def slot_of(label):
        if label not in used:
            used[label] = 2 + len(used)  # slots 1-2 are the two architectures
        return min(used[label], len(SLOTS) - 1)
    return slot_of


def footnote(fig, text):
    if text:
        fig.text(0.0, -0.02, "\n".join(textwrap.wrap("Warning: " + text, 150)), ha="left", va="top",
                 fontsize=7.5, color=INK_2)


def save(fig, out, name, fmt):
    path = os.path.join(out, f"{name}.{fmt}")
    fig.savefig(path)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"wrote {path}")
    return path


def plot_speedup_tau(plt, series, ar, warning, out, fmt, slot_of):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    unpaired = False
    for label, entries in series.items():
        st = style(label, slot_of)
        ks = [k for k, _, _ in entries]
        sps = [summarize.speedup(rs, ar) for _, _, rs in entries]
        unpaired |= any(not paired for sp, paired in sps if sp is not None)
        if all(sp is not None for sp, _ in sps):
            ax1.plot(ks, [sp for sp, _ in sps], label=display(label), **st)
        ax2.plot(ks, [tau(rs) for _, _, rs in entries], label=display(label), **st)
    ax1.axhline(1.0, color=AXIS, linewidth=1.0, zorder=1)
    ax1.text(ax1.get_xlim()[1], 1.0, "verifier alone ", va="bottom", ha="right", fontsize=7.5, color=INK_2)
    ax1.set(title="Decode speedup vs the verifier alone", xlabel="drafts per step (k)", ylabel="speedup (x)")
    ax2.set(title="Tokens per verifier step (τ)", xlabel="drafts per step (k)", ylabel="tokens / step")
    for ax in (ax1, ax2):
        ax.set_xticks(sorted({k for entries in series.values() for k, _, _ in entries}))
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    footnote(fig, warning if unpaired else None)
    fig.tight_layout()
    return save(fig, out, "speedup_tau", fmt)


def bar_group(ax, values, labels_on_bars, ylabel, title, reference=None):
    """Grouped bars: x = training data, one bar per architecture; values[(arch, data)] or missing.
    `reference` = (value, short text): a gray line (the pretrained drafter, named in the legend) with its
    value in the right margin."""
    import numpy as np
    width, gap = 0.34, 0.02
    x = np.arange(len(DATA_ORDER))
    for i, arch in enumerate(ARCH_ORDER):
        offs = (i - 0.5) * (width + gap)
        for j, data in enumerate(DATA_ORDER):
            v = values.get((arch, data))
            if v is None:
                continue
            color = ARCH_COLOR[arch]
            face = SURFACE if data == "WikiText-2" else color
            ax.bar(x[j] + offs, v, width, color=face, edgecolor=color, linewidth=1.4,
                   label=arch if j == len(DATA_ORDER) - 1 else None, zorder=3)
            ax.text(x[j] + offs, v, labels_on_bars[(arch, data)], ha="center", va="bottom", fontsize=8, color=INK,
                    linespacing=1.1)
    if reference is not None:
        value, text = reference
        ax.axhline(value, color=REFERENCE, linewidth=1.2, zorder=2)
        ax.text(len(DATA_ORDER) - 0.42, value, text, ha="right", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, list(DATA_ORDER))
    ax.set_xlim(-0.6, len(DATA_ORDER) - 0.4)
    ax.set(ylabel=ylabel, title=title, xlabel="drafter training data")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)


def ablation_numbers(series, ar, ablation_k, histories):
    """Rows of the ablation table: one per drafter label."""
    table = []
    for label, entries in series.items():
        k, cfg, rs, sp, paired = fastest(entries, ar)
        at_k = next((rs2 for k2, _, rs2 in entries if k2 == ablation_k), None)
        hist = histories.get(label)
        est = val_tau(best_entry(hist)) if hist else None
        table.append({"label": label, "arch": describe(label)[0], "data": describe(label)[1], "config": cfg, "k": k,
                      "speedup": sp, "paired": paired, "tok_per_s": summarize.tok_per_s(rs), "tau": tau(rs),
                      "tau_at_k": tau(at_k) if at_k else None, "train_tau_estimate": est,
                      "est_k": len(best_entry(hist)["tau_greedy"]) if hist and isinstance(
                          best_entry(hist)["tau_greedy"], list) else (hist["arguments"].get("block_size") if hist else None)})
    return table


def plot_ablation(plt, table, ablation_k, warning, out, fmt):
    cell = {(t["arch"], t["data"]): t for t in table if t["arch"] in ARCH_ORDER and t["data"] in DATA_ORDER}
    ref = next((t for t in table if t["data"] == "pretrained"), None)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    sp = {key: t["speedup"] for key, t in cell.items() if t["speedup"] is not None}
    bar_group(ax1, sp, {key: f"{v:.2f}x\nk={cell[key]['k']}" for key, v in sp.items()}, "speedup (x)",
              "Fastest config: decode speedup",
              (ref["speedup"], f"{ref['speedup']:.2f}x")
              if ref and ref["speedup"] is not None else None)
    taus = {key: t["tau_at_k"] for key, t in cell.items() if t["tau_at_k"] is not None}
    bar_group(ax2, taus, {key: f"{v:.2f}" for key, v in taus.items()}, "tokens / step",
              f"Tokens per verifier step (τ) at k = {ablation_k}",
              (ref["tau_at_k"], f"{ref['tau_at_k']:.2f}")
              if ref and ref["tau_at_k"] is not None else None)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=ARCH_COLOR[a], edgecolor=ARCH_COLOR[a], label=f"{a} drafter") for a in ARCH_ORDER]
    handles.append(Patch(facecolor=SURFACE, edgecolor=INK_2, label="hollow = trained on WikiText-2"))
    if ref:
        handles.append(Line2D([], [], color=REFERENCE, linewidth=1.2, label="pretrained distilgpt2 (no fine-tuning)"))
    ax2.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5))
    footnote(fig, warning if any(not t["paired"] for t in table if t["speedup"] is not None) else None)
    fig.tight_layout()
    return save(fig, out, "ablation", fmt)


def plot_acceptance(plt, series, k, out, fmt, slot_of):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    drawn = 0
    for label, entries in series.items():
        rs = next((rs for k2, _, rs in entries if k2 == k), None)
        if rs is None:
            continue
        surv = survival(rs, k)
        cond = [surv[0]] + [surv[j] / surv[j - 1] if surv[j - 1] else float("nan") for j in range(1, k)]
        st = style(label, slot_of)
        js = list(range(1, k + 1))
        ax1.plot(js, surv, label=display(label), **st)
        ax2.plot(js, cond, label=display(label), **st)
        drawn += 1
    if not drawn:
        plt.close(fig)
        print(f"no config with k = {k}: acceptance figure skipped")
        return None
    ax1.set(title=f"Steps accepting the first j drafts (k = {k})", xlabel="draft position j",
            ylabel="share of steps", ylim=(0, 1))
    ax2.set(title="Acceptance of draft j given draft j-1 accepted", xlabel="draft position j",
            ylabel="P(j accepted | j-1 accepted)", ylim=(0, 1))
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, k + 1))
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return save(fig, out, "acceptance", fmt)


def plot_step_time(plt, series, ar, warning, out, fmt):
    """Per decoding step: verifier ms + drafter ms, at each drafter's fastest config; the verifier
    alone spends one call per token."""
    entries = []
    for label, s in series.items():
        k, cfg, rs, _, _ = fastest(s, ar)
        it = sum(r["iterations"] for r in rs)
        entries.append((f"{display(label)} ({cfg})", 1e3 * sum(r["target_time"] for r in rs) / it,
                        1e3 * sum(r["draft_time"] for r in rs) / it,
                        1e3 * sum(r["decode_time"] for r in rs) / sum(r["n_new"] for r in rs), label))
    if ar:
        entries.append(("verifier alone (ar_target)", summarize.verifier_ms_per_call(ar), 0.0,
                        1e3 * sum(r["decode_time"] for r in ar) / sum(r["n_new"] for r in ar), None))
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(entries) + 1.2))
    ys = list(range(len(entries)))[::-1]
    for y, (name, verify, draft, per_token, label) in zip(ys, entries):
        color = ARCH_COLOR.get(describe(label)[0], REFERENCE) if label else REFERENCE
        if label and describe(label)[1] == "pretrained":
            color = REFERENCE
        ax.barh(y, verify, 0.6, color=INK_2, alpha=0.35, zorder=3)
        if draft:
            ax.barh(y, draft - 1.0, 0.6, left=verify + 1.0, color=color, zorder=3)  # 1 ms gap between segments
        step = f"{verify:.0f} + {draft:.0f}" if draft else f"{verify:.0f}"
        ax.text(verify + draft + 3, y, f"{step} ms per step · {per_token:.0f} ms per token", va="center",
                fontsize=7.5, color=INK)
    ax.set_yticks(ys, [e[0] for e in entries])
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(v + d for _, v, d, _, _ in entries) * 1.55)
    ax.set(title="Time per decoding step: verifier call (gray) + drafting (color)", xlabel="ms per step")
    footnote(fig, warning)
    fig.tight_layout()
    return save(fig, out, "step_time", fmt)


def plot_training(plt, histories, out, fmt, slot_of):
    if not histories:
        print("no training histories: training figure skipped")
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for label, hist in sorted(histories.items(), key=lambda kv: order_key(kv[0])):
        st = {**style(label, slot_of), "markersize": 3.5}
        ep = epochs_of(hist)
        ax1.plot(ep, [val_tau(e) for e in hist["val"]], label=display(label), **st)
        ax2.plot(ep, [first_top1(e) for e in hist["val"]], label=display(label), **st)
    ax1.set(title="Validation τ_greedy estimate (k = 7, teacher-forced)", xlabel="epoch", ylabel="tokens / step")
    ax2.set(title="Validation top-1 agreement, first draft", xlabel="epoch", ylabel="agreement with verifier")
    ax2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    return save(fig, out, "training", fmt)


def plot_estimate_vs_measured(plt, series, histories, out, fmt, slot_of):
    """Training-time tau_greedy of the saved checkpoint vs tau measured in decoding, at the same k."""
    points = []
    for label, entries in series.items():
        hist = histories.get(label)
        if not hist:
            continue
        best = best_entry(hist)
        for k, _, rs in entries:
            est = val_tau(best, k) if isinstance(best["tau_greedy"], list) else (
                best["tau_greedy"] if k == hist["arguments"].get("block_size") else None)
            if est is not None:
                points.append((label, k, est, tau(rs)))
    if not points:
        print("no drafter with both a training history and bench rows: estimate figure skipped")
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for label in dict.fromkeys(p[0] for p in points):
        pts = [p for p in points if p[0] == label]
        st = style(label, slot_of)
        ax.plot([p[2] for p in pts], [p[3] for p in pts], label=display(label), **st)
        for _, k, est, meas in (pts[0], pts[-1]) if len(pts) > 1 else pts:
            ax.annotate(f"k={k}", (est, meas), textcoords="offset points", xytext=(5, -10), fontsize=7.5, color=INK_2)
    lo = min(min(p[2], p[3]) for p in points) * 0.95
    hi = max(max(p[2], p[3]) for p in points) * 1.05
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.0, zorder=1)
    ax.text(hi, hi, "estimate = measured ", ha="right", va="bottom", fontsize=7.5, color=INK_2, rotation=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), title="τ: training estimate vs measured in decoding",
           xlabel="validation τ_greedy of the saved checkpoint", ylabel="tokens per verifier step (bench)")
    ax.grid(axis="x")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return save(fig, out, "estimate_vs_measured", fmt)


def arch_of(name, model):
    """Architecture name of a specdec.memory entry (its label, else its kind)."""
    if name in ARCH_COLOR:
        return name
    return {"gpt2": "distilgpt2", "hspec": "H-Spec"}.get(model.get("kind"), describe(name)[0])


def plot_memory(plt, memory, measured, context, max_tokens, out, fmt):
    """Analytic memory (specdec.memory) of the verifier with each drafter; measured GPU memory of
    the loaded models when bench's `memory` rows are given."""
    from matplotlib.patches import Patch
    from specdec import memory as mem
    mb = 1 / 2**20
    bpf = memory.get("bytes_per_float", mem.BYTES_PER_FLOAT)
    v, drafters = memory["verifier"], memory["drafters"]
    measured_models = {}
    for r in measured:
        measured_models[r["model"], r["role"]] = r  # latest measurement per model
    panels = 3 if any(r["device_bytes"] is not None for r in measured_models.values()) else 2
    fig, axes = plt.subplots(1, panels, figsize=(5.2 * panels, 3.9),
                             gridspec_kw={"width_ratios": [1.25, 1, 0.8][:panels]})
    ax1, ax2 = axes[0], axes[1]

    # (a) persistent memory per setup: verifier weights + cache, drafter weights + cache
    setups = [("verifier alone", None, None)] + [(f"+ {n}", n, d) for n, d in drafters.items()]
    ys = list(range(len(setups)))[::-1]
    for y, (name, dname, d) in zip(ys, setups):
        parts = [(bpf * v["params"] * mb, INK_2, 0.35, None), (bpf * v["cache_floats_per_token"] * context * mb, INK_2, 0.6, None)]
        if d:
            color = ARCH_COLOR.get(arch_of(dname, d), SLOTS[2])
            parts += [(bpf * d["params"] * mb, color, 1.0, None),
                      (bpf * d["cache_floats_per_token"] * context * mb, color, 0.45, "////")]
        left = 0.0
        for width, color, alpha, hatch in parts:
            if width > 0:
                ax1.barh(y, width, 0.6, left=left, color=color, alpha=alpha, hatch=hatch, edgecolor=SURFACE,
                         linewidth=0.8, zorder=3)
            left += width
        ax1.text(left + 15, y, f"{left:,.0f} MB", va="center", fontsize=8, color=INK)
    ax1.set_yticks(ys, [s[0] for s in setups])
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)
    ax1.set_xlim(0, max(mem.total_bytes(v, d, context) for _, _, d in setups) * mb * 1.25)
    ax1.set(title=f"One sequence, {context:,} cached tokens", xlabel="MB (float32)")
    ax1.legend(handles=[Patch(facecolor=INK_2, alpha=0.35, label="verifier weights"),
                        Patch(facecolor=INK_2, alpha=0.6, label="verifier MLA latent cache"),
                        Patch(facecolor=INK_2, label="drafter weights (color)"),
                        Patch(facecolor=SURFACE, edgecolor=INK_2, hatch="////", label="drafter's own cache")],
               loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, fontsize=7.5)

    # (b) memory a drafter adds vs cached tokens
    xs = [0, max_tokens]
    for name, d in drafters.items():
        color = ARCH_COLOR.get(arch_of(name, d), SLOTS[2])
        ys_b = [mem.drafter_bytes(d, t) * mb for t in xs]
        ax2.plot(xs, ys_b, color=color, linewidth=1.8, label=f"{name} ({d['cache_floats_per_token'] * bpf / 1024:.0f} KB/token)")
    names = list(drafters)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            x = mem.crossover_tokens(drafters[a], drafters[b])
            if x is not None and 0 < x < max_tokens:
                y = mem.drafter_bytes(drafters[a], x) * mb
                ax2.plot([x], [y], marker="o", color=INK, markersize=5, zorder=4)
                ax2.annotate(f"equal at {x:,.0f} tokens", (x, y), textcoords="offset points", xytext=(8, -14),
                             fontsize=7.5, color=INK_2)
    ax2.set(title="Memory added by the drafter", xlabel="cached tokens (batch x context)", ylabel="MB (float32)",
            xlim=(0, max_tokens))
    ax2.set_ylim(0, ax2.get_ylim()[1] * 1.05)
    ax2.legend(loc="upper left", fontsize=7.5)

    # (c) measured GPU memory of each loaded model's resident weights
    if panels == 3:
        ax3 = axes[2]
        items = [(name, r) for (name, role), r in measured_models.items() if r["device_bytes"] is not None]
        items.sort(key=lambda it: (it[1]["role"] != "verifier", it[0]))
        for i, (name, r) in enumerate(items):
            arch = "verifier" if r["role"] == "verifier" else arch_of(name, {"kind": r["kind"]})
            color = INK_2 if arch == "verifier" else ARCH_COLOR.get(arch, SLOTS[2])
            ax3.bar(i, r["device_bytes"] * mb, 0.6, color=color, alpha=0.35 if arch == "verifier" else 1.0, zorder=3)
            ax3.text(i, r["device_bytes"] * mb, f"{r['device_bytes'] * mb:,.0f}", ha="center", va="bottom", fontsize=8)
        ax3.set_xticks(range(len(items)), ["verifier" if r["role"] == "verifier" else n for n, r in items],
                       rotation=20, ha="right")
        ax3.set(title="Measured GPU memory (MiniTorch)", ylabel="MB")
        ax3.set_ylim(0, ax3.get_ylim()[1] * 1.12)
    fig.tight_layout()
    return save(fig, out, "memory", fmt)


def write_table(table, ablation_k, warning, out):
    lines = ["| drafter | training data | fastest config | speedup | same-run baseline | tokens/step there "
             f"| tokens/step at k={ablation_k} | training τ estimate (k) |", "|---|---|---|---|---|---|---|---|"]
    for t in table:
        sp = f"{t['speedup']:.2f}x" if t["speedup"] is not None else "-"
        est = f"{t['train_tau_estimate']:.2f} (k={t['est_k']})" if t["train_tau_estimate"] is not None else "-"
        at_k = f"{t['tau_at_k']:.2f}" if t["tau_at_k"] is not None else "-"
        lines.append(f"| {t['arch']} | {t['data']} | {t['config']} | {sp} | {'yes' if t['paired'] else 'no'} "
                     f"| {t['tau']:.2f} | {at_k} | {est} |")
    if warning:
        lines += ["", f"Warning: {warning}."]
    path = os.path.join(out, "ablation.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {path}")
    return path


def main(argv=None):
    args = arguments(argv)
    rows, _ = summarize.load(args.results)
    ar = rows.get("ar_target", [])
    series = collect(rows, args.temperature, args.block_size)
    if not series:
        sys.exit(f"no speculative configs at temperature {args.temperature:g} in {args.results}")
    histories = {}
    for label, path in args.runs.items():
        hist = load_history(path if os.path.isabs(path) else os.path.join(ROOT, path))
        if hist and label in series:
            histories[label] = hist
    warning = summarize.timing_check(rows)
    if warning:
        print(f"Warning: {warning}.")
    os.makedirs(args.out, exist_ok=True)
    plt = setup_matplotlib()
    slot_of = slot_allocator()
    table = ablation_numbers(series, ar, args.ablation_k, histories)
    written = [
        plot_speedup_tau(plt, series, ar, warning, args.out, args.format, slot_of),
        plot_ablation(plt, table, args.ablation_k, warning, args.out, args.format),
        plot_acceptance(plt, series, args.acceptance_k, args.out, args.format, slot_of),
        plot_step_time(plt, series, ar, warning, args.out, args.format),
        plot_training(plt, histories, args.out, args.format, slot_of),
        plot_estimate_vs_measured(plt, series, histories, args.out, args.format, slot_of),
        write_table(table, args.ablation_k, warning, args.out),
    ]
    if os.path.isfile(args.memory):
        with open(args.memory) as f:
            memory = json.load(f)
        measured = [r for r in summarize.load(args.results)[1] if r["config"] == "memory"]
        written.append(plot_memory(plt, memory, measured, args.memory_context, args.memory_max_tokens,
                                   args.out, args.format))
    else:
        print(f"no {args.memory} (python -m specdec.memory): memory figure skipped")
    return [w for w in written if w]


if __name__ == "__main__":
    main()
