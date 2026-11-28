"""plot.py: figures for eval.sh's drafter ablation (synthetic rows and histories, plus the
committed results/eval.jsonl when present)."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import plot  # noqa: E402

FIGURES = ("speedup_tau", "ablation", "acceptance", "step_time", "training", "estimate_vs_measured")


def row(config, draft, prompt, accepted_per_iter, decode_time, run_id="r1", verify_s=0.1, draft_s=0.03):
    iters = len(accepted_per_iter)
    k = plot.config_k(config)[0] if config != "ar_target" else 0
    return {"config": config, "target": "mla-gpt2-medium", "draft": draft, "prompt": prompt,
            "temperature": 0.0, "n_new": 20, "tokens": [1, 2, 3], "total_time": decode_time + 0.1,
            "prefill_time": 0.1, "decode_time": decode_time, "target_calls": iters, "draft_calls": iters,
            "target_time": verify_s * iters, "draft_time": draft_s * iters, "iterations": iters,
            "drafted": k * iters, "accepted": sum(accepted_per_iter), "accepted_per_iter": accepted_per_iter,
            "run_id": run_id}


def synthetic_rows():
    rows = [row("ar_target", "x", p, [0] * 20, 2.0) for p in range(2)]
    for label, acc, t in (("distilgpt2-pretrained", [1, 0, 2, 1], 1.8), ("distil-wikitext", [2, 1, 3, 0], 1.5),
                          ("distil-gen", [3, 2, 1, 2], 1.4), ("hspec-wikitext", [1, 1, 0, 2], 1.2),
                          ("hspec-gen", [3, 3, 2, 1], 1.0)):
        configs = ("spec_k2", "spec_k3", "spec_k5") if label.startswith("distil") else ("hspec_k3", "hspec_k5", "hspec")
        for i, cfg in enumerate(configs):
            k = plot.config_k(cfg)[0]
            rows += [row(cfg, label, p, [min(a, k) for a in acc], t + 0.1 * i) for p in range(2)]
    return rows


def write_run(path, tau, top1, block_size=None):
    path.mkdir(parents=True)
    val = [{"step": s, "loss": 1.0 / s, "tau_greedy": tau(s), "top1_agree": top1(s)} for s in (50, 100, 150)]
    train = [{"step": s, "epoch": 1 + (s - 1) // 50, "loss": 1.0} for s in range(1, 151)]
    (path / "loss_history.json").write_text(json.dumps({"train": train, "val": val}))
    (path / "arguments.json").write_text(json.dumps({"block_size": block_size} if block_size else {}))


def test_helpers():
    assert plot.config_k("spec_k3") == (3, 0.0) and plot.config_k("hspec_k5_t1.0") == (5, 1.0)
    assert plot.config_k("hspec", block_size=7) == (7, 0.0)
    with pytest.raises(ValueError):
        plot.config_k("ar_target")
    assert plot.describe("distil-gen") == ("distilgpt2", "verifier-generated")
    assert plot.describe("hspec-wikitext-2") == ("H-Spec", "WikiText-2")  # parsed from the label
    assert plot.display("hspec-gen") == "H-Spec · verifier-generated"
    rs = [{"accepted_per_iter": [0, 1, 3, 3]}]
    assert plot.survival(rs, 3) == [0.75, 0.5, 0.5]
    assert plot.val_tau({"tau_greedy": [1.5, 2.0, 2.4]}, 2) == 2.0
    assert plot.val_tau({"tau_greedy": [1.5, 2.0, 2.4]}) == 2.4 and plot.val_tau({"tau_greedy": 3.1}) == 3.1
    hist = {"train": [{"step": 200, "epoch": 2}], "val": [{"step": 100}, {"step": 200}]}
    assert plot.epochs_of(hist) == [1.0, 2.0]
    order = sorted(["hspec-gen", "distil-gen", "distilgpt2-pretrained", "hspec-wikitext", "distil-wikitext"],
                   key=plot.order_key)
    assert order == ["distilgpt2-pretrained", "distil-wikitext", "distil-gen", "hspec-wikitext", "hspec-gen"]
    with pytest.raises(SystemExit):
        plot.arguments(["--run", "no-equals-sign"])


def test_figures_and_table(tmp_path):
    results = tmp_path / "eval.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in synthetic_rows()))
    write_run(tmp_path / "dw", tau=lambda s: [1.5, 1.8, 2.0, 2.1, 2.2, 2.25, 2.3], top1=lambda s: 0.6)
    write_run(tmp_path / "hg", tau=lambda s: 2.0 + s / 100, top1=lambda s: [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
              block_size=7)
    written = plot.main(["--results", str(results), "--out", str(tmp_path / "figs"),
                         "--memory", str(tmp_path / "no-memory.json"),  # tested in test_memory.py
                         "--run", f"distil-wikitext={tmp_path / 'dw'}", "--run", f"hspec-gen={tmp_path / 'hg'}",
                         "--run", f"distil-gen={tmp_path / 'missing'}", "--run", f"hspec-wikitext={tmp_path / 'missing'}"])
    names = {Path(w).name for w in written}
    assert names == {f"{f}.png" for f in FIGURES} | {"ablation.md"}
    assert all(Path(w).stat().st_size > 1000 for w in written if w.endswith(".png"))

    table = (tmp_path / "figs" / "ablation.md").read_text().splitlines()
    lines = {line.split(" | ")[0] + " " + line.split(" | ")[1]: line for line in table[2:] if line.startswith("|")}
    # hspec-gen's fastest config: hspec_k3 (decode 1.0 s vs the baseline's 2.0 s), same-run baseline
    assert lines["| H-Spec verifier-generated"].split(" | ")[2:5] == ["hspec_k3", "2.00x", "yes"]
    assert lines["| H-Spec verifier-generated"].rstrip().endswith("| 3.50 (k=7) |")  # best: lowest val loss, step 150
    assert lines["| distilgpt2 WikiText-2"].rstrip().endswith("| 2.30 (k=7) |")
    assert lines["| distilgpt2 verifier-generated"].rstrip().endswith("| - |")  # no training history
    assert "Warning" not in "\n".join(table)


def test_unpaired_timing_shift_is_flagged(tmp_path, capsys):
    rows = synthetic_rows()
    for r in rows:
        del r["run_id"]
        if r["draft"].startswith("hspec"):
            r["target_time"] *= 0.7  # verifier 30% faster while the H-Spec configs ran
    results = tmp_path / "eval.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in rows))
    plot.main(["--results", str(results), "--out", str(tmp_path / "figs"), "--format", "svg"])
    assert "not comparable" in capsys.readouterr().out
    assert "Warning:" in (tmp_path / "figs" / "ablation.md").read_text()
    assert (tmp_path / "figs" / "speedup_tau.svg").is_file()


COMMITTED = sorted((ROOT / "results").glob("eval*.jsonl"))


@pytest.mark.skipif(not COMMITTED, reason="no committed eval results")
@pytest.mark.parametrize("results", COMMITTED, ids=lambda p: p.name)
def test_committed_results_plot(results, tmp_path):
    written = plot.main(["--results", str(results), "--out", str(tmp_path)])
    assert {Path(w).stem for w in written} >= set(FIGURES)


def test_step_time_shows_each_runs_baseline(tmp_path, monkeypatch):
    """With same-run baselines the step-time figure labels each drafter with its own run's
    verifier-alone time and draws no pooled ar_target bar."""
    rows = synthetic_rows()
    for r in rows:
        if r["draft"].startswith("hspec") or (r["config"] == "ar_target" and r["prompt"] == 1):
            r["run_id"] = "r2"
    texts = []
    import matplotlib.axes
    real = matplotlib.axes.Axes.text
    monkeypatch.setattr(matplotlib.axes.Axes, "text", lambda self, *a, **k: texts.append(a[2]) or real(self, *a, **k))
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    plt = plot.setup_matplotlib()
    rows_by, _ = plot.summarize.load([str(tmp_path / "eval.jsonl")])
    series = plot.collect(rows_by, 0.0, 7)
    plot.plot_step_time(plt, series, rows_by["ar_target"], None, str(tmp_path), "png")
    labels = [t for t in texts if "ms per step" in t]
    assert len(labels) == 5 and all("verifier alone, same run: 100)" in t for t in labels)
