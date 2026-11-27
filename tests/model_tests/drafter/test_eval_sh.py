"""eval.sh (the drafter ablation) and the bench/summarize options it uses: drafter labels,
--draft-dir, --skip-done and the fastest-config-per-drafter table."""
import io
import json
import os
import shlex
import subprocess
from contextlib import redirect_stdout

import pytest

torch = pytest.importorskip("torch")

from .helpers import ROOT  # noqa: E402  (sets sys.path)

import finetune_distilgpt2 as fd  # noqa: E402
import generate_verifier_data as gvd  # noqa: E402
import train_drafter as td  # noqa: E402
from specdec import bench, summarize  # noqa: E402


# --- bench and summarize -------------------------------------------------------------------------

def row(config, draft, prompt, decode_time, tokens=(1, 2, 3), temperature=0.0, accepted=2, drafted=4, iterations=2):
    return {"config": config, "target": "mla-gpt2-medium", "draft": draft, "prompt": prompt,
            "temperature": temperature, "n_new": 10, "tokens": list(tokens), "total_time": decode_time + 0.1,
            "prefill_time": 0.1, "decode_time": decode_time, "target_calls": iterations, "draft_calls": iterations,
            "target_time": decode_time * 0.8, "draft_time": decode_time * 0.2, "iterations": iterations,
            "drafted": drafted, "accepted": accepted, "accepted_per_iter": [1] * iterations}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_draft_labels():
    assert bench.draft_label("distilgpt2") == "distilgpt2"
    assert bench.draft_label("gpt2", "runs/distil-drafter-gen-s32/best") == "distil-drafter-gen-s32"
    assert bench.draft_label("gpt2", "runs/distil-drafter-wikitext/best/") == "distil-drafter-wikitext"
    assert bench.draft_label("gpt2", "checkpoints/my-drafter") == "my-drafter"
    assert bench.row_key("ar_target", "t", "a") == bench.row_key("ar_target", "t", "b")  # independent of the draft
    assert bench.row_key("spec_k2", "t", "a") != bench.row_key("spec_k2", "t", "b")


def test_done_configs(tmp_path):
    path = tmp_path / "rows.jsonl"
    assert bench.done_configs(str(path), 2) == set()  # no file yet
    write_rows(path, [row("ar_target", "gpt2", p, 2.0) for p in range(2)]
               + [row("spec_k2", "distil-gen", p, 1.0) for p in range(2)]
               + [row("hspec", "hspec-gen", 0, 1.0)]  # one prompt of two: not done
               + [{"config": "micro", "model": "x", "T": 1, "median_s": 0.1}])
    done = bench.done_configs(str(path), 2)
    assert done == {("ar_target", "mla-gpt2-medium"), ("spec_k2", "mla-gpt2-medium", "distil-gen")}
    assert ("hspec", "mla-gpt2-medium", "hspec-gen") in bench.done_configs(str(path), 1)


def test_skip_done_loads_nothing_when_everything_is_there(tmp_path, monkeypatch):
    path = tmp_path / "rows.jsonl"
    write_rows(path, [row("ar_target", "gpt2", p, 2.0) for p in range(3)]
               + [row(c, "hspec-gen", p, 1.0) for c in ("hspec", "hspec_k3") for p in range(3)])

    def no_load(*args, **kwargs):
        raise AssertionError("models loaded although every config is done")

    monkeypatch.setattr(bench.common, "load", no_load)
    monkeypatch.setattr(bench.common, "load_gpt2_dir", no_load)
    before = path.read_text()
    bench.main(["--target", "mla-gpt2-medium", "--n-prompts", "3", "--out", str(path), "--skip-done",
                "--hspec-label", "hspec-gen", "--configs", "ar_target", "hspec", "hspec_k3"])
    assert path.read_text() == before
    # a different label is not done: bench gets as far as loading the target
    with pytest.raises(AssertionError, match="models loaded"):
        bench.main(["--target", "mla-gpt2-medium", "--n-prompts", "3", "--out", str(path), "--skip-done",
                    "--hspec-label", "hspec-wikitext", "--configs", "ar_target", "hspec"])


def test_bench_parser_and_dir_loading(tmp_path):
    args = bench.parser().parse_args(["--configs", "spec_k2", "--out", "x.jsonl", "--draft-dir", "runs/d/best",
                                      "--draft-label", "d", "--hspec-label", "h", "--skip-done"])
    assert (args.draft_dir, args.draft_label, args.hspec_label, args.skip_done) == ("runs/d/best", "d", "h", True)
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        bench.common.load_gpt2_dir(str(tmp_path))


def test_summary_has_the_fastest_config_per_drafter(tmp_path):
    ref = [row("ar_target", "gpt2", p, 2.0) for p in range(2)]
    rows = (ref + [row("spec_k2", "distil-gen", p, 1.0) for p in range(2)]
            + [row("spec_k3", "distil-gen", p, 0.8) for p in range(2)]
            + [row("spec_k3", "distil-wikitext", p, 1.6, tokens=(9, 9, 9)) for p in range(2)]
            + [row("hspec_k5", "hspec-gen", p, 0.5) for p in range(2)]
            + [row("hspec_k5_t1.0", "hspec-gen", p, 0.4, temperature=1.0) for p in range(2)])
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows)
    buf = io.StringIO()
    with redirect_stdout(buf):
        summarize.main([str(path)])
    table = buf.getvalue()
    assert "| spec_k3 [distil-gen] | 2 | 12.50 | 2.50x |" in table  # per-config table unchanged
    best = table.split("Fastest config per drafter:")[1].strip().splitlines()[2:]
    assert [line.split(" | ")[:4] for line in best] == [
        ["| hspec-gen", "0", "hspec_k5", "20.00"],
        ["| distil-gen", "0", "spec_k3", "12.50"],
        ["| distil-wikitext", "0", "spec_k3", "6.25"],
        ["| hspec-gen", "1", "hspec_k5_t1.0", "25.00"],
    ]
    assert best[2].rstrip().endswith("| 0/2 |") and best[1].rstrip().endswith("| 2/2 |")


# --- eval.sh ------------------------------------------------------------------------------------

def dry_run(tmp_path, **env):
    """The commands eval.sh would run (DRY_RUN=1), with every path under tmp_path, as argv lists."""
    paths = {name: str(tmp_path / name.lower()) for name in
             ("GEN", "DISTIL_WIKI", "DISTIL_GEN", "HSPEC_WIKI", "HSPEC_GEN", "VERIFIER_EXPORT")}
    full_env = {**os.environ, **paths, "RESULTS": str(tmp_path / "eval.jsonl"), "DRY_RUN": "1", **env}
    out = subprocess.run(["bash", str(ROOT / "eval.sh")], env=full_env, capture_output=True, text=True, check=True)
    assert out.stdout == ""
    commands = [shlex.split(line[2:]) for line in out.stderr.splitlines() if line.startswith("+ ")]
    return commands, paths


def by_script(commands, script):
    return [c for c in commands if script in c[:3]]


def without_temps(configs):
    return [c for c in configs if "_t" not in c]


def test_eval_sh_trains_four_matched_drafters(tmp_path):
    commands, paths = dry_run(tmp_path)
    gen = gvd.arguments(by_script(commands, "torch_train/generate_verifier_data.py")[0][2:])
    assert gen.out_dir == paths["GEN"] and gen.prompt_stride == 32 and gen.temperature == 0
    assert gen.prompt_len + gen.gen_len == 128

    distil_wiki, distil_gen = (fd.arguments(c[2:]) for c in by_script(commands, "torch_train/finetune_distilgpt2.py"))
    hspec_wiki, hspec_gen = (td.arguments(c[2:]) for c in by_script(commands, "torch_train/train_drafter.py"))
    runs = {"distil_wiki": distil_wiki, "distil_gen": distil_gen, "hspec_wiki": hspec_wiki, "hspec_gen": hspec_gen}
    assert [a.output_dir for a in runs.values()] == [paths[k] for k in ("DISTIL_WIKI", "DISTIL_GEN", "HSPEC_WIKI",
                                                                         "HSPEC_GEN")]
    # only the architecture and the data differ: same start, loss, batch, schedule and evaluation
    for a in runs.values():
        assert a.init == "distilgpt2" and a.verifier_checkpoint == "runs/mla-gpt2-medium/best.pt"
        assert (a.seq_len, a.batch_size, a.epochs, a.warmup_ratio, a.weight_decay, a.clip_grad) == (128, 16, 10, 0.01,
                                                                                                     0.01, 1.0)
        assert (a.ce_weight, a.tv_weight, a.eval_every, a.max_eval_batches, a.seed) == (0.1, 0.9, 500, 50, 42)
    for wiki, gen_run in ((distil_wiki, distil_gen), (hspec_wiki, hspec_gen)):
        assert wiki.train_data is None and wiki.data_cache == "./data"
        assert (gen_run.train_data, gen_run.val_data) == (f"{paths['GEN']}/train.pt", f"{paths['GEN']}/validation.pt")
        assert wiki.lr == gen_run.lr
    td.check_anchor_room(hspec_gen.anchors_per_seq, hspec_gen.seq_len, hspec_gen.block_size, gen.prompt_len - 1)

    # the H-Spec defaults reuse runs trained with exactly these settings (runs/*/arguments.json)
    for args, run in ((hspec_wiki, "hspec-drafter-a64-lr6e-4-e10"), (hspec_gen, "hspec-drafter-gen-s32")):
        saved = json.loads((ROOT / "runs" / run / "arguments.json").read_text())
        for key in ("init", "n_layer", "n_embd", "block_size", "n_taps", "seq_len", "batch_size", "anchors_per_seq",
                    "epochs", "lr", "weight_decay", "warmup_ratio", "clip_grad", "ce_weight", "tv_weight", "gamma",
                    "seed", "summary_to_all_positions", "no_position_embeddings"):
            assert getattr(args, key) == saved[key], (run, key)
        assert (saved.get("train_data") is None) == (args.train_data is None), run  # older runs: no key
    assert json.loads((ROOT / "runs" / "hspec-drafter-gen-s32" / "arguments.json").read_text())["train_data"] \
        == "data/verifier_gen_s32/train.pt"

    exports = by_script(commands, "torch_train/export_hspec_drafter.py")
    assert [c[2:] for c in exports] == [[f"{paths[k]}/best.pt", f"{paths[k]}/minitorch"] for k in ("HSPEC_WIKI",
                                                                                                   "HSPEC_GEN")]
    assert by_script(commands, "torch_train/export_mla_verifier.py")[0][2:] == ["runs/mla-gpt2-medium/best.pt",
                                                                                paths["VERIFIER_EXPORT"]]


def test_eval_sh_benchmarks_every_drafter_once(tmp_path):
    commands, paths = dry_run(tmp_path, TEMPS="1.0", N_PROMPTS="7")
    benches = [bench.parser().parse_args(c[3:]) for c in commands if c[:3] == ["python", "-m", "specdec.bench"]]
    assert all(b.target == "mla-gpt2-medium" and b.skip_done and b.n_prompts == 7 and b.out == str(tmp_path / "eval.jsonl")
               and (b.prompt_len, b.n_new) == (64, 64) for b in benches)
    assert benches[0].configs == ["ar_target"]
    classic = {b.draft_label: b for b in benches[1:] if b.configs[0].startswith("spec")}
    hspec = {b.hspec_label: b for b in benches[1:] if b.configs[0].startswith("hspec")}
    assert set(classic) == {"distilgpt2-pretrained", "distil-wikitext", "distil-gen"}
    assert set(hspec) == {"hspec-wikitext", "hspec-gen"}
    assert classic["distilgpt2-pretrained"].draft == "distilgpt2" and classic["distilgpt2-pretrained"].draft_dir is None
    assert classic["distil-wikitext"].draft_dir == f"{paths['DISTIL_WIKI']}/best"
    assert classic["distil-gen"].draft_dir == f"{paths['DISTIL_GEN']}/best"
    assert hspec["hspec-wikitext"].hspec_drafter == f"{paths['HSPEC_WIKI']}/minitorch"
    assert hspec["hspec-gen"].hspec_drafter == f"{paths['HSPEC_GEN']}/minitorch"
    for b in classic.values():
        assert without_temps(b.configs) == ["spec_k2", "spec_k3", "spec_k4", "spec_k5"]
        assert b.configs[1] == "spec_k2_t1.0" and len(b.configs) == 8
    for b in hspec.values():
        assert b.configs == ["hspec_k3", "hspec_k3_t1.0", "hspec_k5", "hspec_k5_t1.0", "hspec", "hspec_t1.0"]
    assert all(bench.is_config(c) for b in benches for c in b.configs)
    assert commands[-1] == ["python", "-m", "specdec.summarize", str(tmp_path / "eval.jsonl")]


def test_eval_sh_reuses_finished_runs_and_stages(tmp_path):
    for path in ("distil_wiki/best/model.safetensors", "hspec_gen/best.pt", "hspec_gen/minitorch/config.json",
                 "gen/train.pt", "gen/validation.pt", "verifier_export/config.json"):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).touch()
    commands, paths = dry_run(tmp_path)
    scripts = [c[1] for c in commands if c[0] == "python" and c[1] != "-m"]
    assert scripts == ["torch_train/finetune_distilgpt2.py", "torch_train/train_drafter.py",
                       "torch_train/export_hspec_drafter.py"]
    assert fd.arguments(by_script(commands, "torch_train/finetune_distilgpt2.py")[0][2:]).output_dir == paths["DISTIL_GEN"]
    assert td.arguments(by_script(commands, "torch_train/train_drafter.py")[0][2:]).output_dir == paths["HSPEC_WIKI"]

    commands, _ = dry_run(tmp_path, STAGES="train", REFERENCE="0")
    assert not any("specdec.bench" in c for c in commands)
    commands, _ = dry_run(tmp_path, STAGES="bench", REFERENCE="0")
    assert not by_script(commands, "torch_train/finetune_distilgpt2.py")
    labels = [c[c.index("--draft-label") + 1] for c in commands if "--draft-label" in c]
    assert labels == ["distil-wikitext", "distil-gen"]  # no pretrained reference


def test_eval_sh_bench_checks_the_drafters_exist(tmp_path):
    env = {**os.environ, "STAGES": "bench", "DISTIL_WIKI": str(tmp_path / "missing"),
           "VERIFIER_EXPORT": str(tmp_path / "v"), "HSPEC_WIKI": str(tmp_path / "h"), "HSPEC_GEN": str(tmp_path / "g")}
    for d in ("v", "h/minitorch", "g/minitorch"):
        (tmp_path / d).mkdir(parents=True)
        (tmp_path / d / "config.json").touch()
    out = subprocess.run(["bash", str(ROOT / "eval.sh")], env=env, capture_output=True, text=True)
    assert out.returncode == 1 and "missing" in out.stderr and "STAGES=train" in out.stderr
