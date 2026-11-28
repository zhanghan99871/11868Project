"""Memory comparison: specdec.memory's accounting (checked against the PyTorch modules), the H-Spec
drafter reading the verifier's LM head in place, bench's `memory` rows, and their summary/figure."""
import dataclasses
import io
import json
from contextlib import redirect_stdout

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from .helpers import TRAINED_VERIFIER, drafter_config, make_drafter, make_verifier  # noqa: E402  (sets sys.path)

from decoder_mla import DecoderLM, MLAConfig  # noqa: E402
from export_hspec_drafter import FORMAT as DRAFTER_FORMAT, export_drafter  # noqa: E402
from export_mla_verifier import export_state as export_verifier  # noqa: E402
from hspec_drafter import DrafterConfig, HSpecDrafter  # noqa: E402
from minitorch_drafter import MiniTorchHSpecDrafter  # noqa: E402
from specdec import bench, memory, summarize  # noqa: E402
from specdec.mla_model import FORMAT as VERIFIER_FORMAT, MiniTorchMLAGPT2  # noqa: E402

DISTILGPT2 = dict(n_layer=6, n_embd=768, n_head=12)  # distilbert/distilgpt2 (vocab 50257, 1024 positions)


def n_torch(module):
    return sum(p.numel() for p in module.parameters())


def trained_configs():
    vcfg = MLAConfig(**TRAINED_VERIFIER)
    return vcfg, DrafterConfig.for_verifier(vcfg)


# --- accounting ------------------------------------------------------------------------------------

@pytest.mark.parametrize("which", ["tiny", "trained"])
def test_counts_match_the_pytorch_modules(which):
    if which == "tiny":
        verifier = make_verifier()
        vcfg, dcfg, gcfg = verifier.cfg, drafter_config(verifier.cfg), dict(n_layer=2, n_embd=24, n_head=4,
                                                                            vocab_size=97, n_positions=64)
    else:
        (vcfg, dcfg), gcfg = trained_configs(), DISTILGPT2
    with torch.device("meta"):  # shapes only
        tv, td = DecoderLM(vcfg), HSpecDrafter(dcfg)
        tg = transformers.GPT2LMHeadModel(transformers.GPT2Config(**gcfg))
    v, h = memory.mla_verifier(dataclasses.asdict(vcfg)), memory.hspec_drafter(dataclasses.asdict(dcfg))
    g = memory.gpt2_drafter(tg.config.to_dict())
    assert (v["params"], h["params"], g["params"]) == (n_torch(tv), n_torch(td), n_torch(tg))
    assert h["shared_params"] == td.target_head.numel() and h["cache_floats_per_token"] == 0
    assert v["cache_floats_per_token"] == vcfg.n_layer * (vcfg.kv_lora_rank + vcfg.qk_rope_head_dim)
    assert g["cache_floats_per_token"] == 2 * tg.config.n_layer * tg.config.n_embd
    if which == "trained":
        assert (v["params"], h["params"], g["params"]) == (355_634_944, 95_189_952, 81_912_576)
        assert (v["cache_floats_per_token"], v["mha_cache_floats_per_token"], g["cache_floats_per_token"]) == (
            13_056, 49_152, 9_216)
        assert h["transient_floats_per_pass"] == 8 * 6 * 544  # block of 7 + anchor, 6 layers


def test_setups_and_crossover():
    v = {"params": 1000, "cache_floats_per_token": 10}
    a = {"params": 100, "cache_floats_per_token": 5}   # small weights, own cache
    b = {"params": 300, "cache_floats_per_token": 0}   # bigger weights, no cache
    assert memory.total_bytes(v, a, 10) == 4 * (1000 + 100 + 15 * 10)
    assert memory.drafter_bytes(b, 10 ** 6) == 4 * 300
    assert memory.crossover_tokens(a, b) == 40 and memory.crossover_tokens(b, a) == 40
    assert memory.crossover_tokens(a, a) is None
    assert memory.crossover_tokens({"params": 500, "cache_floats_per_token": 5}, b) is None  # a always above


def test_cli_reads_the_exports(tmp_path):
    vcfg, dcfg = trained_configs()
    for name, cfg in (("v", {"format": VERIFIER_FORMAT, "config": dataclasses.asdict(vcfg)}),
                      ("h", {"format": DRAFTER_FORMAT, "config": dataclasses.asdict(dcfg)})):
        (tmp_path / name).mkdir()
        (tmp_path / name / "config.json").write_text(json.dumps(cfg))
    transformers.GPT2Config(**DISTILGPT2).save_pretrained(tmp_path / "d")
    buf = io.StringIO()
    with redirect_stdout(buf):
        memory.main(["--verifier-dir", str(tmp_path / "v"), "--draft-dir", f"distilgpt2={tmp_path / 'd'}",
                     "--hspec-drafter", f"H-Spec={tmp_path / 'h'}", "--out", str(tmp_path / "out" / "memory.json")])
    saved = json.loads((tmp_path / "out" / "memory.json").read_text())
    assert set(saved["drafters"]) == {"distilgpt2", "H-Spec"} and saved["bytes_per_float"] == 4
    text = (tmp_path / "out" / "memory.md").read_text()
    assert text.strip() == buf.getvalue().strip()
    assert "| + distilgpt2 | 1680 | 317 | 1756 | 348 |" in text and "| + H-Spec | 1726 | 363 | 1771 | 363 |" in text
    assert "same memory at 1,441 cached tokens" in text and "3.76x its MLA latent cache" in text


# --- the loaded models ------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def models(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("memory")
    verifier = make_verifier()
    target = MiniTorchMLAGPT2(str(export_verifier(verifier.state_dict(), dataclasses.asdict(verifier.cfg), tmp / "v")))
    drafter = MiniTorchHSpecDrafter.from_verifier(str(export_drafter(make_drafter(verifier), tmp / "d")), target)
    return target, drafter, tmp


def test_hspec_drafter_reads_the_verifier_head_in_place(models):
    target, drafter, tmp = models
    assert drafter.shares_head and drafter.head is target.lm_head
    copy = MiniTorchHSpecDrafter(str(tmp / "d"), target.head_weight)  # no verifier: its own copy
    assert not copy.shares_head and copy.head is not target.lm_head
    np.testing.assert_array_equal(copy.head, target.lm_head)
    cache = target.new_cache()
    prompt = [int(t) for t in np.random.default_rng(0).integers(0, 97, 9)]
    _, hidden = target.forward(prompt[:-1], cache, hidden_layers=drafter.cfg.tap_layers)
    np.testing.assert_array_equal(drafter.draft_from_cache(prompt[-1], cache, hidden[:, -1]),
                                  copy.draft_from_cache(prompt[-1], cache, hidden[:, -1]))


def test_memory_rows(models, tmp_path):
    import minitorch
    from specdec.model import MiniTorchGPT2
    target, drafter, _ = models
    gpt2 = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=97, n_positions=64, n_embd=24,
                                                                n_layer=2, n_head=4))
    gpt2.save_pretrained(tmp_path / "g", safe_serialization=True)
    classic = MiniTorchGPT2(str(tmp_path / "g"), minitorch.SimpleBackend, max_len=64, resident=False)
    loaded = [("mla-gpt2-medium", "verifier", target), ("distil", "drafter", classic), ("hspec", "drafter", drafter)]
    rows = bench.memory_rows(loaded, {"mla-gpt2-medium": (100, 200)}, "distil", "mla-gpt2-medium", {"run_id": "r"})
    by = {r["model"]: r for r in rows}
    v, c, h = by["mla-gpt2-medium"], by["distil"], by["hspec"]
    assert (v["device_bytes"], v["host_bytes"], c["device_bytes"]) == (100, 200, None)
    assert v["params"] == memory.mla_verifier(target.cfg)["params"] and c["params"] == n_torch(gpt2)
    assert h["params"] == memory.hspec_drafter(drafter.cfg)["params"] and h["shares_head"] and h["cache_bytes_per_token"] == 0
    cfg = target.cfg
    assert v["cache_bytes_per_token"] == 4 * cfg.n_layer * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    assert v["cache_bytes_allocated"] == v["cache_bytes_per_token"] * target.max_len
    assert c["cache_bytes_per_token"] == 4 * 2 * 2 * 24  # K and V, 2 layers, 24 wide
    assert all(r["config"] == "memory" and r["draft"] == "distil" and r["run_id"] == "r" for r in rows)
    device, host = bench.memory_in_use()
    assert host > 0

    path = tmp_path / "rows.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ("memory", "mla-gpt2-medium", "distil") in bench.done_configs(str(path), 10)  # measured once
    buf = io.StringIO()
    with redirect_stdout(buf):
        summarize.main([str(path)])
    out = buf.getvalue()
    assert "| mla-gpt2-medium | verifier |" in out and "| hspec | drafter |" in out and "| yes |" in out


def test_memory_labels_and_skipping():
    assert bench.loads_classic_draft(["ar_target", "memory", "spec_k2"])
    assert not bench.loads_classic_draft(["ar_target", "memory", "hspec"])
    assert bench.loads_classic_draft(["memory"], draft_dir="runs/x/best") and not bench.loads_classic_draft(["memory"])
    done = {("ar_target", "t"), ("hspec", "t", "h"), ("spec_k2", "t", "a")}
    # decoding done, memory not: only memory runs, without re-timing the baseline
    assert bench.todo_configs(["ar_target", "memory", "hspec"], done, "t", "a", "h") == ["memory"]
    assert bench.todo_configs(["ar_target", "memory", "spec_k2"], done | {("memory", "t", "a")}, "t", "a", "h") == []
    assert bench.todo_configs(["ar_target", "memory", "hspec"], done | {("memory", "t", "a")}, "t", "a", "h") == [
        "memory"]  # H-Spec runs label memory rows with the H-Spec label


def test_memory_figure(tmp_path):
    pytest.importorskip("matplotlib")
    import plot
    from .test_plot import synthetic_rows
    vcfg, dcfg = trained_configs()
    mem = {"bytes_per_float": 4, "verifier": memory.mla_verifier(dataclasses.asdict(vcfg)),
           "drafters": {"distilgpt2": memory.gpt2_drafter(transformers.GPT2Config(**DISTILGPT2).to_dict()),
                        "H-Spec": memory.hspec_drafter(dataclasses.asdict(dcfg))}}
    (tmp_path / "memory.json").write_text(json.dumps(mem))
    rows = synthetic_rows() + [
        {"config": "memory", "target": "mla-gpt2-medium", "draft": d, "model": m, "role": role, "kind": kind,
         "params": 1, "shared_params": 0, "shares_head": kind == "hspec", "resident": True, "device_bytes": b,
         "host_bytes": 2 * b, "cache_bytes_allocated": 0, "cache_bytes_per_token": 0}
        for d, m, role, kind, b in (("distil-gen", "mla-gpt2-medium", "verifier", "mla-verifier", 2 ** 30),
                                    ("distil-gen", "distil-gen", "drafter", "gpt2", 2 ** 28),
                                    ("hspec-gen", "hspec-gen", "drafter", "hspec", 2 ** 28 + 2 ** 26))]
    (tmp_path / "eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    written = plot.main(["--results", str(tmp_path / "eval.jsonl"), "--out", str(tmp_path / "figs"),
                         "--memory", str(tmp_path / "memory.json")])
    assert str(tmp_path / "figs" / "memory.png") in written
    assert (tmp_path / "figs" / "memory.png").stat().st_size > 1000
    # without specdec.memory output the figure is skipped, the rest still drawn
    written = plot.main(["--results", str(tmp_path / "eval.jsonl"), "--out", str(tmp_path / "figs2"),
                         "--memory", str(tmp_path / "missing.json")])
    assert not any(w.endswith("memory.png") for w in written) and len(written) == 7
