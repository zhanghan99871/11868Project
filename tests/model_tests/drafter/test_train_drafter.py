"""torch_train/train_drafter.py and train.sh (CPU, tiny models, synthetic token batches)."""
import json
import shlex
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("matplotlib")

from .helpers import ROOT, VERIFIER, make_verifier  # noqa: E402  (sets sys.path)

import train_drafter as td  # noqa: E402
from export_hspec_drafter import export_checkpoint  # noqa: E402
from export_mla_verifier import export_state as export_verifier  # noqa: E402

SEQ, B = 20, 2
TINY = ["--device", "cpu", "--no-bf16", "--init", "none", "--n-layer", "2", "--n-embd", "24",
        "--n-taps", "3", "--block-size", "4", "--seq-len", str(SEQ), "--batch-size", str(B),
        "--anchors-per-seq", "3", "--log-every", "1000", "--max-eval-batches", "2"]


def tiny_args(tmp_path, *extra):
    return td.arguments(TINY + ["--output-dir", str(tmp_path / "out"),
                                "--verifier-checkpoint", str(tmp_path / "verifier.pt")] + list(extra))


def batches(n, seed):
    """n batches of [B, SEQ + 1] token ids, like dataloader.make_dataloader yields."""
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VERIFIER["n_vocab"], (B, SEQ + 1), generator=g) for _ in range(n)]


@pytest.fixture
def verifier_ckpt(tmp_path):
    """A tiny verifier saved the way torch_train/train.py saves checkpoints."""
    model = make_verifier()
    path = tmp_path / "verifier.pt"
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "step": 1, "metadata": {}}, path)
    return model, path


# --- train.sh and arguments --------------------------------------------------------------------

def test_train_sh_flags_parse():
    """The commands in train.sh use only flags their scripts accept, with the intended values."""
    import generate_verifier_data as gvd
    text = (ROOT / "train.sh").read_text().replace("\\\n", " ")
    commands = [shlex.split(line) for line in text.splitlines()
                if line.strip().startswith("python torch_train/")]
    train_cmds = [td.arguments([a for a in c[2:] if a != "$@"])
                  for c in commands if c[1] == "torch_train/train_drafter.py"]
    assert len(train_cmds) == 2  # DATA=generated (default) and DATA=wikitext
    for args in train_cmds:
        assert args.init == "distilgpt2" and args.verifier_checkpoint == "$VERIFIER" and args.output_dir == "$OUT"
        assert (args.seq_len, args.block_size, args.n_layer, args.n_embd) == (128, 7, 6, 768)
    gen_train, wiki_train = train_cmds
    assert (gen_train.train_data, gen_train.val_data) == ("$GEN/train.pt", "$GEN/validation.pt")
    assert wiki_train.train_data is None and (wiki_train.lr, wiki_train.anchors_per_seq) == (3e-4, 32)

    gen = gvd.arguments(next(c for c in commands if c[1] == "torch_train/generate_verifier_data.py")[2:])
    assert gen.out_dir == "$GEN" and gen.temperature == 0 and gen.prompt_len + gen.gen_len == gen_train.seq_len
    # the generated-data run's anchors fit between the last prompt token and the end of the row
    td.check_anchor_room(gen_train.anchors_per_seq, gen_train.seq_len, gen_train.block_size, gen.prompt_len - 1)

    export_cmd = next(c for c in commands if c[1] == "torch_train/export_hspec_drafter.py")
    assert export_cmd[2:] == ["$OUT/best.pt", "$OUT/minitorch"]


def test_argument_checks():
    with pytest.raises(ValueError, match="anchors-per-seq"):
        td.arguments(["--seq-len", "16", "--block-size", "7", "--anchors-per-seq", "9"])
    with pytest.raises(ValueError, match="batch-size"):
        td.arguments(["--batch-size", "0"])


def test_lr_schedule():
    total, warmup, peak = 100, 10, 1e-3
    lrs = [td.lr_at(s, total, peak, warmup) for s in range(1, total + 1)]
    assert lrs[0] == pytest.approx(peak / warmup) and lrs[warmup - 1] == pytest.approx(peak)
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))  # cosine decay
    assert lrs[-1] == pytest.approx(0, abs=1e-12)
    assert td.lr_at(1, total, peak, 0) == pytest.approx(peak, rel=1e-3)  # no warmup


def test_param_groups_skip_decay_on_vectors(verifier_ckpt, tmp_path):
    verifier, _ = verifier_ckpt
    drafter, _ = td.build_drafter(verifier, tiny_args(tmp_path))
    decay, no_decay = td.param_groups(drafter, 0.01)
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert any(p is drafter.mask_embedding for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(list(drafter.parameters()))


# --- building the drafter ----------------------------------------------------------------------

def test_load_verifier_is_frozen(verifier_ckpt):
    model, path = verifier_ckpt
    loaded = td.load_verifier(path, torch.device("cpu"))
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    ids = torch.randint(0, VERIFIER["n_vocab"], (1, 9))
    with torch.no_grad():
        torch.testing.assert_close(loaded(ids), model(ids))


def test_build_drafter_from_gpt2_weights(verifier_ckpt, tmp_path):
    transformers = pytest.importorskip("transformers")
    verifier, _ = verifier_ckpt
    torch.manual_seed(0)
    gpt2 = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=VERIFIER["n_vocab"], n_positions=VERIFIER["n_positions"], n_embd=24, n_layer=4, n_head=4))
    args = tiny_args(tmp_path, "--init", "distilgpt2")  # gpt2_state replaces the download
    drafter, info = td.build_drafter(verifier, args, gpt2_state=gpt2.state_dict())
    assert info["trunk_source_layers"] == [0, 3]
    torch.testing.assert_close(drafter.layers[1].ff.linear_in.weight, gpt2.state_dict()["transformer.h.3.mlp.c_fc.weight"].T)
    for layer, m in zip(drafter.layers, drafter.cfg.target_layers):
        torch.testing.assert_close(layer.attention.kv_up.weight, verifier.layers[m].attention.kv_up.weight)
    torch.testing.assert_close(drafter.target_head, verifier.lm_head.weight)


# --- training ------------------------------------------------------------------------------------

def test_training_run_checkpoints_exports_and_resumes(verifier_ckpt, tmp_path):
    verifier, ckpt_path = verifier_ckpt
    args = tiny_args(tmp_path, "--epochs", "40", "--lr", "5e-3", "--eval-every", "20", "--warmup-ratio", "0.1")
    torch.manual_seed(0)
    drafter, info = td.build_drafter(verifier, args)
    train_data, val_data = batches(2, seed=1), batches(1, seed=1)  # validation = seen data: must improve
    device = torch.device("cpu")

    def autocast():
        return torch.autocast(device_type="cpu", enabled=False)

    before = td.evaluate(drafter, verifier, val_data, args, device, autocast)
    history, best = td.train(drafter, verifier, train_data, val_data, args, device, args.output_dir, info=info)

    out = tmp_path / "out"
    assert [h["step"] for h in history["val"]] == [20, 40, 60, 80]  # every 20 steps; 80 = final
    assert len(history["train"]) == 80
    # observed: loss 1.32 -> 0.90 (0.68x), mean acceptance 0.02 -> 0.21
    assert best < 0.85 * before["loss"]
    assert history["val"][-1]["mean_acceptance"] > before["mean_acceptance"] + 0.1
    for name in ("best.pt", "arguments.json", "loss_history.json", "loss_curve.png"):
        assert (out / name).is_file(), name
    saved = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert saved["metadata"]["validation"]["loss"] == pytest.approx(best)
    assert "target_head" not in saved["model"]
    assert json.loads((out / "arguments.json").read_text())["anchors_per_seq"] == 3

    # resume: weights from the checkpoint (trained kv_up kept), head from the verifier
    resumed, rinfo = td.build_drafter(verifier, tiny_args(tmp_path, "--init-checkpoint", str(out / "best.pt")))
    assert rinfo["resumed_step"] == saved["step"]
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, saved["model"][name])
    m = resumed.cfg.target_layers[0]
    assert not torch.equal(resumed.layers[0].attention.kv_up.weight, verifier.layers[m].attention.kv_up.weight)
    torch.testing.assert_close(resumed.target_head, verifier.lm_head.weight)

    # the trained drafter runs in MiniTorch and matches PyTorch
    from minitorch_drafter import MiniTorchHSpecDrafter
    from specdec.mla_model import MiniTorchMLAGPT2
    mt_verifier = MiniTorchMLAGPT2(str(export_verifier(verifier.state_dict(), asdict(verifier.cfg), tmp_path / "v")))
    mt = MiniTorchHSpecDrafter.from_verifier(str(export_checkpoint(out / "best.pt", tmp_path / "d")), mt_verifier)
    prompt = train_data[0][0, :9].tolist()
    cache = mt_verifier.new_cache()
    _, hidden = mt_verifier.forward(prompt, cache, hidden_layers=mt.cfg.tap_layers)
    logits = mt.draft_from_cache(int(train_data[0][0, 9]), cache, hidden[:, -1])
    with torch.no_grad():
        ref = resumed.eval().draft_block(int(train_data[0][0, 9]),
                                         [tuple(torch.from_numpy(a) for a in cache.latents(m))
                                          for m in resumed.cfg.target_layers],
                                         torch.from_numpy(hidden[:, -1])).numpy()
    np.testing.assert_allclose(logits, ref, atol=1e-4, rtol=1e-4)


def test_grad_accumulation_and_max_steps(verifier_ckpt, tmp_path):
    verifier, _ = verifier_ckpt
    args = tiny_args(tmp_path, "--epochs", "3", "--grad-accum", "2", "--eval-every", "0")
    drafter, info = td.build_drafter(verifier, args)
    history, _ = td.train(drafter, verifier, batches(5, seed=2), batches(1, seed=3), args,
                          torch.device("cpu"), args.output_dir, info=info)
    assert len(history["train"]) == 9  # ceil(5 / 2) optimizer steps per epoch
    assert [h["step"] for h in history["val"]] == [9]  # final evaluation only

    args = tiny_args(tmp_path, "--epochs", "10", "--max-steps", "4", "--eval-every", "2")
    drafter, info = td.build_drafter(verifier, args)
    history, _ = td.train(drafter, verifier, batches(3, seed=2), batches(1, seed=3), args,
                          torch.device("cpu"), args.output_dir, info=info)
    assert len(history["train"]) == 4 and [h["step"] for h in history["val"]] == [2, 4]


def test_evaluate_uses_the_same_anchors_every_time(verifier_ckpt, tmp_path):
    verifier, _ = verifier_ckpt
    args = tiny_args(tmp_path)
    drafter, _ = td.build_drafter(verifier, args)

    def autocast():
        return torch.autocast(device_type="cpu", enabled=False)

    data = batches(3, seed=5)
    first = td.evaluate(drafter, verifier, data, args, torch.device("cpu"), autocast)
    assert first == td.evaluate(drafter, verifier, data, args, torch.device("cpu"), autocast)
    assert first["batches"] == 2 and len(first["acceptance"]) == 4  # --max-eval-batches 2, k = 4
    assert first["tau_greedy"] == pytest.approx(1 + first["greedy_accept_len"])
