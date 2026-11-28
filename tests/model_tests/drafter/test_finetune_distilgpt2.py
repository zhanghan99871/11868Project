"""torch_train/finetune_distilgpt2.py: the classic distilgpt2 drafter of eval.sh's ablation (CPU, tiny
models; the MiniTorch logits check needs CUDA)."""
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("matplotlib")

from .helpers import VERIFIER, make_verifier  # noqa: E402  (sets sys.path)

import finetune_distilgpt2 as fd  # noqa: E402
import generate_verifier_data as gvd  # noqa: E402

V = VERIFIER["n_vocab"]
PROMPT_LEN, GEN_LEN = 8, 12


def rand_tokens(*shape, seed=0):
    return torch.randint(0, V, shape, generator=torch.Generator().manual_seed(seed))


def tiny_gpt2(seed=0):
    torch.manual_seed(seed)
    return transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=V, n_positions=VERIFIER["n_positions"], n_embd=24, n_layer=2, n_head=4)).eval()


@pytest.fixture
def setup(tmp_path):
    """A tiny verifier checkpoint (train.py format) and a tiny GPT-2 saved with save_pretrained."""
    verifier = make_verifier()
    torch.save({"model": verifier.state_dict(), "config": asdict(verifier.cfg), "step": 1, "metadata": {}},
               tmp_path / "verifier.pt")
    tiny_gpt2().save_pretrained(tmp_path / "init", safe_serialization=True)
    return verifier


def argv(tmp_path, *extra):
    return ["--verifier-checkpoint", str(tmp_path / "verifier.pt"), "--init", str(tmp_path / "init"),
            "--output-dir", str(tmp_path / "out"), "--data-cache", str(tmp_path / "data"), "--device", "cpu",
            "--no-bf16", "--seq-len", "20", "--batch-size", "4", "--eval-k", "4", "--log-every", "1000",
            "--max-eval-batches", "2", *extra]


def write_token_cache(data_dir, n=400):
    """The same tokens for train and validation (dataloader.get_tokens format), so training must
    improve the validation loss."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        torch.save({"ids": rand_tokens(n, seed=1)}, data_dir / f"wikitext2_raw_gpt2_{split}.pt")


def write_generated(verifier, tmp_path, n_train=8, n_val=4):
    (tmp_path / "gen").mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", n_train, 1), ("validation", n_val, 2)):
        ids = gvd.generate(verifier, rand_tokens(n, PROMPT_LEN, seed=seed), GEN_LEN, device="cpu")
        torch.save({"ids": ids, "prompt_len": PROMPT_LEN, "gen_len": GEN_LEN}, tmp_path / "gen" / f"{split}.pt")
    return tmp_path / "gen" / "train.pt", tmp_path / "gen" / "validation.pt"


def cpu_autocast():
    return torch.autocast(device_type="cpu", enabled=False)


# --- loss and metrics ----------------------------------------------------------------------------

def test_distill_loss():
    g = torch.Generator().manual_seed(0)
    logits, teacher = torch.randn(2, 5, V, generator=g), torch.randn(2, 5, V, generator=g)
    targets = torch.randint(0, V, (2, 5), generator=g)
    loss, stats = fd.distill_loss(logits, teacher, targets, ce_weight=0.3, tv_weight=0.7)
    ce = torch.nn.functional.cross_entropy(logits.flatten(0, 1), targets.flatten())
    tv = 0.5 * (teacher.softmax(-1) - logits.softmax(-1)).abs().sum(-1).mean()
    torch.testing.assert_close(loss, 0.3 * ce + 0.7 * tv)
    assert stats["ce"] == pytest.approx(ce.item(), rel=1e-5) and stats["tv"] == pytest.approx(tv.item(), rel=1e-5)
    assert stats["acceptance"] == pytest.approx(1 - stats["tv"])
    _, same = fd.distill_loss(teacher, teacher, targets)
    assert same["tv"] == pytest.approx(0, abs=1e-6) and same["top1_agree"] == 1.0


def test_greedy_runs_and_tau():
    agree = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 0]], dtype=torch.bool)
    runs = fd.greedy_runs(agree, 3)
    assert runs.tolist() == [[2, 1, 0, 3, 2, 1]]  # starts 0..5: every run has 3 positions ahead
    counts = torch.bincount(runs.flatten(), minlength=4).numpy()
    assert fd.tau_from_counts(counts) == pytest.approx([1 + 5 / 6, 1 + 8 / 6, 1 + 9 / 6])
    with pytest.raises(ValueError, match="at least 9"):
        fd.greedy_runs(agree, 9)


def test_tau_is_what_greedy_speculative_decoding_accepts(setup, tmp_path):
    """On verifier-generated rows, min(run, k) at start t is exactly the number of drafts that greedy
    speculative decoding accepts in a step from the context row[:t + 1]."""
    verifier = setup
    drafter = tiny_gpt2(seed=3)
    rows = gvd.generate(verifier, rand_tokens(3, PROMPT_LEN, seed=4), GEN_LEN, device="cpu")
    args = fd.arguments(argv(tmp_path))
    start, k = PROMPT_LEN - 1, 4
    _, _, agree = fd.distill_step(drafter, verifier, rows, args, cpu_autocast, start)
    runs = fd.greedy_runs(agree, k)
    with torch.no_grad():
        for b, row in enumerate(rows.tolist()):
            for i in range(runs.size(1)):
                t = start + i
                ctx = row[:t + 1]
                for _ in range(k):  # the drafter's k greedy drafts
                    ctx.append(int(drafter(torch.tensor([ctx])).logits[0, -1].argmax()))
                drafts, truth = ctx[t + 1:], row[t + 1:t + 1 + k]  # row continues with the verifier's greedy tokens
                accepted = next((j for j in range(k) if drafts[j] != truth[j]), k)
                assert accepted == runs[b, i].item(), (b, t)
    assert 0 < runs.float().mean() < k  # neither trivially all-accepted nor all-rejected


# --- training ------------------------------------------------------------------------------------

def test_training_on_wikitext_saves_a_better_drafter(setup, tmp_path):
    verifier = setup
    write_token_cache(tmp_path / "data")
    args = fd.arguments(argv(tmp_path, "--epochs", "30", "--lr", "5e-3", "--eval-every", "40",
                             "--warmup-ratio", "0.05"))
    train_loader, val_loader, input_len = fd.build_loaders(args, torch.device("cpu"))
    assert (args.loss_start, input_len) == (0, 20)
    before = fd.evaluate(fd.load_drafter(args.init), verifier, val_loader, args, torch.device("cpu"), cpu_autocast)

    history, best = fd.main(argv(tmp_path, "--epochs", "30", "--lr", "5e-3", "--eval-every", "40",
                                 "--warmup-ratio", "0.05"))
    out = tmp_path / "out"
    assert [h["step"] for h in history["val"]] == [40, 80, 120, 150]  # 5 batches/epoch x 30; 150 = final
    # observed: loss 1.34 -> 0.68, TV 0.98 -> 0.42; the tiny verifier's argmax is nearly the same
    # token everywhere, so top-1 agreement starts high: tau_greedy(k=4) 3.04 -> 3.21
    assert best < 0.75 * before["loss"]
    final = history["val"][-1]
    assert final["tv"] < before["tv"] - 0.3
    assert final["tau_greedy"][-1] > before["tau_greedy"][-1] and len(final["tau_greedy"]) == 4
    assert all(a <= b for a, b in zip(final["tau_greedy"], final["tau_greedy"][1:]))  # more drafts, more tokens
    for name in ("arguments.json", "loss_history.json", "loss_curve.png", "best/config.json",
                 "best/model.safetensors", "best/training_state.json"):
        assert (out / name).is_file(), name
    state = json.loads((out / "best" / "training_state.json").read_text())
    assert state["validation"]["loss"] == pytest.approx(best) == min(h["loss"] for h in history["val"])
    assert json.loads((out / "arguments.json").read_text())["loss_start"] == 0

    # best/ reloads with transformers and gives the validation loss it was saved with
    reloaded = fd.load_drafter(str(out / "best"))
    again = fd.evaluate(reloaded, verifier, val_loader, args, torch.device("cpu"), cpu_autocast)
    assert again["loss"] == pytest.approx(best, rel=1e-5)


def test_generated_rows_score_only_generated_tokens(setup, tmp_path, monkeypatch):
    verifier = setup
    train_pt, val_pt = write_generated(verifier, tmp_path)
    lengths = []
    real = fd.distill_loss

    def spy(logits, teacher, targets, **kwargs):
        lengths.append((logits.size(1), teacher.size(1), targets.size(1)))
        return real(logits, teacher, targets, **kwargs)

    monkeypatch.setattr(fd, "distill_loss", spy)
    fd.main(argv(tmp_path, "--train-data", str(train_pt), "--val-data", str(val_pt), "--seq-len", "128",
                 "--epochs", "2", "--eval-every", "0"))
    # rows of 20 tokens: inputs 0..18, logit positions 7..18 predict the 12 generated tokens
    assert set(lengths) == {(GEN_LEN, GEN_LEN, GEN_LEN)}
    saved = json.loads((tmp_path / "out" / "arguments.json").read_text())
    assert saved["loss_start"] == PROMPT_LEN - 1
    assert saved["data"] == {"train_rows": 8, "val_rows": 4, "prompt_len": PROMPT_LEN, "row_len": PROMPT_LEN + GEN_LEN}


def test_argument_and_compatibility_checks(setup, tmp_path):
    verifier = setup
    train_pt, val_pt = write_generated(verifier, tmp_path)
    with pytest.raises(ValueError, match="go together"):
        fd.arguments(["--train-data", "x.pt"])
    with pytest.raises(ValueError, match="positive"):
        fd.arguments(["--eval-k", "0"])
    with pytest.raises(ValueError, match="scored positions"):  # 19 inputs - 16 = 3 < eval-k 4
        fd.main(argv(tmp_path, "--train-data", str(train_pt), "--val-data", str(val_pt), "--loss-start", "16"))
    big = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=V + 1, n_positions=64, n_embd=8,
                                                               n_layer=1, n_head=2))
    with pytest.raises(ValueError, match="vocabulary"):
        fd.check_compatible(big, verifier, 20)
    with pytest.raises(ValueError, match="positions"):
        fd.check_compatible(tiny_gpt2(), verifier, 65)


# --- MiniTorch ---------------------------------------------------------------------------------

def test_best_dir_loads_in_minitorch(setup, tmp_path):
    """specdec.model.MiniTorchGPT2 finds every weight in a save_pretrained directory (tied head)."""
    import minitorch
    from specdec.model import MiniTorchGPT2
    model = tiny_gpt2(seed=5)
    model.save_pretrained(tmp_path / "best", safe_serialization=True)
    mt = MiniTorchGPT2(str(tmp_path / "best"), minitorch.SimpleBackend, max_len=64, resident=False)
    assert (mt.n_layer, mt.n_head, mt.n_embd, mt.vocab_size) == (2, 4, 24, V)
    state = model.state_dict()
    np.testing.assert_array_equal(mt.wte, state["transformer.wte.weight"].numpy())
    np.testing.assert_array_equal(mt.layers[1]["fc_w"].to_numpy(), state["transformer.h.1.mlp.c_fc.weight"].numpy())


def test_cuda_minitorch_logits_match_transformers(cuda_backend, tmp_path):
    from specdec.model import MiniTorchGPT2
    model = tiny_gpt2(seed=6)
    model.save_pretrained(tmp_path / "best", safe_serialization=True)
    mt = MiniTorchGPT2(str(tmp_path / "best"), cuda_backend, max_len=64)
    ids = rand_tokens(9, seed=7).tolist()
    with torch.no_grad():
        ref = model(torch.tensor([ids])).logits[0].numpy()
    np.testing.assert_allclose(mt.forward(ids, mt.new_cache()), ref, atol=1e-3, rtol=1e-3)
