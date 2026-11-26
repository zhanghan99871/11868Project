"""torch_train/generate_verifier_data.py and training the drafter on its rows (CPU, tiny models)."""
import json
from dataclasses import asdict

import pytest

torch = pytest.importorskip("torch")

from .helpers import VERIFIER, make_verifier  # noqa: E402  (sets sys.path)

import generate_verifier_data as gvd  # noqa: E402
import train_drafter as td  # noqa: E402
from hspec_drafter import sample_anchors  # noqa: E402

V = VERIFIER["n_vocab"]


def rand_tokens(*shape, seed=0):
    return torch.randint(0, V, shape, generator=torch.Generator().manual_seed(seed))


def greedy_uncached(verifier, prompt, n):
    """Greedy continuation by re-running the whole sequence each step (no cache involved)."""
    ids = list(prompt)
    with torch.no_grad():
        for _ in range(n):
            ids.append(int(verifier(torch.tensor([ids]))[0, -1].argmax()))
    return ids


@pytest.fixture
def ckpt(tmp_path):
    model = make_verifier()
    path = tmp_path / "verifier.pt"
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), "step": 1, "metadata": {}}, path)
    return model, path


# --- prompts and generation -------------------------------------------------------------------

def test_prompts_from_tokens():
    ids = torch.arange(100)
    p = gvd.prompts_from_tokens(ids, 10)
    assert p.shape == (10, 10) and torch.equal(p[3], torch.arange(30, 40))
    p = gvd.prompts_from_tokens(ids, 10, stride=5)
    assert p.shape == (19, 10) and torch.equal(p[1], torch.arange(5, 15))
    sub = gvd.prompts_from_tokens(ids, 10, max_prompts=4, seed=1)
    assert sub.shape == (4, 10) and torch.equal(sub, gvd.prompts_from_tokens(ids, 10, max_prompts=4, seed=1))
    assert torch.equal(sub[:, 0], sub[:, 0].sort().values) and all(s % 10 == 0 for s in sub[:, 0].tolist())
    with pytest.raises(ValueError, match="fewer than"):
        gvd.prompts_from_tokens(torch.arange(5), 10)


def test_batched_greedy_generation_equals_one_row_at_a_time():
    verifier = make_verifier()
    prompts = rand_tokens(5, 8)
    out = gvd.generate(verifier, prompts, gen_len=6, batch_size=2, device="cpu")
    assert out.shape == (5, 14) and torch.equal(out[:, :8], prompts)
    for row, prompt in zip(out, prompts):
        assert row.tolist() == greedy_uncached(verifier, prompt.tolist(), 6)


def test_rows_are_what_minitorch_decoding_produces(tmp_path):
    """The continuation is exactly what specdec's greedy decoding with the MiniTorch verifier gives,
    so the drafter trains on the text it will see."""
    from export_mla_verifier import export_state
    from specdec.decoding import autoregressive
    from specdec.mla_model import MiniTorchMLAGPT2
    verifier = make_verifier()
    mt = MiniTorchMLAGPT2(str(export_state(verifier.state_dict(), asdict(verifier.cfg), tmp_path / "v")))
    prompts = rand_tokens(3, 9, seed=2)
    out = gvd.generate(verifier, prompts, gen_len=10, device="cpu")
    for row, prompt in zip(out, prompts):
        assert row[9:].tolist() == autoregressive(mt, prompt.tolist(), 10)[0]


def test_sampled_generation_is_seeded():
    verifier = make_verifier()
    prompts = rand_tokens(4, 6, seed=1)
    # the tiny verifier is nearly deterministic at temperature 1 (logits ~27), so sample at 8
    a = gvd.generate(verifier, prompts, 12, temperature=8.0, seed=3, device="cpu")
    assert torch.equal(a, gvd.generate(verifier, prompts, 12, temperature=8.0, seed=3, device="cpu"))
    assert not torch.equal(a, gvd.generate(verifier, prompts, 12, temperature=8.0, seed=4, device="cpu"))
    assert not torch.equal(a, gvd.generate(verifier, prompts, 12, device="cpu"))  # differs from greedy
    assert a.min() >= 0 and a.max() < V and torch.equal(a[:, :6], prompts)


def test_generation_checks_length():
    with pytest.raises(ValueError, match="positions"):
        gvd.generate(make_verifier(), rand_tokens(1, 60), gen_len=10, device="cpu")  # 70 > 64 positions


# --- the script ---------------------------------------------------------------------------------

def write_token_cache(data_dir, prefix="wikitext2", n_train=100, n_val=50):
    data_dir.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", n_train, 1), ("validation", n_val, 2)):  # dataloader.get_tokens format
        torch.save({"ids": rand_tokens(n, seed=seed)}, data_dir / f"{prefix}_raw_gpt2_{split}.pt")


def gen_argv(tmp_path, ckpt_path, *extra):
    return ["--verifier-checkpoint", str(ckpt_path), "--data-cache", str(tmp_path / "data"),
            "--out-dir", str(tmp_path / "gen"), "--prompt-len", "8", "--gen-len", "6", "--batch-size", "4",
            "--device", "cpu", *extra]


def test_main_writes_split_files_and_skips_existing(ckpt, tmp_path):
    _, path = ckpt
    write_token_cache(tmp_path / "data")
    gvd.main(gen_argv(tmp_path, path, "--max-val-prompts", "3"))
    ids, prompt_len = gvd.load_generated(tmp_path / "gen" / "train.pt")
    assert ids.shape == (100 // 8, 14) and prompt_len == 8
    val, _ = gvd.load_generated(tmp_path / "gen" / "validation.pt")
    assert val.shape == (3, 14)
    meta = torch.load(tmp_path / "gen" / "train.pt", weights_only=True)
    assert (meta["gen_len"], meta["temperature"], meta["source"], meta["split"]) == (6, 0.0, "wikitext-2", "train")

    before = (tmp_path / "gen" / "train.pt").stat().st_mtime_ns
    gvd.main(gen_argv(tmp_path, path))  # outputs exist: nothing regenerated
    assert (tmp_path / "gen" / "train.pt").stat().st_mtime_ns == before


def test_main_with_wikitext103_and_sampling(ckpt, tmp_path):
    _, path = ckpt
    write_token_cache(tmp_path / "data", prefix="wikitext103", n_train=40)
    gvd.main(gen_argv(tmp_path, path, "--source", "wikitext-103", "--splits", "train", "--temperature", "8",
                      "--samples-per-prompt", "2", "--prompt-stride", "4"))
    ids, _ = gvd.load_generated(tmp_path / "gen" / "train.pt")
    assert ids.shape == (2 * 9, 14)  # windows at 0, 4, ..., 32; two samples each
    assert torch.equal(ids[0, :8], ids[1, :8])  # same prompt, independent samples


def test_generation_argument_checks(tmp_path):
    with pytest.raises(ValueError, match="temperature"):
        gvd.arguments(["--samples-per-prompt", "2"])
    with pytest.raises(ValueError, match="not written by"):
        torch.save({"rows": 1}, tmp_path / "x.pt")
        gvd.load_generated(tmp_path / "x.pt")


# --- training on generated rows -----------------------------------------------------------------

def test_sample_anchors_min_anchor():
    a = sample_anchors(4, 20, 4, 6, generator=torch.Generator().manual_seed(0), min_anchor=10)
    assert a.min() >= 10 and a.max() <= 15
    assert sample_anchors(1, 20, 4, 6, min_anchor=10).tolist() == [list(range(10, 16))]
    with pytest.raises(ValueError):
        sample_anchors(1, 20, 4, 7, min_anchor=10)


def train_argv(tmp_path, ckpt_path, *extra):
    return ["--verifier-checkpoint", str(ckpt_path), "--output-dir", str(tmp_path / "out"),
            "--train-data", str(tmp_path / "gen" / "train.pt"), "--val-data", str(tmp_path / "gen" / "validation.pt"),
            "--device", "cpu", "--no-bf16", "--init", "none", "--n-layer", "2", "--n-embd", "24", "--n-taps", "3",
            "--block-size", "4", "--seq-len", "128", "--batch-size", "2", "--anchors-per-seq", "3", "--epochs", "2",
            "--eval-every", "0", "--max-eval-batches", "2", "--log-every", "1000", *extra]


def write_generated(verifier, tmp_path, prompt_len=8, gen_len=12, n_train=6, n_val=2):
    (tmp_path / "gen").mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", n_train, 1), ("validation", n_val, 2)):
        ids = gvd.generate(verifier, rand_tokens(n, prompt_len, seed=seed), gen_len, device="cpu")
        torch.save({"ids": ids, "prompt_len": prompt_len, "gen_len": gen_len}, tmp_path / "gen" / f"{split}.pt")


def test_training_on_generated_rows_drafts_only_generated_tokens(ckpt, tmp_path, monkeypatch):
    verifier, path = ckpt
    write_generated(verifier, tmp_path)
    seen = []
    real = td.sample_anchors

    def spy(*args, **kwargs):
        anchors = real(*args, **kwargs)
        seen.append(anchors)
        return anchors

    monkeypatch.setattr(td, "sample_anchors", spy)
    td.main(train_argv(tmp_path, path))
    anchors = torch.cat([a.flatten() for a in seen])
    assert anchors.min() >= 7 and anchors.max() <= 20 - 1 - 4  # last prompt token .. end of row
    saved = json.loads((tmp_path / "out" / "arguments.json").read_text())
    assert saved["min_anchor"] == 7 and saved["seq_len"] == 20  # capped to the row length
    assert saved["data"] == {"train_rows": 6, "val_rows": 2, "prompt_len": 8, "row_len": 20}
    assert (tmp_path / "out" / "best.pt").is_file()


def test_generated_data_checks(ckpt, tmp_path):
    verifier, path = ckpt
    write_generated(verifier, tmp_path)
    with pytest.raises(ValueError, match="go together"):
        td.arguments(["--train-data", "x.pt"])
    with pytest.raises(ValueError, match="anchors-per-seq must be <= 9"):  # anchors 7..15
        td.main(train_argv(tmp_path, path, "--anchors-per-seq", "10"))
    td.main(train_argv(tmp_path, path, "--anchors-per-seq", "10", "--min-anchor", "1"))  # all positions: fits
    torch.save({"ids": rand_tokens(2, 20), "prompt_len": 5}, tmp_path / "gen" / "validation.pt")
    with pytest.raises(ValueError, match="different prompt"):
        td.main(train_argv(tmp_path, path))
