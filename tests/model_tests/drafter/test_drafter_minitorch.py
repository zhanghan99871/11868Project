"""The MiniTorch drafter (project/minitorch_drafter.py) against the PyTorch drafter it ports.

CPU tests use the numpy backend; the CUDA tests (skipped without pycuda/GPU) run the MiniTorch
kernels. The verifier side uses specdec/mla_model.py, as in decoding.
"""
import dataclasses
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.numpy import load_file, save_file  # noqa: E402

from .helpers import DRAFTER_VARIANTS, ROOT, make_drafter, make_verifier  # noqa: E402  (sets sys.path)

import hspec_drafter as hd  # noqa: E402
import minitorch_drafter as mtd  # noqa: E402
from export_hspec_drafter import FORMAT, export_drafter  # noqa: E402
from export_mla_verifier import export_state as export_verifier  # noqa: E402
from specdec.mla_model import MiniTorchMLAGPT2  # noqa: E402

ATOL = RTOL = 1e-4


@pytest.fixture(scope="module")
def verifier():
    return make_verifier()


@pytest.fixture(scope="module")
def verifier_dir(verifier, tmp_path_factory):
    return export_verifier(verifier.state_dict(), dataclasses.asdict(verifier.cfg),
                           tmp_path_factory.mktemp("verifier"))


@pytest.fixture(params=sorted(DRAFTER_VARIANTS))
def drafter_and_dir(request, verifier, tmp_path):
    drafter = make_drafter(verifier, **DRAFTER_VARIANTS[request.param])
    return drafter, export_drafter(drafter, tmp_path / "drafter")


def random_context(cfg, prefix_len, seed=0):
    rng = np.random.default_rng(seed)
    latents = [(rng.standard_normal((prefix_len, cfg.kv_lora_rank)).astype(np.float32),
                rng.standard_normal((prefix_len, cfg.qk_rope_head_dim)).astype(np.float32))
               for _ in range(cfg.n_layer)]
    summary = rng.standard_normal((len(cfg.tap_layers), cfg.target_n_embd)).astype(np.float32)
    return latents, summary


def torch_draft(drafter, anchor, latents, summary):
    with torch.no_grad():
        return drafter.draft_block(anchor, [(torch.from_numpy(c), torch.from_numpy(kr)) for c, kr in latents],
                                   torch.from_numpy(summary)).numpy()


# --- export / loading ---------------------------------------------------------------------------

def test_export_round_trip(drafter_and_dir):
    drafter, out = drafter_and_dir
    meta = json.loads((out / "config.json").read_text())
    assert meta["format"] == FORMAT == mtd.FORMAT
    cfg = mtd.DrafterConfig.from_dict(meta["config"])
    assert dataclasses.asdict(cfg) == dataclasses.asdict(drafter.cfg)
    weights = load_file(str(out / "model.safetensors"))
    state = drafter.state_dict()
    assert set(weights) == set(state) and "target_head" not in weights
    assert {k: tuple(v.shape) for k, v in weights.items()} == mtd.expected_shapes(cfg)
    for name, value in weights.items():
        np.testing.assert_array_equal(value, state[name].numpy(), err_msg=name)


def test_loader_rejects_mismatches(drafter_and_dir, verifier, tmp_path):
    drafter, out = drafter_and_dir
    head = verifier.lm_head.weight.detach().numpy()

    def variant(name, edit_meta=None, edit_weights=None):
        dst = shutil.copytree(out, tmp_path / name)
        if edit_meta:
            meta = json.loads((dst / "config.json").read_text())
            edit_meta(meta)
            (dst / "config.json").write_text(json.dumps(meta))
        if edit_weights:
            weights = load_file(str(dst / "model.safetensors"))
            edit_weights(weights)
            save_file(weights, str(dst / "model.safetensors"))
        return dst

    cases = {
        "format": variant("f", edit_meta=lambda m: m.update(format="other")),
        "unknown": variant("u", edit_meta=lambda m: m["config"].update(extra=1)),
        "one target layer": variant("t", edit_meta=lambda m: m["config"].update(target_layers=[0])),
        "missing": variant("m", edit_weights=lambda w: w.pop("head_proj.weight")),
        "wrong shapes": variant("s", edit_weights=lambda w: w.update(mask_embedding=np.zeros(5, np.float32))),
    }
    for match, path in cases.items():
        with pytest.raises(ValueError, match=match):
            mtd.MiniTorchHSpecDrafter(str(path), head)
    with pytest.raises(ValueError, match="target head"):
        mtd.MiniTorchHSpecDrafter(str(out), head[:, :5])


def test_from_verifier_checks_compatibility(drafter_and_dir, tmp_path):
    _, out = drafter_and_dir
    from decoder_mla import DecoderLM
    cfg = dataclasses.replace(make_verifier(seed=3).cfg, n_layer=2)  # the drafter reads layer 2
    small = DecoderLM(cfg).eval()
    path = export_verifier(small.state_dict(), dataclasses.asdict(cfg), tmp_path / "small_verifier")
    with pytest.raises(ValueError, match="does not fit"):
        mtd.MiniTorchHSpecDrafter.from_verifier(str(out), MiniTorchMLAGPT2(str(path)))


# --- numerics -----------------------------------------------------------------------------------

@pytest.mark.parametrize("prefix_len", [1, 7, 30])
def test_numpy_drafter_matches_torch_draft_block(drafter_and_dir, verifier, prefix_len):
    drafter, out = drafter_and_dir
    mt = mtd.MiniTorchHSpecDrafter(str(out), verifier.lm_head.weight.detach().numpy())
    latents, summary = random_context(drafter.cfg, prefix_len)
    got = mt.draft_logits(5, latents, summary)
    assert got.shape == (drafter.cfg.block_size, drafter.cfg.n_vocab)
    np.testing.assert_allclose(got, torch_draft(drafter, 5, latents, summary), atol=ATOL, rtol=RTOL)


def test_end_to_end_with_the_minitorch_verifier(drafter_and_dir, verifier, verifier_dir):
    """MiniTorch verifier cache + hidden taps -> MiniTorch drafter equals the PyTorch pipeline
    (verifier context from hspec_drafter.target_context -> HSpecDrafter.draft_block)."""
    drafter, out = drafter_and_dir
    mt_verifier = MiniTorchMLAGPT2(str(verifier_dir))
    mt = mtd.MiniTorchHSpecDrafter.from_verifier(str(out), mt_verifier)
    ids = [int(t) for t in np.random.default_rng(1).integers(0, drafter.cfg.n_vocab, 12)]
    prompt, anchor = ids[:-1], ids[-1]

    cache = mt_verifier.new_cache()
    _, hidden = mt_verifier.forward(prompt, cache, hidden_layers=drafter.cfg.tap_layers)
    tokens, logits = mt.greedy(anchor, cache, hidden[:, -1])

    latents, ref_hidden, _ = hd.target_context(verifier, torch.tensor([prompt]), drafter.cfg)
    with torch.no_grad():
        ref = drafter.draft_block(anchor, [(c[0], kr[0]) for c, kr in latents], ref_hidden[0, -1]).numpy()
    np.testing.assert_allclose(logits, ref, atol=ATOL, rtol=RTOL)
    assert tokens == ref.argmax(-1).tolist()
    assert cache.length == len(prompt)  # drafting never touches the verifier's cache


def test_input_checks(drafter_and_dir, verifier):
    drafter, out = drafter_and_dir
    mt = mtd.MiniTorchHSpecDrafter(str(out), verifier.lm_head.weight.detach().numpy())
    latents, summary = random_context(drafter.cfg, 6)
    with pytest.raises(ValueError, match="latents for"):
        mt.draft_logits(1, latents[:1], summary)
    with pytest.raises(ValueError, match="same prefix"):
        mt.draft_logits(1, [latents[0], (latents[1][0][:3], latents[1][1][:3])], summary)
    with pytest.raises(ValueError, match="summary"):
        mt.draft_logits(1, latents, summary[:1])
    long, s = random_context(drafter.cfg, drafter.cfg.n_positions - drafter.cfg.block_size)
    with pytest.raises(ValueError, match="n_positions"):
        mt.draft_logits(1, long, s)


def test_minitorch_drafter_does_not_import_torch(drafter_and_dir, verifier_dir):
    _, out = drafter_and_dir
    code = (
        "import sys, numpy as np; sys.path.insert(0, 'project')\n"
        "from specdec.mla_model import MiniTorchMLAGPT2\n"
        "from minitorch_drafter import MiniTorchHSpecDrafter\n"
        f"v = MiniTorchMLAGPT2({str(verifier_dir)!r}); d = MiniTorchHSpecDrafter.from_verifier({str(out)!r}, v)\n"
        "c = v.new_cache(); _, h = v.forward([1, 2, 3], c, hidden_layers=d.cfg.tap_layers)\n"
        "d.greedy(4, c, h[:, -1])\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n")
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


# --- CUDA ---------------------------------------------------------------------------------------

def test_cuda_drafter_matches_numpy(drafter_and_dir, verifier_dir, cuda_backend):
    drafter, out = drafter_and_dir
    gpu_verifier = MiniTorchMLAGPT2(str(verifier_dir), backend=cuda_backend)
    gpu = mtd.MiniTorchHSpecDrafter.from_verifier(str(out), gpu_verifier, backend=cuda_backend)
    cpu = mtd.MiniTorchHSpecDrafter.from_verifier(str(out), MiniTorchMLAGPT2(str(verifier_dir)))
    assert gpu.resident and gpu.ops.name == "minitorch"
    for prefix_len in (1, 9):
        latents, summary = random_context(drafter.cfg, prefix_len, seed=prefix_len)
        np.testing.assert_allclose(gpu.draft_logits(3, latents, summary), cpu.draft_logits(3, latents, summary),
                                   atol=2e-3, rtol=1e-3)
    cache = gpu_verifier.new_cache()
    _, hidden = gpu_verifier.forward([5, 6, 7, 8], cache, hidden_layers=drafter.cfg.tap_layers)
    tokens, logits = gpu.greedy(9, cache, hidden[:, -1])
    np.testing.assert_allclose(logits, torch_draft(drafter, 9, [cache.latents(m) for m in drafter.cfg.target_layers],
                                                   hidden[:, -1]), atol=2e-3, rtol=1e-3)
