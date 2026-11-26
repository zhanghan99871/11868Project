"""torch_train/export_mla_verifier.py and the loader in specdec/mla_model.py (CPU)."""
import dataclasses
import json
import shutil

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from safetensors.numpy import load_file, save_file  # noqa: E402

import specdec.mla_model as mla  # noqa: E402
from export_mla_verifier import FORMAT, export_checkpoint  # noqa: E402

from .helpers import export, make_torch_model, numpy_verifier, token_ids, torch_logits  # noqa: E402


def test_export_writes_config_and_weights_in_pytorch_layout(exported):
    model, out = exported
    meta = json.loads((out / "config.json").read_text())
    assert meta["format"] == FORMAT == mla.FORMAT
    assert meta["config"] == dataclasses.asdict(model.cfg)

    weights = load_file(str(out / "model.safetensors"))
    state = model.state_dict()
    assert set(weights) == set(state) - {"lm_head.weight"}  # tied to token_embeddings
    for name, value in weights.items():
        assert value.dtype == np.float32
        np.testing.assert_array_equal(value, state[name].numpy(), err_msg=name)
    shapes = mla.expected_shapes(mla.MLAConfig.from_dict(meta["config"]))
    assert {k: tuple(v.shape) for k, v in weights.items()} == shapes


def test_untied_lm_head_is_exported_and_used(tmp_path):
    model = make_torch_model()
    emb = model.token_embeddings.weight.detach()
    model.lm_head.weight = torch.nn.Parameter(emb + 0.05 * torch.randn_like(emb))
    out = export(model, tmp_path / "verifier")
    assert "lm_head.weight" in load_file(str(out / "model.safetensors"))

    ids = token_ids(12)
    verifier = numpy_verifier(out)
    np.testing.assert_allclose(verifier.forward(ids, verifier.new_cache()), torch_logits(model, ids),
                               atol=1e-4, rtol=1e-4)


def test_export_checkpoint_reads_train_py_checkpoints(tmp_path):
    model = make_torch_model()
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(model.cfg), "step": 7,
                "metadata": {"source": "gpt2", "validation": {"loss": 1.5}}}, ckpt)
    out = export_checkpoint(ckpt, tmp_path / "verifier")

    meta = json.loads((out / "config.json").read_text())
    assert meta["metadata"]["step"] == 7
    assert meta["metadata"]["training_metadata"]["validation"]["loss"] == 1.5
    ids = token_ids(10)
    verifier = numpy_verifier(out)
    np.testing.assert_allclose(verifier.forward(ids, verifier.new_cache()), torch_logits(model, ids),
                               atol=1e-4, rtol=1e-4)


def test_export_checkpoint_rejects_other_files(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="not a torch_train"):
        export_checkpoint(tmp_path / "other.pt", tmp_path / "out")


def _variant(src, dst, edit_meta=None, edit_weights=None):
    shutil.copytree(src, dst)
    meta = json.loads((dst / "config.json").read_text())
    if edit_meta:
        edit_meta(meta)
        (dst / "config.json").write_text(json.dumps(meta))
    if edit_weights:
        weights = load_file(str(dst / "model.safetensors"))
        edit_weights(weights)
        save_file(weights, str(dst / "model.safetensors"))
    return dst


def test_loader_rejects_mismatched_exports(exported, tmp_path):
    _, out = exported
    cases = {
        "format": dict(edit_meta=lambda m: m.update(format="something-else")),
        "unknown": dict(edit_meta=lambda m: m["config"].update(extra_field=1)),
        "missing": dict(edit_meta=lambda m: m["config"].pop("kv_lora_rank")),
        "missing \\['layers.1.ln_2.bias'\\]": dict(edit_weights=lambda w: w.pop("layers.1.ln_2.bias")),
        "unexpected \\['layers.9.ln_1.weight'\\]": dict(
            edit_weights=lambda w: w.update({"layers.9.ln_1.weight": np.zeros(32, np.float32)})),
        "wrong shapes": dict(edit_weights=lambda w: w.update(
            {"ln.weight": np.zeros(31, np.float32)})),
    }
    for i, (match, edits) in enumerate(cases.items()):
        bad = _variant(out, tmp_path / f"bad{i}", **edits)
        with pytest.raises(ValueError, match=match):
            numpy_verifier(bad)


def test_mla_config_rejects_odd_rope_dim(exported):
    model, _ = exported
    d = dataclasses.asdict(model.cfg)
    d["qk_rope_head_dim"] = 3
    with pytest.raises(ValueError, match="even"):
        mla.MLAConfig.from_dict(d)
