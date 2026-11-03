import pytest
import torch
from pathlib import Path
import sys
# The existing converter is located in <repo>/project/gpt2_loader.py.
PROJECT_DIR = Path(__file__).resolve().parents[2] / "torch_train"
sys.path.insert(0, str(PROJECT_DIR))

from decoder_mla import MLAConfig, DecoderLM


@pytest.fixture
def model():
    torch.manual_seed(7)
    cfg = MLAConfig(n_vocab=129, n_embd=64, n_head=4, n_layer=2,
                    n_positions=32, q_lora_rank=32, kv_lora_rank=16,
                    qk_nope_head_dim=12, qk_rope_head_dim=8,
                    v_head_dim=16, p_dropout=0.0)
    return DecoderLM(cfg).eval()


def test_forward_shape(model):
    assert model(torch.randint(0, 129, (2, 6))).shape == (2, 6, 129)


def test_cache_matches_full_sequence(model):
    ids = torch.randint(0, 129, (2, 7))
    with torch.no_grad():
        full = model(ids)
        cache = None
        out = []
        for i in range(ids.size(1)):
            logits, cache = model(ids[:, i:i+1], past_key_values=cache, use_cache=True)
            out.append(logits)
        cached = torch.cat(out, dim=1)
    torch.testing.assert_close(full, cached, atol=1e-5, rtol=1e-5)
    for latent, k_rope in cache:
        assert latent.shape == (2, 7, model.cfg.kv_lora_rank)
        assert k_rope.shape == (2, 7, model.cfg.qk_rope_head_dim)


def test_causal(model):
    ids = torch.randint(0, 129, (1, 7))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 129
    with torch.no_grad():
        a = model(ids)
        b = model(changed)
    torch.testing.assert_close(a[:, :-1], b[:, :-1], atol=0, rtol=0)


def test_backward(model):
    ids = torch.randint(0, 129, (2, 5))
    loss = model(ids).square().mean()
    loss.backward()
    assert model.layers[0].attention.kv_down.weight.grad is not None
    assert torch.isfinite(model.layers[0].attention.kv_down.weight.grad).all()
