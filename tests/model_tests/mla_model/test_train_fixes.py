"""Regression tests for torch_train fixes (offline, CPU only).

1. DecoderLM.gradient_checkpointing used to be a plain attribute that nothing read, so
   train.py logged "gradient checkpointing=True" while storing every activation.
2. train.py only saved best.pt inside the periodic evaluation, so with --eval-every 0, or when
   the final evaluation was the best, the trained weights were never written.
"""
import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2] / "torch_train"
sys.path.insert(0, str(PROJECT_DIR))
import decoder_mla  # noqa: E402
from decoder_mla import DecoderLM, MLAConfig  # noqa: E402
from train import lm_loss, save_if_best  # noqa: E402


def small_cfg():
    return MLAConfig(n_vocab=41, n_embd=32, n_head=4, n_layer=3,
                     n_positions=16, kv_lora_rank=12, q_lora_rank=None,
                     qk_nope_head_dim=8, qk_rope_head_dim=4,
                     v_head_dim=8, use_kv_norm=False, p_dropout=0.0)


@pytest.fixture
def count_checkpoint_calls(monkeypatch):
    calls = []
    real = decoder_mla.checkpoint

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(decoder_mla, "checkpoint", counting)
    return calls


def loss_and_grads(model, batch):
    model.zero_grad(set_to_none=True)
    loss = lm_loss(model, batch, bf16=False)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters()
                           if p.grad is not None}


def test_checkpointing_is_off_by_default():
    assert DecoderLM(small_cfg()).gradient_checkpointing is False


def test_checkpointing_recomputes_every_layer_and_matches(count_checkpoint_calls):
    torch.manual_seed(0)
    model = DecoderLM(small_cfg()).train()
    batch = torch.randint(0, 41, (2, 9))

    plain_loss, plain_grads = loss_and_grads(model, batch)
    assert count_checkpoint_calls == []

    model.gradient_checkpointing = True
    ckpt_loss, ckpt_grads = loss_and_grads(model, batch)
    assert len(count_checkpoint_calls) == model.cfg.n_layer

    torch.testing.assert_close(ckpt_loss, plain_loss)
    assert ckpt_grads.keys() == plain_grads.keys()
    for name in plain_grads:
        torch.testing.assert_close(ckpt_grads[name], plain_grads[name],
                                   atol=1e-6, rtol=1e-5, msg=name)


def test_checkpointing_skipped_for_eval_no_grad_and_cache(count_checkpoint_calls):
    torch.manual_seed(1)
    model = DecoderLM(small_cfg())
    model.gradient_checkpointing = True
    ids = torch.randint(0, 41, (1, 6))

    model.eval()
    with torch.no_grad():
        full = model(ids)
        cache, steps = None, []
        for i in range(ids.size(1)):
            logits, cache = model(ids[:, i:i + 1], past_key_values=cache, use_cache=True)
            steps.append(logits)
    torch.testing.assert_close(full, torch.cat(steps, 1), atol=1e-4, rtol=1e-4)

    model.train()
    with torch.no_grad():
        model(ids)
    logits, cache = model(ids, use_cache=True)  # training mode, but a cache was requested
    assert len(cache) == model.cfg.n_layer
    assert count_checkpoint_calls == []


def test_save_if_best_saves_first_result_and_only_improvements(tmp_path):
    torch.manual_seed(2)
    cfg = small_cfg()
    model = DecoderLM(cfg)
    path = tmp_path / "best.pt"

    # e.g. --eval-every 0: the final evaluation is the first one, so it must be saved
    best = save_if_best({"loss": 3.0}, math.inf, path, model, 10, cfg, "src")
    assert best == 3.0 and path.exists()
    assert torch.load(path, weights_only=True)["step"] == 10

    assert save_if_best({"loss": 3.5}, best, path, model, 20, cfg, "src") == 3.0
    assert torch.load(path, weights_only=True)["step"] == 10  # worse: not overwritten

    assert save_if_best({"loss": float("nan")}, best, path, model, 30, cfg, "src") == 3.0
    assert torch.load(path, weights_only=True)["step"] == 10  # NaN: not overwritten

    best = save_if_best({"loss": 2.5}, best, path, model, 40, cfg, "src")
    saved = torch.load(path, weights_only=True)
    assert best == 2.5 and saved["step"] == 40
    assert saved["metadata"] == {"source": "src", "validation": {"loss": 2.5}}

    # the checkpoint loads back into a fresh model
    fresh = DecoderLM(MLAConfig(**saved["config"]))
    fresh.load_state_dict(saved["model"], strict=True)
