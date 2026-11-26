"""Offline CPU smoke tests; no downloads or GPU required."""
import torch
from pathlib import Path
import sys
# The existing converter is located in <repo>/project/gpt2_loader.py.
PROJECT_DIR = Path(__file__).resolve().parents[2] / "torch_train"
sys.path.insert(0, str(PROJECT_DIR))
from decoder_mla import MLAConfig, DecoderLM
from dataloader import TokenBlockDataset
from train import lm_loss


def test_next_token_blocks():
    data = TokenBlockDataset(torch.arange(21), seq_len=4)
    assert len(data) == 5
    torch.testing.assert_close(data[0], torch.tensor([0, 1, 2, 3, 4]))
    torch.testing.assert_close(data[1], torch.tensor([4, 5, 6, 7, 8]))


def test_one_optimizer_step_and_cache():
    torch.manual_seed(123)
    cfg = MLAConfig(n_vocab=41, n_embd=32, n_head=4, n_layer=2,
                    n_positions=16, kv_lora_rank=12, q_lora_rank=None,
                    qk_nope_head_dim=8, qk_rope_head_dim=4,
                    v_head_dim=8, use_kv_norm=False, p_dropout=0.0)
    model = DecoderLM(cfg)
    model.gradient_checkpointing = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = torch.randint(0, 41, (2, 9))
    loss = lm_loss(model, batch, bf16=False)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.layers[0].attention.kv_down.weight.grad is not None
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.eval()
    x = batch[:, :6]
    with torch.no_grad():
        full = model(x)
        cache, steps = None, []
        for i in range(x.size(1)):
            logits, cache = model(x[:, i:i+1], past_key_values=cache,
                                  use_cache=True)
            steps.append(logits)
        torch.testing.assert_close(full, torch.cat(steps, 1),
                                   atol=1e-4, rtol=1e-4)
