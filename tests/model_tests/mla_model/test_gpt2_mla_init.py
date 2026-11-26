"""Offline GPT-2 state-dict transfer verification using a synthetic checkpoint."""
from types import SimpleNamespace
import torch
from pathlib import Path
import sys
# The existing converter is located in <repo>/project/gpt2_loader.py.
PROJECT_DIR = Path(__file__).resolve().parents[2] / "torch_train"
sys.path.insert(0, str(PROJECT_DIR))
from decoder_mla import DecoderLM
from gpt2_mla_init import make_gpt2_mla_config, initialize_from_gpt2


class FakeGPT2:
    def __init__(self):
        self.config = SimpleNamespace(n_embd=16, n_head=4, n_layer=2,
                                      n_positions=32, vocab_size=49,
                                      layer_norm_epsilon=1e-5)
        cfg = self.config
        d, v = cfg.n_embd, cfg.vocab_size
        self.sd = {
            'transformer.wte.weight': torch.randn(v, d),
            'transformer.wpe.weight': torch.randn(cfg.n_positions, d),
            'transformer.ln_f.weight': torch.randn(d),
            'transformer.ln_f.bias': torch.randn(d),
        }
        for i in range(cfg.n_layer):
            p = f'transformer.h.{i}'
            for name in ('ln_1', 'ln_2'):
                self.sd[f'{p}.{name}.weight'] = torch.randn(d)
                self.sd[f'{p}.{name}.bias'] = torch.randn(d)
            self.sd[f'{p}.attn.c_attn.weight'] = torch.randn(d, 3*d)
            self.sd[f'{p}.attn.c_attn.bias'] = torch.randn(3*d)
            self.sd[f'{p}.attn.c_proj.weight'] = torch.randn(d, d)
            self.sd[f'{p}.attn.c_proj.bias'] = torch.randn(d)
            self.sd[f'{p}.mlp.c_fc.weight'] = torch.randn(d, 4*d)
            self.sd[f'{p}.mlp.c_fc.bias'] = torch.randn(4*d)
            self.sd[f'{p}.mlp.c_proj.weight'] = torch.randn(4*d, d)
            self.sd[f'{p}.mlp.c_proj.bias'] = torch.randn(d)

    def state_dict(self):
        return self.sd


def test_pretrained_transfer_synthetic():
    torch.manual_seed(10)
    hf = FakeGPT2()
    cfg = make_gpt2_mla_config(hf.config, kv_rank=8, rope_dim=2)
    model = DecoderLM(cfg)
    errors = initialize_from_gpt2(model, hf, svd_device='cpu', svd_niter=1)
    assert len(errors) == hf.config.n_layer
    assert all(0 <= x <= 1 for x in errors)
    torch.testing.assert_close(model.token_embeddings.weight, hf.sd['transformer.wte.weight'])
    assert model.lm_head.weight.data_ptr() == model.token_embeddings.weight.data_ptr()
    assert isinstance(model.layers[0].attention.kv_norm, torch.nn.Identity)
    x = torch.randint(0, cfg.n_vocab, (2, 5))
    logits = model(x)
    assert logits.shape == (2, 5, cfg.n_vocab)
    logits.square().mean().backward()
    assert model.layers[0].attention.kv_up.weight.grad is not None
