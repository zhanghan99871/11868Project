"""Causal behavior must not depend on future tokens."""
import numpy as np
import pytest

from minitorch.modules_transfomer import MultiHeadAttention, TransformerLayer, DecoderLM
from minitorch.tensor_ops import SimpleBackend
from minitorch.tensor import tensor_from_numpy


def test_decoder_transformer_layers_are_causal():
    model = DecoderLM(n_vocab=16, n_embd=8, n_head=2, n_positions=8,
                      p_dropout=0.0, backend=SimpleBackend)
    for name in ("t_layer_1", "t_layer_2", "t_layer_3", "t_layer_4"):
        attention = getattr(model, name).attention
        assert attention.causal, f"{name} has causal=False: future tokens leak"


def test_causal_mask_is_upper_triangular():
    mha = MultiHeadAttention(4, 1, causal=True, p_dropout=0.0, backend=SimpleBackend)
    mask = mha.create_causal_mask(1, 1, 4).to_numpy()
    assert mask.shape == (1, 1, 4, 4)
    assert np.all(mask[0, 0][np.tril_indices(4)] == 0)
    assert np.all(mask[0, 0][np.triu_indices(4, 1)] < -1e20)


@pytest.mark.parametrize("embed,heads", [(7,2), (8,0), (3,4)])
def test_invalid_attention_dimensions_rejected(embed, heads):
    with pytest.raises((ValueError, AssertionError), match="head|divis|positive|embed"):
        MultiHeadAttention(embed, heads, p_dropout=0, backend=SimpleBackend)
