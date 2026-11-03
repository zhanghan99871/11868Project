
import numpy as np
from transformers import GPT2LMHeadModel

import minitorch


def load_gpt2_weights(mt_model, hf_model, backend):
    sd = hf_model.state_dict()

    def assign(parameter, torch_weight):
        array = (
            torch_weight.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .copy()
        )

        assert parameter.value.shape == array.shape, (
            parameter.value.shape, array.shape
        )

        parameter.value = minitorch.tensor_from_numpy(
            array,
            backend=backend,
            requires_grad=True,
        )

    def load_linear(mt_linear, weight, bias=None):
        # MiniTorch Linear: (in_features, out_features)
        assign(mt_linear.weights, weight)

        if mt_linear.bias is not None:
            assert bias is not None
            assign(mt_linear.bias, bias)

    # Embeddings
    assign(
        mt_model.token_embeddings.weights,
        sd["transformer.wte.weight"],
    )
    assign(
        mt_model.position_embeddings.weights,
        sd["transformer.wpe.weight"],
    )

    # Transformer blocks
    for i, layer in enumerate(mt_model.layers):
        prefix = f"transformer.h.{i}"

        # GPT-2 QKV Conv1D weights are already
        # stored as (input_dim, 3 * output_dim).
        qkv = sd[f"{prefix}.attn.c_attn.weight"]
        q, k, v = qkv.chunk(3, dim=1)

        qkv_bias = sd[f"{prefix}.attn.c_attn.bias"]
        bq, bk, bv = qkv_bias.chunk(3, dim=0)

        load_linear(layer.attention.q_projection, q, bq)
        load_linear(layer.attention.k_projection, k, bk)
        load_linear(layer.attention.v_projection, v, bv)

        load_linear(
            layer.attention.out_projection,
            sd[f"{prefix}.attn.c_proj.weight"],
            sd[f"{prefix}.attn.c_proj.bias"],
        )

        # LayerNorm
        assign(layer.ln_1.weights, sd[f"{prefix}.ln_1.weight"])
        assign(layer.ln_1.bias, sd[f"{prefix}.ln_1.bias"])

        assign(layer.ln_2.weights, sd[f"{prefix}.ln_2.weight"])
        assign(layer.ln_2.bias, sd[f"{prefix}.ln_2.bias"])

        # MLP
        load_linear(
            layer.ff.linear_in,
            sd[f"{prefix}.mlp.c_fc.weight"],
            sd[f"{prefix}.mlp.c_fc.bias"],
        )
        load_linear(
            layer.ff.linear_out,
            sd[f"{prefix}.mlp.c_proj.weight"],
            sd[f"{prefix}.mlp.c_proj.bias"],
        )

    # Final LayerNorm
    assign(mt_model.ln.weights, sd["transformer.ln_f.weight"])
    assign(mt_model.ln.bias, sd["transformer.ln_f.bias"])

    # Tied LM head, transposed for MiniTorch Linear
    assign(
        mt_model.lm_head.weights,
        sd["transformer.wte.weight"].T,
    )
