"""The PyTorch H-Spec drafter (torch_train/hspec_drafter.py), against a tiny MLA verifier (CPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from .helpers import (DRAFTER_VARIANTS, TRAINED_VERIFIER, DrafterConfig, MLAConfig,  # noqa: E402  (sets sys.path)
                      drafter_config, make_drafter, make_verifier, token_ids)

import hspec_drafter as hd  # noqa: E402

T, K = 20, 4  # sequence length, block size of the tiny drafter


@pytest.fixture(scope="module")
def verifier():
    return make_verifier()


@pytest.fixture(scope="module")
def batch(verifier):
    tokens = token_ids(2, T)
    cfg = drafter_config(verifier.cfg)
    latents, hidden, logits = hd.target_context(verifier, tokens, cfg)
    anchors = torch.tensor([[1, 6, 13], [4, 5, 15]])  # includes the extremes 1 and T-1-K
    return tokens, anchors, latents, hidden, logits


# --- configuration ---------------------------------------------------------------------------

def test_defaults_fit_the_trained_verifier():
    vcfg = MLAConfig(**TRAINED_VERIFIER)
    cfg = DrafterConfig.for_verifier(vcfg)
    assert cfg == DrafterConfig()  # the class defaults are this verifier's
    assert cfg.target_layers == (12, 14, 16, 18, 20, 22) and cfg.tap_layers == (1, 6, 11, 16, 22)
    assert (cfg.n_embd, cfg.n_layer, cfg.middle_dim, cfg.block_size) == (768, 6, 3072, 7)  # distilgpt2 trunk
    cfg.check_verifier(vcfg)


def test_default_layer_maps():
    assert hd.default_layer_maps(36, 3)[1] == (1, 9, 17, 25, 34)  # H-Spec's Qwen3 taps
    assert hd.default_layer_maps(32, 3)[1] == (1, 8, 15, 22, 30)  # H-Spec's Llama-3.1 taps
    assert hd.default_layer_maps(24, 6) == ((12, 14, 16, 18, 20, 22), (1, 6, 11, 16, 22))
    for L in (1, 2, 3):  # tiny verifiers still give valid indices
        reads, taps = hd.default_layer_maps(L, 2)
        assert all(0 <= i < L for i in reads + taps)


def test_config_validation(verifier):
    with pytest.raises(ValueError, match="one target layer per drafter layer"):
        DrafterConfig(n_layer=6, target_layers=(1, 2))
    with pytest.raises(ValueError, match="even"):
        DrafterConfig(qk_rope_head_dim=3)
    cfg = drafter_config(verifier.cfg)
    cfg.check_verifier(verifier.cfg)
    with pytest.raises(ValueError, match="n_head"):
        drafter_config(verifier.cfg, n_head=2).check_verifier(verifier.cfg)
    with pytest.raises(ValueError, match="out of range"):
        drafter_config(verifier.cfg, target_layers=(0, 4)).check_verifier(verifier.cfg)


# --- building blocks -------------------------------------------------------------------------

def test_rope_matches_decoder_mla():
    from decoder_mla import apply_rope
    x = torch.randn(2, 3, 5, 8)
    pos = torch.arange(7, 12)
    np.testing.assert_allclose(hd.rope(x, pos.expand(2, 5)), apply_rope(x, pos), atol=1e-6)
    per_token = torch.tensor([[3, 9, 4, 0, 30]] * 2)  # arbitrary positions per token
    for t in range(5):
        np.testing.assert_allclose(hd.rope(x, per_token)[:, :, t], apply_rope(x[:, :, t:t + 1], per_token[0, t:t + 1])[:, :, 0],
                                   atol=1e-6)


def test_block_attention_mask():
    mask = hd.block_attention_mask(torch.tensor([[2, 0]]), n_prefix_keys=3, block_len=2)
    expected = torch.tensor([
        # prefix 0 1 2 | block0 q0 q1 | block1 q0 q1
        [1, 1, 0, 1, 0, 0, 0],  # block 0 (prefix 2), token 0
        [1, 1, 0, 1, 1, 0, 0],  # block 0, token 1
        [0, 0, 0, 0, 0, 1, 0],  # block 1 (prefix 0), token 0
        [0, 0, 0, 0, 0, 1, 1],  # block 1, token 1
    ], dtype=torch.bool)
    assert torch.equal(mask[0], expected)


# --- forward ----------------------------------------------------------------------------------

@pytest.mark.parametrize("variant", sorted(DRAFTER_VARIANTS))
def test_training_forward_equals_draft_block(verifier, batch, variant):
    tokens, anchors, latents, hidden, _ = batch
    drafter = make_drafter(verifier, **DRAFTER_VARIANTS[variant])
    with torch.no_grad():
        logits = drafter(tokens, anchors, latents, hidden)
        assert logits.shape == (2, 3, K, drafter.cfg.n_vocab)
        for b in range(2):
            for n in range(3):
                a = int(anchors[b, n])
                single = drafter.draft_block(int(tokens[b, a]), [(c[b, :a], kr[b, :a]) for c, kr in latents],
                                             hidden[b, a - 1])
                torch.testing.assert_close(logits[b, n], single, atol=1e-5, rtol=1e-5)


def test_blocks_see_only_their_prefix(verifier, batch):
    """No leakage: a block's logits depend on the prefix before its anchor, never on later
    tokens/latents/hidden states or on other blocks."""
    tokens, anchors, latents, hidden, _ = batch
    drafter = make_drafter(verifier)
    b, n = 1, 1
    a = int(anchors[b, n])
    with torch.no_grad():
        ref = drafter(tokens, anchors, latents, hidden)[b, n]

        def run(tok=tokens, anc=anchors, lat=latents, hid=hidden, idx=n):
            return drafter(tok, anc, lat, hid)[b, idx]

        tok2 = tokens.clone()
        tok2[b, a + 1:] = (tok2[b, a + 1:] + 1) % drafter.cfg.n_vocab
        lat2 = [(c.clone(), kr.clone()) for c, kr in latents]
        for c, kr in lat2:
            c[b, a:] += 1.0
            kr[b, a:] += 1.0
        hid2 = hidden.clone()
        hid2[b, a:] += 1.0
        torch.testing.assert_close(run(tok=tok2, lat=lat2, hid=hid2), ref, atol=0, rtol=0)
        other = anchors.clone()
        other[b, 0], other[b, 2] = 2, 9  # different neighbouring blocks
        torch.testing.assert_close(run(anc=other), ref, atol=1e-6, rtol=1e-6)

        # sanity: the inputs it may see do matter. Perturb with seeded noise, not a constant: the
        # summary LayerNorms each tap, so adding the same value to every element cancels exactly.
        # Require a change far above float32 rounding (~1e-7), which `not allclose` would accept.
        g = torch.Generator().manual_seed(0)

        def changed(out):
            return (out - ref).abs().max().item() > 1e-5

        lat3 = [(c.clone(), kr.clone()) for c, kr in latents]
        lat3[0][0][b, a - 1] += torch.randn(lat3[0][0][b, a - 1].shape, generator=g)
        assert changed(run(lat=lat3))
        hid3 = hidden.clone()
        hid3[b, a - 1] += torch.randn(hid3[b, a - 1].shape, generator=g)
        assert changed(run(hid=hid3))
        tok3 = tokens.clone()
        tok3[b, a] = (tok3[b, a] + 1) % drafter.cfg.n_vocab
        assert changed(run(tok=tok3))


def test_input_checks(verifier, batch):
    tokens, anchors, latents, hidden, _ = batch
    drafter = make_drafter(verifier)
    with pytest.raises(ValueError, match="anchors"):
        drafter(tokens, torch.tensor([[0, 3], [4, 5]]), latents, hidden)
    with pytest.raises(ValueError, match="same prefix"):
        drafter.draft_block(1, [(latents[0][0][0, :5], latents[0][1][0, :5]),
                                (latents[1][0][0, :4], latents[1][1][0, :4])], hidden[0, 4])
    fresh = hd.HSpecDrafter(drafter.cfg)
    with pytest.raises(RuntimeError, match="set_target_head"):
        fresh(tokens, anchors, latents, hidden)
    with pytest.raises(ValueError, match="target head"):
        fresh.set_target_head(torch.zeros(3, 3))


def test_target_context_matches_the_verifier(verifier):
    tokens = token_ids(2, 12, seed=4)
    cfg = drafter_config(verifier.cfg)
    hooked = {}
    hooks = [verifier.layers[m].register_forward_hook(
        lambda mod, inp, out, m=m: hooked.__setitem__(m, out[0].clone())) for m in range(verifier.cfg.n_layer)]
    with torch.no_grad():
        ref_logits, ref_cache = verifier(tokens, use_cache=True)
    for h in hooks:
        h.remove()
    verifier.train()
    latents, hidden, logits = hd.target_context(verifier, tokens, cfg)
    assert verifier.training  # restored
    verifier.eval()
    torch.testing.assert_close(logits, ref_logits)
    for (c, kr), m in zip(latents, cfg.target_layers):
        torch.testing.assert_close(c, ref_cache[m][0])
        torch.testing.assert_close(kr, ref_cache[m][1])
    for i, m in enumerate(cfg.tap_layers):
        torch.testing.assert_close(hidden[:, :, i], hooked[m])


# --- initialization --------------------------------------------------------------------------

def test_init_from_verifier(verifier):
    drafter = make_drafter(verifier)
    fresh = hd.HSpecDrafter(drafter.cfg)
    hd.init_from_verifier(fresh, verifier)
    for layer, m in zip(fresh.layers, fresh.cfg.target_layers):
        torch.testing.assert_close(layer.attention.kv_up.weight, verifier.layers[m].attention.kv_up.weight)
        torch.testing.assert_close(layer.attention.kv_up.bias, verifier.layers[m].attention.kv_up.bias)
    torch.testing.assert_close(fresh.target_head, verifier.lm_head.weight)
    assert "target_head" not in fresh.state_dict()  # frozen and not saved
    assert all(p is not fresh.target_head for p in fresh.parameters())
    with pytest.raises(ValueError, match="does not match"):
        hd.init_from_verifier(hd.HSpecDrafter(drafter_config(verifier.cfg, n_head=2)), verifier)


def test_init_from_gpt2():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    gpt2 = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=97, n_positions=64, n_embd=24, n_layer=4, n_head=4))
    vcfg = make_verifier().cfg
    cfg = drafter_config(vcfg)
    drafter = hd.HSpecDrafter(cfg)
    used = hd.init_from_gpt2(drafter, gpt2.state_dict())
    assert used == [0, 3]  # evenly spaced over 4 source blocks
    sd = gpt2.state_dict()
    torch.testing.assert_close(drafter.token_embeddings.weight, sd["transformer.wte.weight"])
    torch.testing.assert_close(drafter.position_embeddings.weight, sd["transformer.wpe.weight"])
    for layer, i in zip(drafter.layers, used):
        p = f"transformer.h.{i}."
        torch.testing.assert_close(layer.ln_1.weight, sd[p + "ln_1.weight"])
        torch.testing.assert_close(layer.ln_2.bias, sd[p + "ln_2.bias"])
        torch.testing.assert_close(layer.ff.linear_in.weight, sd[p + "mlp.c_fc.weight"].T)
        torch.testing.assert_close(layer.ff.linear_out.weight, sd[p + "mlp.c_proj.weight"].T)
        torch.testing.assert_close(layer.ff.linear_out.bias, sd[p + "mlp.c_proj.bias"])
    torch.testing.assert_close(drafter.ln.weight, sd["transformer.ln_f.weight"])
    assert hd.init_from_gpt2(hd.HSpecDrafter(cfg), sd, layers=[2, 2]) == [2, 2]
    with pytest.raises(ValueError, match="source has 4 blocks"):
        hd.init_from_gpt2(hd.HSpecDrafter(DrafterConfig.for_verifier(vcfg, n_layer=5, n_embd=24, n_taps=3)), sd)
    with pytest.raises(ValueError, match="embeddings"):
        hd.init_from_gpt2(hd.HSpecDrafter(drafter_config(vcfg, n_embd=32)), sd)


# --- training helpers ------------------------------------------------------------------------

def test_sample_anchors():
    anchors = hd.sample_anchors(3, T, K, 5, generator=torch.Generator().manual_seed(0))
    assert anchors.shape == (3, 5)
    assert anchors.min() >= 1 and anchors.max() <= T - 1 - K
    for row in anchors:
        assert row.unique().numel() == 5 and torch.equal(row, row.sort().values)
    assert hd.sample_anchors(1, T, K, T - 1 - K).tolist() == [list(range(1, T - K))]  # every anchor
    with pytest.raises(ValueError):
        hd.sample_anchors(1, T, K, T - K)


def test_block_targets(batch):
    tokens, anchors, _, _, logits = batch
    targets, teacher = hd.block_targets(tokens, logits, anchors, K)
    for b in range(2):
        for n in range(3):
            a = int(anchors[b, n])
            assert targets[b, n].tolist() == tokens[b, a + 1:a + 1 + K].tolist()
            torch.testing.assert_close(teacher[b, n], logits[b, a:a + K])  # predicted a+1..a+K
    with pytest.raises(ValueError):
        hd.block_targets(tokens, logits, torch.tensor([[T - K], [1]]), K)


def test_loss_terms_and_gradients(verifier, batch):
    tokens, anchors, latents, hidden, logits = batch
    targets, teacher = hd.block_targets(tokens, logits, anchors, K)
    w = hd.position_weights(K)
    assert w[0] == 1 and torch.all(w[1:] < w[:-1])

    perfect, stats = hd.drafter_loss(teacher, teacher.argmax(-1), teacher)  # drafter == verifier
    assert stats["tv"] < 1e-6 and min(stats["acceptance"]) > 1 - 1e-6 and min(stats["top1_agree"]) == 1

    drafter = make_drafter(verifier).train()
    loss, stats = hd.drafter_loss(drafter(tokens, anchors, latents, hidden), targets, teacher)
    assert loss.item() > 0 and len(stats["acceptance"]) == K
    loss.backward()
    for name, p in drafter.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        if name != "position_embeddings.weight" and name != "token_embeddings.weight":
            assert p.grad.abs().sum() > 0, name
    assert all(p.grad is None for p in verifier.parameters())


def test_overfits_a_fixed_batch(verifier, batch):
    """Smoke test of a training step loop: loss drops and agreement with the verifier rises."""
    tokens, anchors, latents, hidden, logits = batch
    targets, teacher = hd.block_targets(tokens, logits, anchors, K)
    drafter = make_drafter(verifier).train()
    opt = torch.optim.AdamW(drafter.parameters(), lr=3e-3)
    history = []
    for _ in range(80):
        loss, stats = hd.drafter_loss(drafter(tokens, anchors, latents, hidden), targets, teacher)
        opt.zero_grad()
        loss.backward()
        opt.step()
        history.append((loss.item(), np.mean(stats["acceptance"])))
    assert history[-1][0] < 0.5 * history[0][0]
    assert history[-1][1] > history[0][1] + 0.2
