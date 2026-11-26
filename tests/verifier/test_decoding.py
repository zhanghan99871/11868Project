"""The verifier inside specdec/decoding.py's loops (numpy backend, CPU)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from specdec.decoding import autoregressive, speculative  # noqa: E402

from .helpers import export, make_torch_model, numpy_verifier, token_ids, torch_greedy  # noqa: E402

PROMPT = token_ids(8, seed=3)


@pytest.fixture
def target_and_draft(exported, tmp_path):
    model, out = exported
    draft_model = make_torch_model("gpt2_style", seed=1)  # an unrelated model: low acceptance
    return model, numpy_verifier(out), numpy_verifier(export(draft_model, tmp_path / "draft"))


def test_greedy_decoding_matches_pytorch_cached_greedy(target_and_draft):
    model, target, _ = target_and_draft
    out, _ = autoregressive(target, PROMPT, 16)
    assert out == torch_greedy(model, PROMPT, 16)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_speculative_equals_autoregressive(target_and_draft, k):
    _, target, draft = target_and_draft
    ref, _ = autoregressive(target, PROMPT, 20)
    out, stats = speculative(target, draft, PROMPT, 20, k)
    assert out == ref
    assert stats.drafted > 0


def test_self_drafting_accepts_every_draft(target_and_draft):
    _, target, _ = target_and_draft
    out, stats = speculative(target, target, PROMPT, 20, 4)
    assert out == autoregressive(target, PROMPT, 20)[0]
    assert stats.accepted == stats.drafted > 0


def test_sampling_runs(target_and_draft):
    _, target, draft = target_and_draft
    out, _ = speculative(target, draft, PROMPT, 12, 3, temperature=1.0, rng=np.random.default_rng(0))
    assert len(out) == 12
