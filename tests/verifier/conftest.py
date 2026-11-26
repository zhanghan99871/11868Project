import pytest


@pytest.fixture(params=["gpt2_style", "lora_q_kv_norm_no_bias"])
def exported(request, tmp_path):
    """(PyTorch reference model, directory with its torch-free export)."""
    pytest.importorskip("torch")
    from .helpers import export, make_torch_model

    model = make_torch_model(request.param)
    return model, export(model, tmp_path / "verifier")
