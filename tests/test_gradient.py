"""Exercise the project's gradient checking API using MiniTorch, not torch tensors."""
import numpy as np
import pytest
from minitorch.tensor import tensor_from_numpy
from minitorch.tensor_ops import SimpleBackend
from minitorch.tensor_functions import grad_check, grad_central_difference


def test_central_difference_accepts_minitorch_only_function():
    a = tensor_from_numpy(np.array([2.0,3.0], dtype=np.float32), backend=SimpleBackend)
    def f(x):
        assert hasattr(x, "_tensor"), "Gradient checker passed a non-MiniTorch Tensor"
        return x*x
    derivative = grad_central_difference(f, a, arg=0, ind=(0,), epsilon=1e-3)
    assert derivative == pytest.approx(4.0, rel=1e-2, abs=1e-2)


def test_grad_check_respects_tolerance(monkeypatch):
    # Assert the caller-provided tolerance reaches np.testing.assert_allclose.
    import minitorch.tensor_functions as tf
    a = tensor_from_numpy(np.array([2.0,3.0], dtype=np.float32), backend=SimpleBackend)
    seen = []
    original = np.testing.assert_allclose
    def capture(*args, **kwargs):
        seen.append((kwargs.get("rtol", args[2] if len(args)>2 else None),
                     kwargs.get("atol", args[3] if len(args)>3 else None)))
        return original(*args, **kwargs)
    monkeypatch.setattr(tf.np.testing, "assert_allclose", capture)
    try:
        tf.grad_check(lambda x: x*x, a, tol=1e-3)
    finally:
        monkeypatch.setattr(tf.np.testing, "assert_allclose", original)
    assert seen and all(r == pytest.approx(1e-3) and at == pytest.approx(1e-3)
                        for r,at in seen), f"tol ignored: {seen}"
