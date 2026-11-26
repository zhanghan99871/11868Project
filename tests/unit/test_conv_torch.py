
import numpy as np
import pytest
import torch
import torch.nn.functional as F
import minitorch


@pytest.mark.parametrize(
    "x_shape,w_shape",
    [
        ((1, 1, 6), (1, 1, 4)),
        ((2, 2, 6), (3, 2, 2)),
    ],
)
def test_conv1d_against_torch(x_shape, w_shape):
    rng = np.random.default_rng(42)

    x_np = rng.normal(size=x_shape).astype(np.float32)
    w_np = rng.normal(size=w_shape).astype(np.float32)

    x = minitorch.tensor_from_numpy(
        x_np.copy(), requires_grad=True
    )
    w = minitorch.tensor_from_numpy(
        w_np.copy(), requires_grad=True
    )

    y = minitorch.Conv1dFun.apply(x, w)
    y.sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    wt = torch.tensor(w_np, requires_grad=True)

    # MiniTorch uses right-side zero padding.
    yt = F.conv1d(
        F.pad(xt, (0, w_shape[-1] - 1)),
        wt,
    )
    yt.sum().backward()

    np.testing.assert_allclose(
        y.to_numpy(), yt.detach().numpy(),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        x.grad.to_numpy(), xt.grad.numpy(),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        w.grad.to_numpy(), wt.grad.numpy(),
        atol=1e-4, rtol=1e-4,
    )


@pytest.mark.parametrize(
    "x_shape,w_shape",
    [
        ((1, 1, 6, 6), (1, 1, 2, 4)),
        ((2, 1, 6, 6), (1, 1, 2, 4)),
        ((2, 2, 6, 6), (3, 2, 2, 4)),
    ],
)
def test_conv2d_against_torch(x_shape, w_shape):
    rng = np.random.default_rng(42)

    x_np = rng.normal(size=x_shape).astype(np.float32)
    w_np = rng.normal(size=w_shape).astype(np.float32)

    x = minitorch.tensor_from_numpy(
        x_np.copy(), requires_grad=True
    )
    w = minitorch.tensor_from_numpy(
        w_np.copy(), requires_grad=True
    )

    y = minitorch.Conv2dFun.apply(x, w)
    y.sum().backward()

    xt = torch.tensor(x_np, requires_grad=True)
    wt = torch.tensor(w_np, requires_grad=True)

    kh, kw = w_shape[-2:]

    # MiniTorch's convention: padding on bottom and right.
    yt = F.conv2d(
        F.pad(xt, (0, kw - 1, 0, kh - 1)),
        wt,
    )
    yt.sum().backward()

    np.testing.assert_allclose(
        y.to_numpy(), yt.detach().numpy(),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        x.grad.to_numpy(), xt.grad.numpy(),
        atol=1e-4, rtol=1e-4,
    )
    np.testing.assert_allclose(
        w.grad.to_numpy(), wt.grad.numpy(),
        atol=1e-4, rtol=1e-4,
    )
