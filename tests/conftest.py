"""Regression tests for reported MiniTorch implementation issues."""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def cuda_backend():
    torch = pytest.importorskip("torch")
    pytest.importorskip("pycuda.driver")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    try:
        from minitorch.cuda_kernel_ops import CudaKernelOps
        from minitorch.tensor_ops import TensorBackend
    except (ImportError, OSError) as exc:
        pytest.skip(f"CUDA extension unavailable: {exc}")
    return TensorBackend(CudaKernelOps)
