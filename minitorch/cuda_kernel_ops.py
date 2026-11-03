from typing import Callable, Optional

from . import operators
from .tensor import Tensor
from .tensor_data import (
    MAX_DIMS,
    Shape,
    Storage,
    Strides,
    TensorData,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps
from .tensor_functions import tensor_from_numpy

import ctypes
from pathlib import Path
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import torch

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/combine.so")
lib_softmax = ctypes.CDLL("minitorch/cuda_kernels/softmax_kernel.so")
lib_layernorm = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")
lib_mla = ctypes.CDLL(str(Path(__file__).resolve().parent / "cuda_kernels" / "mla_kernel.so"))
datatype = np.float32

# function map
fn_map = {
  operators.add: 1,
  operators.mul: 2,
  operators.id: 3,
  operators.neg: 4,
  operators.lt: 5,
  operators.eq: 6,
  operators.sigmoid: 7,
  operators.relu: 8,
  operators.relu_back: 9,
  operators.log: 10,
  operators.log_back: 11,
  operators.exp: 12,
  operators.inv: 13,
  operators.inv_back: 14,
  operators.is_close: 15,
  operators.max: 16,
  operators.pow: 17, 
  operators.tanh: 18
}

THREADS_PER_BLOCK = 32

class CudaKernelOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        "See `tensor_ops.py`"
        fn_id = fn_map[fn]

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)

            lib.tensorMap.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # in_size
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorMap.restype = None
            
            # assert out.size == a.size, f"zip {out.size}, {a.size}"

            lib.tensorMap(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)

            lib.tensorZip.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                ctypes.c_int,                                                            # out_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # a_strides
                ctypes.c_int,                                                            # a_size
                ctypes.c_int,                                                            # a_shape_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),    # b_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # b_strides
                ctypes.c_int,                                                            # b_size
                ctypes.c_int,                                                            # b_shape_size
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorZip.restype = None

            # assert out.size == a.size, f"zip {out.size}, {a.size}"
            # assert out.size == b.size, f"zip {out.size}, {b.size}"

            lib.tensorZip(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                len(out.shape),
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                a.size,
                len(a.shape),
                b._tensor._storage,
                b._tensor._shape.astype(np.int32),
                b._tensor._strides.astype(np.int32),
                b.size,
                len(b.shape),
                fn_id
            )
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0) -> Callable[[Tensor, int], Tensor]:
        fn_id = fn_map[fn]

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))

            lib.tensorReduce.argtypes = [
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # out_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # out_strides
                ctypes.c_int,                                                            # out_size
                np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),  # in_storage
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_shape
                np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),    # in_strides
                ctypes.c_int,                                                            # reduce_dim
                ctypes.c_float,                                                          # reduce_value
                ctypes.c_int,                                                            # shape_len
                ctypes.c_int,                                                            # fn_id
            ]

            lib.tensorReduce.restype = None

            lib.tensorReduce(
                out._tensor._storage,
                out._tensor._shape.astype(np.int32),
                out._tensor._strides.astype(np.int32),
                out.size,
                a._tensor._storage,
                a._tensor._shape.astype(np.int32),
                a._tensor._strides.astype(np.int32),
                dim,
                start,
                len(a.shape),
                fn_id
            )

            return out

        return ret

    @staticmethod
    def matrix_multiply_cublas(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]

        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2],
                                    a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2],
                                    b.shape[-1])
        assert a.shape[0] == b.shape[0]

        bs, m, n, k = a.shape[0], a.shape[1], b.shape[2], a.shape[2]
        A, B = a.to_numpy(), b.to_numpy()

        # Convert A and B to column-major order
        A_fortran = np.transpose(A, (0, 2, 1))
        B_fortran = np.transpose(B, (0, 2, 1))

        # Flatten A and B for sending to GPU
        A_flat = A_fortran.reshape(bs, -1)
        B_flat = B_fortran.reshape(bs, -1)

        # Allocate memory on GPU
        A_gpu = cuda.mem_alloc(A_flat.nbytes)
        B_gpu = cuda.mem_alloc(B_flat.nbytes)
        C_gpu = cuda.mem_alloc(bs * m * n * A.itemsize)

        # Copy data to GPU
        cuda.memcpy_htod(A_gpu, A_flat)
        cuda.memcpy_htod(B_gpu, B_flat)

        # Prepare arrays of pointers
        A_gpu_ptrs = np.array(
            [int(A_gpu) + i * m * k * A.itemsize for i in range(bs)],
            dtype=np.uint64)
        B_gpu_ptrs = np.array(
            [int(B_gpu) + i * k * n * B.itemsize for i in range(bs)],
            dtype=np.uint64)
        C_gpu_ptrs = np.array(
            [int(C_gpu) + i * m * n * A.itemsize for i in range(bs)],
            dtype=np.uint64)

        # Allocate device memory for arrays of pointers
        A_array_gpu = cuda.mem_alloc(A_gpu_ptrs.nbytes)
        B_array_gpu = cuda.mem_alloc(B_gpu_ptrs.nbytes)
        C_array_gpu = cuda.mem_alloc(C_gpu_ptrs.nbytes)

        # Copy arrays of pointers to device memory
        cuda.memcpy_htod(A_array_gpu, A_gpu_ptrs)
        cuda.memcpy_htod(B_array_gpu, B_gpu_ptrs)
        cuda.memcpy_htod(C_array_gpu, C_gpu_ptrs)

        # Set argument types for the kernel function
        lib_mm.batchedMatMulKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int]

        # Launch kernel
        lib_mm.batchedMatMulKernel(
            int(A_array_gpu), int(B_array_gpu), int(C_array_gpu), m, k, n, bs)

        # Synchronize device to ensure computation is complete
        cuda.Context.synchronize()

        # Copy back the result
        C = np.empty((bs, n, m), dtype=A.dtype)
        cuda.memcpy_dtoh(C, C_gpu)
        C = np.transpose(C, (0, 2, 1))

        c = tensor_from_numpy(
            np.ascontiguousarray(C),
            backend=a.backend, requires_grad=a.requires_grad()).contiguous()

        # Undo 3d if we added it.
        if both_2d:
            c = c.view(c.shape[1], c.shape[2])
        if len(ls) > 3:
            c = c.view(*ls)
        return c

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        # handle cases with more dimensions [64, 4, 32, 128] x [64, 4, 128, 32]
        more_3d = False
        if len(out.shape) > 3:
            # print(f"Debug in matmul: output shape {ls}")
            more_3d = True
            out = out.view(np.prod(out.shape[:-2]), out.shape[-2], out.shape[-1])
            nshape = out._tensor._shape
            nstrides = out._tensor._strides
            # print(f"Debug in matmul: batched dim [:-2] and get the strides {nshape, nstrides}")
        if len(a.shape) > 3:
            a = a.contiguous().view(np.prod(a.shape[:-2]), a.shape[-2], a.shape[-1])
        if len(b.shape) > 3:
            b = b.contiguous().view(np.prod(b.shape[:-2]), b.shape[-2], b.shape[-1])
        
        assert a.shape[0] == b.shape[0]
        assert a.shape[0] == out.shape[0]

        lib.MatrixMultiply.argtypes = [
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # out_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # out_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # a_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # a_strides
            np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),   # b_storage
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_shape
            np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),     # b_strides
            ctypes.c_int,                                                             # batch_size
            ctypes.c_int,                                                             # out_shape[1], m
            ctypes.c_int                                                              # out_shape[2], p
        ]

        lib.MatrixMultiply.restype = None

        assert len(out._tensor._shape) == 3, f"{len(out._tensor._shape)}"
        assert len(out._tensor._strides) == 3, f"{len(out._tensor._strides)}"
        assert len(a._tensor._shape) == 3
        assert len(a._tensor._strides) == 3
        assert len(b._tensor._shape) == 3
        assert len(b._tensor._strides) == 3

        lib.MatrixMultiply(
            out._tensor._storage,
            out._tensor._shape.astype(np.int32),
            out._tensor._strides.astype(np.int32),
            a._tensor._storage,
            a._tensor._shape.astype(np.int32),
            a._tensor._strides.astype(np.int32),
            b._tensor._storage,
            b._tensor._shape.astype(np.int32),
            b._tensor._strides.astype(np.int32),
            a.shape[0],
            a.shape[1],
            b.shape[2]
        )

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        if more_3d:
            out = out.view(*ls)
            # print(f"Debug in matmul: output shape {out.shape}")
        return out

    @staticmethod
    def attn_softmax_fw(inp: Tensor, mask: Tensor):
      """Fused row-wise softmax(logits + additive_mask), no input mutation.

      Both tensors have 4 dimensions; mask can broadcast on any dimension.
      The native interface expects contiguous float32 NumPy host buffers.
      """
      if len(inp.shape) != 4 or len(mask.shape) != 4:
        raise ValueError("Attention logits and mask must be 4D")
      batch_size, nhead, from_len, to_len = inp.shape
      mb, mh, mq, mk = mask.shape
      if any(m not in (1, d) for m, d in zip(mask.shape, inp.shape)):
        raise ValueError(f"Mask shape {mask.shape} cannot broadcast to {inp.shape}")
      inp_c = inp.contiguous()
      mask_c = mask.contiguous()
      out = inp.zeros(inp.shape)
      stream = torch.cuda.current_stream().cuda_stream
      ptr = np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags="C_CONTIGUOUS")
      lib_softmax.launch_attn_softmax.argtypes = [
          ptr, ptr, ptr,
          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
          ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
          ctypes.c_void_p,
      ]
      lib_softmax.launch_attn_softmax.restype = None
      lib_softmax.launch_attn_softmax(
          inp_c._tensor._storage,
          mask_c._tensor._storage,
          out._tensor._storage,
          batch_size, nhead, from_len, to_len,
          mb, mh, mq, mk, stream,
      )
      return out

    @staticmethod
    def attn_softmax_bw(out_grad: Tensor, soft_inp: Tensor):
      """Compute softmax input gradient without overwriting out_grad."""
      if len(soft_inp.shape) != 4 or out_grad.shape != soft_inp.shape:
        raise ValueError("Upstream gradient and probabilities must have equal 4D shapes")
      batch_size, nhead, seq_len, softmax_len = soft_inp.shape
      rows = batch_size * nhead * seq_len
      grad_c = out_grad.contiguous()
      soft_c = soft_inp.contiguous()
      grad_input = out_grad.zeros(out_grad.shape)
      stream = torch.cuda.current_stream().cuda_stream
      ptr = np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags="C_CONTIGUOUS")
      lib_softmax.launch_attn_softmax_bw.argtypes = [
          ptr, ptr, ptr, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
      ]
      lib_softmax.launch_attn_softmax_bw.restype = None
      lib_softmax.launch_attn_softmax_bw(
          grad_c._tensor._storage,
          soft_c._tensor._storage,
          grad_input._tensor._storage,
          rows, softmax_len, stream,
      )
      return grad_input

    @staticmethod
    def layernorm_fw(inp: Tensor, gamma: Tensor, beta: Tensor):
      #   BEGIN ASSIGN4_2_1
    #   raise("Not implemented")
      rows, hidden_size = inp.shape
      stream = torch.cuda.current_stream().cuda_stream
      ln_res = inp.zeros((rows, hidden_size))
      vars = inp.zeros((rows,))
      means = inp.zeros((rows,))
      

      lib_layernorm.launch_layernorm.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_layernorm.launch_layernorm.restype = None

      lib_layernorm.launch_layernorm(
        ln_res._tensor._storage,
        vars._tensor._storage,
        means._tensor._storage,
        inp._tensor._storage,
        gamma._tensor._storage,
        beta._tensor._storage,
        rows,
        hidden_size,
        stream
      ) 

      return ln_res, vars, means 
      #   END ASSIGN4_2_1
      
    @staticmethod
    def layernorm_bw(out_grad: Tensor, inp: Tensor, gamma: Tensor, beta: Tensor, var: Tensor, mean: Tensor):
      #   BEGIN ASSIGN4_2_2
    #   raise("Not implemented")
      rows, hidden_size = inp.shape
      stream = torch.cuda.current_stream().cuda_stream
      gamma_grad = inp.zeros((hidden_size,))
      beta_grad = inp.zeros((hidden_size,))
      inp_grad = inp.zeros((rows, hidden_size))
      

      lib_layernorm.launch_layernorm_bw.argtypes = [
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=datatype, ndim=1, flags='C_CONTIGUOUS'),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_void_p
      ]
      lib_layernorm.launch_layernorm_bw.restype = None

      lib_layernorm.launch_layernorm_bw(
        gamma_grad._tensor._storage,
        beta_grad._tensor._storage,
        inp_grad._tensor._storage,
        out_grad._tensor._storage,
        inp._tensor._storage,
        gamma._tensor._storage,
        beta._tensor._storage,
        var._tensor._storage,
        mean._tensor._storage,
        rows,
        hidden_size,
        stream
      ) 

      return inp_grad, gamma_grad, beta_grad
      #   END ASSIGN4_2_2

    @staticmethod
    def mla_fw(q_nope: Tensor, q_rope: Tensor, c_kv: Tensor,
               k_rope: Tensor, w_up: Tensor, b_up: Optional[Tensor] = None,
               query_start: int = 0) -> Tensor:
        """Fused MLA attention context, shape [B,H,Q,Dv].

        Inputs are ALREADY projected / RoPE-applied / KV-normalized.
        The CUDA library consumes contiguous float32 HOST storage.
        This low-level backend call does NOT itself register autograd history.
        """
        if len(q_nope.shape) != 4 or len(q_rope.shape) != 4:
            raise ValueError("q_nope and q_rope must be [B,H,Q,D]")
        if len(c_kv.shape) != 3 or len(k_rope.shape) != 3:
            raise ValueError("c_kv and k_rope must be [B,K,D]")
        if len(w_up.shape) != 2:
            raise ValueError("w_up must be [H*(Dn+Dv), R]")
        B, H, Q, Dn = q_nope.shape
        B2, H2, Q2, Dr = q_rope.shape
        B3, K, R = c_kv.shape
        if (B, H, Q) != (B2, H2, Q2) or B != B3 or k_rope.shape != (B,K,Dr):
            raise ValueError("MLA input dimensions do not match")
        if w_up.shape[1] != R or w_up.shape[0] % H:
            raise ValueError("w_up must have H*(Dn+Dv) rows and R columns")
        Dv = w_up.shape[0] // H - Dn
        if Dv <= 0 or not (0 <= query_start <= K-Q):
            raise ValueError("invalid Dv or query_start")
        if b_up is not None and b_up.size != H * (Dn + Dv):
            raise ValueError("b_up has incorrect size")

        qn, qr, ck, kr, w = [t.contiguous() for t in (q_nope, q_rope, c_kv, k_rope, w_up)]
        bias = None if b_up is None else b_up.contiguous()
        out = q_nope.zeros((B,H,Q,Dv))
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        fn = lib_mla.launch_mla_forward
        fn.argtypes = [ptr,ptr,ptr,ptr,ptr,ctypes.c_void_p,ptr] + [ctypes.c_int]*9 + [ctypes.c_float, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        bias_addr = None if bias is None else ctypes.c_void_p(bias._tensor._storage.ctypes.data)
        status = fn(qn._tensor._storage, qr._tensor._storage,
                    ck._tensor._storage, kr._tensor._storage,
                    w._tensor._storage, bias_addr, out._tensor._storage,
                    B,H,Q,K,Dn,Dr,Dv,R,int(query_start),
                    float((Dn+Dr)**-0.5), None)
        if status:
            raise RuntimeError(f"launch_mla_forward failed with status {status}")
        return out

    @staticmethod
    def mla_bw(out_grad: Tensor, q_nope: Tensor, q_rope: Tensor,
               c_kv: Tensor, k_rope: Tensor, w_up: Tensor,
               b_up: Optional[Tensor] = None, query_start: int = 0):
        """Return grads (dQ_nope,dQ_rope,dC_KV,dK_rope,dW_up,dB_up).

        dB_up is None when b_up is None. Accumulates gradients for shared
        KV latents and weights across all heads/queries in CUDA via atomics.
        """
        B,H,Q,Dn = q_nope.shape
        B2,H2,Q2,Dr = q_rope.shape
        B3,K,R = c_kv.shape
        if (B,H,Q) != (B2,H2,Q2) or B != B3 or k_rope.shape != (B,K,Dr):
            raise ValueError("MLA input dimensions do not match")
        if w_up.shape[1] != R or w_up.shape[0] % H:
            raise ValueError("invalid w_up dimensions")
        Dv = w_up.shape[0] // H - Dn
        if Dv <= 0 or out_grad.shape != (B,H,Q,Dv) or not (0 <= query_start <= K-Q):
            raise ValueError("invalid upstream gradient shape or query_start")
        if b_up is not None and b_up.size != H*(Dn+Dv):
            raise ValueError("invalid b_up size")

        go, qn, qr, ck, kr, w = [t.contiguous() for t in
                                (out_grad, q_nope, q_rope, c_kv, k_rope, w_up)]
        bias = None if b_up is None else b_up.contiguous()
        gqn = q_nope.zeros(q_nope.shape)
        gqr = q_rope.zeros(q_rope.shape)
        gck = c_kv.zeros(c_kv.shape)
        gkr = k_rope.zeros(k_rope.shape)
        gw = w_up.zeros(w_up.shape)
        gb = None if b_up is None else b_up.zeros(b_up.shape)
        ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
        fn = lib_mla.launch_mla_backward
        fn.argtypes = [ptr]*6 + [ctypes.c_void_p] + [ptr]*5 + [ctypes.c_void_p] + [ctypes.c_int]*9 + [ctypes.c_float,ctypes.c_void_p]
        fn.restype = ctypes.c_int
        bias_addr = None if bias is None else ctypes.c_void_p(bias._tensor._storage.ctypes.data)
        gb_addr = None if gb is None else ctypes.c_void_p(gb._tensor._storage.ctypes.data)
        status = fn(go._tensor._storage, qn._tensor._storage,
                    qr._tensor._storage, ck._tensor._storage,
                    kr._tensor._storage, w._tensor._storage, bias_addr,
                    gqn._tensor._storage, gqr._tensor._storage,
                    gck._tensor._storage, gkr._tensor._storage,
                    gw._tensor._storage, gb_addr,
                    B,H,Q,K,Dn,Dr,Dv,R,int(query_start),
                    float((Dn+Dr)**-0.5), None)
        if status:
            raise RuntimeError(f"launch_mla_backward failed with status {status}")
        return gqn, gqr, gck, gkr, gw, gb
