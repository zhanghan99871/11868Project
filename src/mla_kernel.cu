// MLA inference-only fused CUDA attention kernel for host-backed MiniTorch.
//
// Inputs are contiguous float32 arrays in HOST memory, matching the reference
// softmax_kernel.cu launcher's host<->device ABI. The kernel itself operates on
// device allocations; there is one device kernel launch per forward call.
//
// q_nope  [B,H,Q,Dn]     non-rotary queries (already projected)
// q_rope  [B,H,Q,Dr]     rotary queries (RoPE already applied)
// c_kv    [B,K,R]        KV latent (including optional kv_norm)
// k_rope  [B,K,Dr]       shared rotary keys (RoPE already applied)
// w_up    [H*(Dn+Dv),R] torch nn.Linear.weight layout, per head [K,V]
// b_up    [H*(Dn+Dv)]   optional nn.Linear.bias (or NULL)
// out     [B,H,Q,Dv]     attention context before output projection
//
// Computes exactly the reference's reconstructed K/V MLA attention in
// float32 algebra (up to summation/reduction rounding). Fuses the key/value
// up-projection algebra, QK, causal softmax, and AV without materializing
// K/V or a [B,H,Q,K] attention matrix.
//
// NOT INCLUDED: q_down/q_up, kv_down, RoPE, kv_norm, output projection,
// backward gradients, dropout, arbitrary additive masks, fp16/bf16.

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>

namespace lightseq {
namespace cuda {
namespace {

constexpr int kThreads = 128;
struct FloatMax {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};
constexpr int kMaxKeyLen = 2048;

// Each CTA computes one (batch, head, query) row. Shared layout:
// q_latent[R] | scores[K] | pooled_latent[R]
// Use dynamic shared storage so rank and context length are runtime inputs.
__global__ void ker_mla_forward(
    const float* __restrict__ q_nope,
    const float* __restrict__ q_rope,
    const float* __restrict__ c_kv,
    const float* __restrict__ k_rope,
    const float* __restrict__ w_up,
    const float* __restrict__ b_up,
    float* __restrict__ out,
    int H, int Q, int K, int Dn, int Dr, int Dv, int R,
    int query_start, float scale) {

  const int q = blockIdx.x;
  const int h = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int per_head_up = Dn + Dv;
  const int absolute_q = query_start + q;

  extern __shared__ float workspace[];
  float* q_latent = workspace;            // R
  float* scores = q_latent + R;           // K, subsequently exp(scores - max)
  float* pooled = scores + K;             // R

  const size_t qn_base = (((size_t)b * H + h) * Q + q) * Dn;
  const size_t qr_base = (((size_t)b * H + h) * Q + q) * Dr;
  const size_t wb_base = ((size_t)h * per_head_up) * R;
  const size_t ob_base = (((size_t)b * H + h) * Q + q) * Dv;

  // Absorb W_K^T into the query (one-time per query), rather than
  // reconstructing per-token keys. For each rank r:
  // projected_q[r] = sum_d q_nope[d] * W_K[d,r]
  for (int r = tid; r < R; r += blockDim.x) {
    float s = 0.0f;
    for (int d = 0; d < Dn; ++d) {
      s += q_nope[qn_base + d] * w_up[wb_base + (size_t)d * R + r];
    }
    q_latent[r] = s;
  }
  __syncthreads();

  // K-up biases add a constant to every key for a given query;
  // softmax cancels it exactly. Thus only W_K contributes here.
  using Reduce = cub::BlockReduce<float, kThreads>;
  __shared__ typename Reduce::TempStorage reduce_storage;
  __shared__ float row_max;
  __shared__ float row_sum;

  // 1. Materialize a single row of *scores* in shared memory.
  float local_max = -CUDART_INF_F;
  for (int k = tid; k < K; k += blockDim.x) {
    float score = -CUDART_INF_F;
    if (k <= absolute_q) {
      const size_t c_base = ((size_t)b * K + k) * R;
      const size_t kr_base = ((size_t)b * K + k) * Dr;
      score = 0.0f;
      for (int r = 0; r < R; ++r) {
        score += q_latent[r] * c_kv[c_base + r];
      }
      for (int d = 0; d < Dr; ++d) {
        score += q_rope[qr_base + d] * k_rope[kr_base + d];
      }
      score *= scale;
    }
    scores[k] = score;
    local_max = fmaxf(local_max, score);
  }
  float max_val = Reduce(reduce_storage).Reduce(local_max, FloatMax{});
  if (tid == 0) row_max = max_val;
  __syncthreads();

  // 2. Numerically stable softmax. Masked keys have score=-inf and exp=0.
  float local_sum = 0.0f;
  for (int k = tid; k < K; k += blockDim.x) {
    float e = __expf(scores[k] - row_max);
    scores[k] = e;
    local_sum += e;
  }
  __syncthreads(); // reuse BlockReduce storage after the first reduction
  float sum_val = Reduce(reduce_storage).Sum(local_sum);
  if (tid == 0) row_sum = sum_val;
  __syncthreads();

  // 3. AV without reconstructing V: pool c_KV over key positions, then
  // apply W_V and bias. This relies on attention weights summing to one.
  // Each r is independent, so no atomics are needed.
  for (int r = tid; r < R; r += blockDim.x) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += scores[k] * c_kv[((size_t)b * K + k) * R + r];
    }
    pooled[r] = acc / row_sum;
  }
  __syncthreads();

  const size_t w_v_start = wb_base + (size_t)Dn * R;
  for (int d = tid; d < Dv; d += blockDim.x) {
    float acc = (b_up != nullptr) ? b_up[(size_t)h * per_head_up + Dn + d] : 0.0f;
    for (int r = 0; r < R; ++r) {
      acc += pooled[r] * w_up[w_v_start + (size_t)d * R + r];
    }
    out[ob_base + d] = acc;
  }
}

inline int check_cuda(cudaError_t err, const char* where) {
  if (err == cudaSuccess) return 0;
  std::fprintf(stderr, "MLA CUDA %s: %s\n", where, cudaGetErrorString(err));
  return static_cast<int>(err);
}

} // namespace

// C ABI suitable for ctypes. Returns 0 on success; CUDA error code otherwise.
// Host array pointers are used intentionally, as in the provided softmax API.
// The mask is implicit causal masking; query_start=K-Q for append-only cache.
extern "C" int launch_mla_forward(
    const float* q_nope, const float* q_rope,
    const float* c_kv, const float* k_rope,
    const float* w_up, const float* b_up,
    float* out,
    int B, int H, int Q, int K, int Dn, int Dr, int Dv, int R,
    int query_start, float scale, cudaStream_t stream) {

  if (!q_nope || !q_rope || !c_kv || !k_rope || !w_up || !out ||
      B <= 0 || H <= 0 || Q <= 0 || K <= 0 ||
      Dn <= 0 || Dr <= 0 || Dv <= 0 || R <= 0 ||
      K > kMaxKeyLen || query_start < 0 ||
      (long long)query_start + Q > K ||
      !std::isfinite(scale) || scale <= 0.0f) {
    std::fprintf(stderr, "launch_mla_forward: invalid dimensions/arguments\n");
    return -1;
  }
  const size_t shared_bytes = (size_t)(2 * R + K) * sizeof(float);
  int device = 0, shared_limit = 0;
  int status = check_cuda(cudaGetDevice(&device), "get device");
  if (status) return status;
  status = check_cuda(cudaDeviceGetAttribute(
      &shared_limit, cudaDevAttrMaxSharedMemoryPerBlock, device), "shared limit");
  if (status) return status;
  if (shared_bytes > static_cast<size_t>(shared_limit)) {
    std::fprintf(stderr, "launch_mla_forward: shared memory requirement %zu exceeds %d\n",
                 shared_bytes, shared_limit);
    return -2;
  }

  const size_t nq = (size_t)B * H * Q;
  const size_t nk = (size_t)B * K;
  const size_t w_len = (size_t)H * (Dn + Dv);
  const size_t qn_bytes = nq * Dn * sizeof(float);
  const size_t qr_bytes = nq * Dr * sizeof(float);
  const size_t ck_bytes = nk * R * sizeof(float);
  const size_t kr_bytes = nk * Dr * sizeof(float);
  const size_t w_bytes = w_len * R * sizeof(float);
  const size_t bias_bytes = w_len * sizeof(float);
  const size_t out_bytes = nq * Dv * sizeof(float);

  float *d_qn = nullptr, *d_qr = nullptr, *d_ck = nullptr, *d_kr = nullptr;
  float *d_w = nullptr, *d_bias = nullptr, *d_out = nullptr;
  // Failures return status after cleaning up allocated buffers.
  #define CUDA_TRY(op, label) do { \
    status = check_cuda((op), (label)); \
    if (status != 0) goto cleanup; \
  } while(0)

  CUDA_TRY(cudaMalloc((void**)&d_qn, qn_bytes), "malloc q_nope");
  CUDA_TRY(cudaMalloc((void**)&d_qr, qr_bytes), "malloc q_rope");
  CUDA_TRY(cudaMalloc((void**)&d_ck, ck_bytes), "malloc c_kv");
  CUDA_TRY(cudaMalloc((void**)&d_kr, kr_bytes), "malloc k_rope");
  CUDA_TRY(cudaMalloc((void**)&d_w, w_bytes), "malloc W_up");
  if (b_up) CUDA_TRY(cudaMalloc((void**)&d_bias, bias_bytes), "malloc b_up");
  CUDA_TRY(cudaMalloc((void**)&d_out, out_bytes), "malloc output");

  CUDA_TRY(cudaMemcpyAsync(d_qn, q_nope, qn_bytes, cudaMemcpyHostToDevice, stream), "copy q_nope");
  CUDA_TRY(cudaMemcpyAsync(d_qr, q_rope, qr_bytes, cudaMemcpyHostToDevice, stream), "copy q_rope");
  CUDA_TRY(cudaMemcpyAsync(d_ck, c_kv, ck_bytes, cudaMemcpyHostToDevice, stream), "copy c_kv");
  CUDA_TRY(cudaMemcpyAsync(d_kr, k_rope, kr_bytes, cudaMemcpyHostToDevice, stream), "copy k_rope");
  CUDA_TRY(cudaMemcpyAsync(d_w, w_up, w_bytes, cudaMemcpyHostToDevice, stream), "copy W_up");
  if (b_up) CUDA_TRY(cudaMemcpyAsync(d_bias, b_up, bias_bytes, cudaMemcpyHostToDevice, stream), "copy b_up");

  ker_mla_forward<<<dim3(Q, H, B), kThreads, shared_bytes, stream>>>(
      d_qn, d_qr, d_ck, d_kr, d_w, d_bias, d_out,
      H, Q, K, Dn, Dr, Dv, R, query_start, scale);
  CUDA_TRY(cudaGetLastError(), "launch");
  CUDA_TRY(cudaMemcpyAsync(out, d_out, out_bytes, cudaMemcpyDeviceToHost, stream), "copy result");
  CUDA_TRY(cudaStreamSynchronize(stream), "synchronize");

cleanup:
  // Ensure no active kernel/copy still references buffers before freeing.
  if (status != 0) cudaStreamSynchronize(stream);
  if (d_qn) cudaFree(d_qn);
  if (d_qr) cudaFree(d_qr);
  if (d_ck) cudaFree(d_ck);
  if (d_kr) cudaFree(d_kr);
  if (d_w) cudaFree(d_w);
  if (d_bias) cudaFree(d_bias);
  if (d_out) cudaFree(d_out);
  #undef CUDA_TRY
  return status;
}

} // namespace cuda
} // namespace lightseq

// Step 2: correctness-first backward for the fused MLA attention operation.
// Follows the forward kernel's layout and host-buffer ABI.
// One CUDA thread handles each (b,h,q) row; atomicAdd combines gradients
// shared across queries/heads. Not a performance-optimized training kernel.
namespace lightseq {
namespace cuda {
namespace {

__global__ void ker_mla_backward_reference(
    const float* __restrict__ grad_out,
    const float* __restrict__ q_nope,
    const float* __restrict__ q_rope,
    const float* __restrict__ c_kv,
    const float* __restrict__ k_rope,
    const float* __restrict__ w_up,
    const float* __restrict__ b_up,
    float* __restrict__ grad_q_nope,
    float* __restrict__ grad_q_rope,
    float* __restrict__ grad_c_kv,
    float* __restrict__ grad_k_rope,
    float* __restrict__ grad_w_up,
    float* __restrict__ grad_b_up,
    int H, int Q, int K, int Dn, int Dr, int Dv, int R,
    int query_start, float scale) {
  if (threadIdx.x != 0) return;
  int q = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int absolute_q = query_start + q;
  const int head_width = Dn + Dv;
  const size_t qn_base = (((size_t)b * H + h) * Q + q) * Dn;
  const size_t qr_base = (((size_t)b * H + h) * Q + q) * Dr;
  const size_t go_base = (((size_t)b * H + h) * Q + q) * Dv;
  const size_t wbase = (size_t)h * head_width * R;
  const size_t biasbase = (size_t)h * head_width;

  // Shared: attention probabilities and dL/dp for a single query.
  extern __shared__ float tmp[];
  float* prob = tmp;
  float* grad_prob = tmp + K;

  // Pass 1: stable softmax scores, using reconstructed K for clarity.
  float max_score = -CUDART_INF_F;
  for (int k = 0; k < K; ++k) {
    if (k > absolute_q) {
      prob[k] = -CUDART_INF_F;
      continue;
    }
    const size_t cb = ((size_t)b * K + k) * R;
    const size_t kb = ((size_t)b * K + k) * Dr;
    float score = 0.0f;
    for (int d = 0; d < Dn; ++d) {
      float key = b_up ? b_up[biasbase + d] : 0.0f;
      for (int r = 0; r < R; ++r)
        key += w_up[wbase + (size_t)d * R + r] * c_kv[cb + r];
      score += q_nope[qn_base + d] * key;
    }
    for (int d = 0; d < Dr; ++d)
      score += q_rope[qr_base + d] * k_rope[kb + d];
    score *= scale;
    prob[k] = score;
    max_score = fmaxf(max_score, score);
  }
  float denom = 0.0f;
  for (int k = 0; k < K; ++k) {
    prob[k] = k <= absolute_q ? expf(prob[k] - max_score) : 0.0f;
    denom += prob[k];
  }
  for (int k = 0; k < K; ++k) prob[k] /= denom;

  // Pass 2: dL / d(softmax probability) = dot(grad_out, V_k).
  float dot_gp = 0.0f;
  for (int k = 0; k <= absolute_q; ++k) {
    const size_t cb = ((size_t)b * K + k) * R;
    float gp = 0.0f;
    for (int d = 0; d < Dv; ++d) {
      float value = b_up ? b_up[biasbase + Dn + d] : 0.0f;
      for (int r = 0; r < R; ++r)
        value += w_up[wbase + (size_t)(Dn + d) * R + r] * c_kv[cb + r];
      gp += grad_out[go_base + d] * value;
    }
    grad_prob[k] = gp;
    dot_gp += prob[k] * gp;
  }

  // Pass 3: propagate through softmax, QK, and the value path.
  for (int k = 0; k <= absolute_q; ++k) {
    const size_t cb = ((size_t)b * K + k) * R;
    const size_t kb = ((size_t)b * K + k) * Dr;
    float gs = prob[k] * (grad_prob[k] - dot_gp) * scale;

    for (int d = 0; d < Dn; ++d) {
      float key = b_up ? b_up[biasbase + d] : 0.0f;
      for (int r = 0; r < R; ++r)
        key += w_up[wbase + (size_t)d * R + r] * c_kv[cb + r];
      atomicAdd(&grad_q_nope[qn_base + d], gs * key);
      float g_key = gs * q_nope[qn_base + d];
      for (int r = 0; r < R; ++r) {
        atomicAdd(&grad_w_up[wbase + (size_t)d * R + r], g_key * c_kv[cb + r]);
        atomicAdd(&grad_c_kv[cb + r], g_key * w_up[wbase + (size_t)d * R + r]);
      }
      // Key bias cancels in softmax (sum_k dL/dscore_k = 0).
      // Leaving its gradient exactly zero avoids cancellation noise.
    }
    for (int d = 0; d < Dr; ++d) {
      atomicAdd(&grad_q_rope[qr_base + d], gs * k_rope[kb + d]);
      atomicAdd(&grad_k_rope[kb + d], gs * q_rope[qr_base + d]);
    }
    for (int d = 0; d < Dv; ++d) {
      float gv = prob[k] * grad_out[go_base + d];
      if (grad_b_up) atomicAdd(&grad_b_up[biasbase + Dn + d], gv);
      for (int r = 0; r < R; ++r) {
        atomicAdd(&grad_w_up[wbase + (size_t)(Dn + d) * R + r], gv * c_kv[cb + r]);
        atomicAdd(&grad_c_kv[cb + r], gv * w_up[wbase + (size_t)(Dn + d) * R + r]);
      }
    }
  }
}
} // namespace

// The nine pointers after grad_out: six forward inputs + six gradients,
// where b_up/grad_b_up may be NULL. All others are required, host FP32.
extern "C" int launch_mla_backward(
    const float* grad_out,
    const float* q_nope, const float* q_rope,
    const float* c_kv, const float* k_rope,
    const float* w_up, const float* b_up,
    float* grad_q_nope, float* grad_q_rope,
    float* grad_c_kv, float* grad_k_rope,
    float* grad_w_up, float* grad_b_up,
    int B, int H, int Q, int K, int Dn, int Dr, int Dv, int R,
    int query_start, float scale, cudaStream_t stream) {
  if (!grad_out || !q_nope || !q_rope || !c_kv || !k_rope || !w_up ||
      !grad_q_nope || !grad_q_rope || !grad_c_kv || !grad_k_rope || !grad_w_up ||
      ((b_up == nullptr) != (grad_b_up == nullptr)) ||
      B <= 0 || H <= 0 || Q <= 0 || K <= 0 || Dn <= 0 || Dr <= 0 ||
      Dv <= 0 || R <= 0 || K > kMaxKeyLen || query_start < 0 ||
      (long long)query_start + Q > K || !std::isfinite(scale) || scale <= 0) {
    std::fprintf(stderr, "launch_mla_backward: invalid arguments\n");
    return -1;
  }
  const size_t nq = (size_t)B * H * Q;
  const size_t nk = (size_t)B * K;
  const size_t nwb = (size_t)H * (Dn + Dv);
  const size_t sizes[7] = {
      nq*Dv*sizeof(float), nq*Dn*sizeof(float), nq*Dr*sizeof(float),
      nk*R*sizeof(float), nk*Dr*sizeof(float), nwb*R*sizeof(float),
      nwb*sizeof(float)};
  const float* src[7] = {grad_out, q_nope, q_rope, c_kv, k_rope, w_up, b_up};
  float* dst[6] = {grad_q_nope, grad_q_rope, grad_c_kv, grad_k_rope,
                   grad_w_up, grad_b_up};
  const int grad_index[6] = {1,2,3,4,5,6};
  float* d_src[7] = {};
  float* d_dst[6] = {};
  int status = 0;
  const size_t shared_bytes = 2 * (size_t)K * sizeof(float);
  int device = 0, shared_limit = 0;
  status = check_cuda(cudaGetDevice(&device), "backward get device");
  if (status) return status;
  status = check_cuda(cudaDeviceGetAttribute(&shared_limit,
      cudaDevAttrMaxSharedMemoryPerBlock, device), "backward shared limit");
  if (status) return status;
  if (shared_bytes > (size_t)shared_limit) return -2;

  #define BW_TRY(op, label) do { \
    status = check_cuda((op), (label)); \
    if (status != 0) goto bw_cleanup; \
  } while (0)
  for (int i = 0; i < 7; ++i) {
    if (!src[i]) continue;
    BW_TRY(cudaMalloc((void**)&d_src[i], sizes[i]), "bw input alloc");
    BW_TRY(cudaMemcpyAsync(d_src[i], src[i], sizes[i],
                            cudaMemcpyHostToDevice, stream), "bw input copy");
  }
  for (int i = 0; i < 6; ++i) {
    if (!dst[i]) continue;
    BW_TRY(cudaMalloc((void**)&d_dst[i], sizes[grad_index[i]]), "bw grad alloc");
    BW_TRY(cudaMemsetAsync(d_dst[i], 0, sizes[grad_index[i]], stream), "bw grad zero");
  }
  ker_mla_backward_reference<<<dim3(Q,H,B), 1, shared_bytes, stream>>>(
      d_src[0], d_src[1], d_src[2], d_src[3], d_src[4], d_src[5], d_src[6],
      d_dst[0], d_dst[1], d_dst[2], d_dst[3], d_dst[4], d_dst[5],
      H, Q, K, Dn, Dr, Dv, R, query_start, scale);
  BW_TRY(cudaGetLastError(), "bw launch");
  for (int i = 0; i < 6; ++i) {
    if (!dst[i]) continue;
    BW_TRY(cudaMemcpyAsync(dst[i], d_dst[i], sizes[grad_index[i]],
                           cudaMemcpyDeviceToHost, stream), "bw copy gradient");
  }
  BW_TRY(cudaStreamSynchronize(stream), "bw synchronize");
bw_cleanup:
  if (status) cudaStreamSynchronize(stream);
  for (float* ptr : d_src) if (ptr) cudaFree(ptr);
  for (float* ptr : d_dst) if (ptr) cudaFree(ptr);
  #undef BW_TRY
  return status;
}
} // namespace cuda
} // namespace lightseq
