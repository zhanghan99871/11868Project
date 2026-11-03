#include <math.h>

#include <cub/block/block_load.cuh>
#include <cub/cub.cuh>

#include "includes/block_reduce.h"
#include "includes/kernels.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;
const float EPSILON = 1e-8f;

namespace lightseq {
namespace cuda {
/**
 * Masked softmax: out[b,h,q,:] = softmax(inp[b,h,q,:] + mask[b,h,q,:]).
 * Mask is contiguous, 4D, and may broadcast in each dimension.
 * Both input and mask are read-only; output has separate storage.
 */
__device__ __forceinline__ size_t mask_index(
    int b, int h, int q, int k,
    int mb, int mh, int mq, int mk) {
  int bb = (mb == 1 ? 0 : b);
  int hh = (mh == 1 ? 0 : h);
  int qq = (mq == 1 ? 0 : q);
  int kk = (mk == 1 ? 0 : k);
  return (((size_t)bb * mh + hh) * mq + qq) * mk + kk;
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax_lt32(const T *inp, const T *attn_mask, T *out,
                                     int from_len, int to_len, int mb, int mh, int mq, int mk) {
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;


  size_t base = (size_t)flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  inp += base;
  out += base;
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    // use fmaxf() to compute max
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        const int key_id = ele_per_thread * threadIdx.x + j;
        const int query_id = token_id + i;
        float temp_val = REDUCE_FLOAT_INF_NEG;
        if (key_id < to_len && query_id < from_len) {
          temp_val = (float)inp_val[i][j];
          if (attn_mask != nullptr) {
            temp_val += (float)attn_mask[mask_index(
                batch_id, head_id, query_id, key_id, mb, mh, mq, mk)];
          }
        }
        val[i][j] = temp_val;
        l_max[i] = fmaxf(l_max[i], temp_val);
      }
    }
    // warp reduce max
    warpReduce<ReduceType::kMax, token_per_reduce>(l_max);

    /* step 2. compute sum */
    // thread local sum
    // use __expf() to compute exp
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - l_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // warp reduce sum
    warpReduce<ReduceType::kSum, token_per_reduce>(l_sum);

    /* step 3. compute final result */
    // use __fdividef() to compute division
    // use BlockStore to store the result
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      l_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * l_sum[i]);
      }
      BlockStore(ts_store).Store(out + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
  }  // blockIdx.x
}

template <typename T, int block_dim, int ele_per_thread>
__global__ void ker_attn_softmax(const T *inp, const T *attn_mask, T *out,
                                     int from_len, int to_len, int mb, int mh, int mq, int mk) {
  
  int batch_id = blockIdx.y;
  int head_id = blockIdx.z;
  const int nhead = gridDim.z;
  const int token_per_reduce = 1;
  typedef cub::BlockLoad<T, block_dim, ele_per_thread,
                         cub::BLOCK_LOAD_VECTORIZE>
      BlockLoad;
  __shared__ typename BlockLoad::TempStorage ts_load;
  typedef cub::BlockStore<T, block_dim, ele_per_thread,
                          cub::BLOCK_STORE_VECTORIZE>
      BlockStore;
  __shared__ typename BlockStore::TempStorage ts_store;


  size_t base = (size_t)flat_3dim(batch_id, head_id, 0, nhead, from_len * to_len);
  inp += base;
  out += base;
  for (int token_id = blockIdx.x * token_per_reduce; token_id < from_len;
       token_id += gridDim.x * token_per_reduce) {
    T inp_val[token_per_reduce][ele_per_thread];
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      BlockLoad(ts_load).Load(inp + (token_id + i) * to_len, inp_val[i], to_len,
                              REDUCE_FLOAT_INF_NEG);
    }

    /* step 1. compute max */
    // thread local max
    // BEGIN ASSIGN4_1_1
    float val[token_per_reduce][ele_per_thread];
    float l_max[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_max[i] = REDUCE_FLOAT_INF_NEG;
      for (int j = 0; j < ele_per_thread; j++) {
        const int key_id = ele_per_thread * threadIdx.x + j;
        const int query_id = token_id + i;
        float temp_val = REDUCE_FLOAT_INF_NEG;
        if (key_id < to_len && query_id < from_len) {
          temp_val = (float)inp_val[i][j];
          if (attn_mask != nullptr) {
            temp_val += (float)attn_mask[mask_index(
                batch_id, head_id, query_id, key_id, mb, mh, mq, mk)];
          }
        }
        val[i][j] = temp_val;
        l_max[i] = fmaxf(l_max[i], temp_val);
      }
    }
    // END ASSIGN4_1_1
    // block reduce max
    blockReduce<ReduceType::kMax, token_per_reduce>(l_max);
    // write shared
    __shared__ float s_max[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_max[i] = l_max[i];
      }
    }
    __syncthreads();

    /* step 2. compute sum */
    // thread local sum
    // BEGIN ASSIGN4_1_1
    float l_sum[token_per_reduce];
    for (int i = 0; i < token_per_reduce; i++) {
      l_sum[i] = 0.f;
      for (int j = 0; j < ele_per_thread; j++) {
        val[i][j] = __expf(val[i][j] - s_max[i]);
        l_sum[i] += val[i][j];
      }
    }
    // END ASSIGN4_1_1
    // block reduce sum
    blockReduce<ReduceType::kSum, token_per_reduce>(l_sum);
    // write shared
    __shared__ float s_sum[token_per_reduce];
    if (threadIdx.x == 0) {
      for (int i = 0; i < token_per_reduce; i++) {
        s_sum[i] = __fdividef(1.0f, l_sum[i] + EPSILON);
      }
    }
    __syncthreads();

    /* step 3. compute final result */
    // BEGIN ASSIGN4_1_1
    for (int i = 0; i < token_per_reduce && (token_id + i) < from_len; i++) {
      for (int j = 0; j < ele_per_thread; j++) {
        inp_val[i][j] = (T)(val[i][j] * s_sum[i]);
      }
      BlockStore(ts_store).Store(out + (token_id + i) * to_len, inp_val[i],
                                 to_len);
    }
    // END ASSIGN4_1_1
  }  // blockIdx.x
}

// Host-backed MiniTorch storage: copy to GPU, execute one fused kernel, copy out.
extern "C" void launch_attn_softmax(
    const float *inp, const float *attn_mask, float *out,
    int batch_size, int nhead, int from_len, int to_len,
    int mb, int mh, int mq, int mk, cudaStream_t stream) {
  if (batch_size <= 0 || nhead <= 0 || from_len <= 0 ||
      to_len <= 0 || to_len > 1024) {
    fprintf(stderr, "launch_attn_softmax: unsupported dimensions\n");
    return;
  }
  auto cuda_check = [](cudaError_t e, const char *where) {
    if (e != cudaSuccess) {
      fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
      exit(EXIT_FAILURE);
    }
  };
  const size_t count = (size_t)batch_size * nhead * from_len * to_len;
  const size_t bytes = count * sizeof(float);
  const size_t mask_bytes = (size_t)mb * mh * mq * mk * sizeof(float);
  float *d_inp = nullptr, *d_mask = nullptr, *d_out = nullptr;
  cuda_check(cudaMalloc((void **)&d_inp, bytes), "malloc inp");
  cuda_check(cudaMalloc((void **)&d_out, bytes), "malloc out");
  cuda_check(cudaMemcpy(d_inp, inp, bytes, cudaMemcpyHostToDevice), "copy inp");
  if (attn_mask != nullptr) {
    cuda_check(cudaMalloc((void **)&d_mask, mask_bytes), "malloc mask");
    cuda_check(cudaMemcpy(d_mask, attn_mask, mask_bytes,
                          cudaMemcpyHostToDevice), "copy mask");
  }
  dim3 grid_dim(1, batch_size, nhead);
  #define ARGS d_inp, d_mask, d_out, from_len, to_len, mb, mh, mq, mk
  if (to_len <= 32) {
    ker_attn_softmax_lt32<float, 32, 1><<<grid_dim, 32, 0, stream>>>(ARGS);
  } else if (to_len <= 64) {
    ker_attn_softmax_lt32<float, 32, 2><<<grid_dim, 32, 0, stream>>>(ARGS);
  } else if (to_len <= 128) {
    grid_dim.x = 16;
    ker_attn_softmax<float, 64, 2><<<grid_dim, 64, 0, stream>>>(ARGS);
  } else if (to_len <= 256) {
    grid_dim.x = 32;
    ker_attn_softmax<float, 128, 2><<<grid_dim, 128, 0, stream>>>(ARGS);
  } else if (to_len <= 512) {
    grid_dim.x = 64;
    ker_attn_softmax<float, 256, 2><<<grid_dim, 256, 0, stream>>>(ARGS);
  } else {
    grid_dim.x = 128;
    ker_attn_softmax<float, 512, 2><<<grid_dim, 512, 0, stream>>>(ARGS);
  }
  #undef ARGS
  cuda_check(cudaGetLastError(), "softmax launch");
  cuda_check(cudaStreamSynchronize(stream), "softmax synchronize");
  cuda_check(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost), "copy out");
  cudaFree(d_inp);
  if (d_mask) cudaFree(d_mask);
  cudaFree(d_out);
}

/**
@brief: ker_attn_softmax_bw
Softmax backward in self attention.

@thread
gridDim.x = batch_size * nhead * seq_len / warps_per_block
blockDim.x = WARP_SIZE
blockDim.y = warps_per_block

@param
grad: [batch_size, nhead, seq_len, seq_len], output grad.
inp: [batch_size, nhead, seq_len, seq_len], **output** of softmax forward.
*/
template <typename T, int ITERATIONS>
__global__ void ker_attn_softmax_bw(T *grad, const T *inp, int rows, int softmax_length) {
  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  if (batch_idx >= rows) return;
  int offset = batch_idx * softmax_length + threadIdx.x;

  grad += offset;
  inp += offset;

  T grad_reg[ITERATIONS];
  T inp_reg[ITERATIONS];
  float sum = 0.0;

  #pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length) {
      grad_reg[i] = grad[i * WARP_SIZE];
      inp_reg[i] = inp[i * WARP_SIZE];
      sum += (float)grad_reg[i] * (float)inp_reg[i];
    }
  }

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<WARP_SIZE> g = cg::tiled_partition<WARP_SIZE>(b);

  for (int i = 1; i < WARP_SIZE; i <<= 1) sum += g.shfl_xor(sum, i);

  #pragma unroll
  for (int i = 0; i < ITERATIONS; ++i) {
    int curr_idx = threadIdx.x + i * WARP_SIZE;
    if (curr_idx < softmax_length)
      grad[i * WARP_SIZE] = (T)((float)inp_reg[i] * ((float)grad_reg[i] - sum));
  }
}

// template <typename T>
extern "C" {
void launch_attn_softmax_bw(const float *out_grad,
                                const float *soft_outp, float *grad_input, int rows,
                                int softmax_len,
                                cudaStream_t stream) {
  
  const int warps_per_block = 4;
  dim3 grid_dim((rows + warps_per_block - 1) / warps_per_block);
  dim3 block_dim(WARP_SIZE, warps_per_block);
  // BEGIN ASSIGN4_1_2
  
  float *d_out_grad; 
  float *d_soft_outp; 
  int float_size = sizeof(float);
  int out_grad_size = rows * softmax_len * float_size; 
  cudaMalloc((void**)&d_out_grad, out_grad_size); 
  cudaMalloc((void**)&d_soft_outp, out_grad_size); 

  cudaMemcpy(d_out_grad, out_grad, out_grad_size, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_soft_outp, soft_outp, out_grad_size, cudaMemcpyHostToDevice); 
  // Launch kernel
  // Hint: use ker_attn_softmax_bw<float, ITERATIONS> depending on softmax_len
  if (softmax_len <= 32) {
    ker_attn_softmax_bw<float, 1><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 64) {
    ker_attn_softmax_bw<float, 2><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 128) {
    ker_attn_softmax_bw<float, 4><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 256) {
    ker_attn_softmax_bw<float, 8><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 512) {
    ker_attn_softmax_bw<float, 16><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 1024) {
    ker_attn_softmax_bw<float, 32><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  } else if (softmax_len <= 2048) {
    ker_attn_softmax_bw<float, 64><<<grid_dim, block_dim, 0, stream>>>(d_out_grad, d_soft_outp, rows, softmax_len); 
  }
  else {
    throw std::runtime_error(
        "Sequence length greater than 2048 is currently not supported");
  }
  // Copy back to the host
  
  cudaStreamSynchronize(stream);
  cudaMemcpy(grad_input, d_out_grad, out_grad_size, cudaMemcpyDeviceToHost);
  // Free memory on device
  cudaFree(d_out_grad); 
  cudaFree(d_soft_outp); 
  // END ASSIGN4_1_2

}}

}  
} 
