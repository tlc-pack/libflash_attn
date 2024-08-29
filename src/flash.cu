#include <cuda.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "../hopper/flash_fwd_launch_template.h"
#include "flash.h"
#include "flash_internal.h"

namespace flash_attn {

inline void _assert(bool result, const char* const file, int const line,
                    std::string const& info = "") {
  if (!result) {
    throw std::runtime_error(std::string("[ERROR] ") + info + " Assertion fail: " + file + ":" +
                             std::to_string(line) + " \n");
  }
}

#define CHECK(val) _assert(val, __FILE__, __LINE__)

void run(Flash_fwd_params params, cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  params.sm = major * 10 + minor;

  // call method in hopper dir
  // if (params.d >= 64) {
  if (0) {
    if (params.sm >= 90) {
      if (!params.is_e4m3) {
        if (params.d == 64) {
          flash3::run_mha_fwd_<cutlass::half_t, 64>(params, stream);
        } else if (params.d == 128) {
          flash3::run_mha_fwd_<cutlass::half_t, 128>(params, stream);
        } else {
          flash3::run_mha_fwd_<cutlass::half_t, 256>(params, stream);
        }
      } else {
        if (params.d == 64) {
          flash3::run_mha_fwd_<cutlass::float_e4m3_t, 64>(params, stream);
        } else if (params.d == 128) {
          flash3::run_mha_fwd_<cutlass::float_e4m3_t, 128>(params, stream);
        } else if (params.d == 256) {
          flash3::run_mha_fwd_<cutlass::float_e4m3_t, 256>(params, stream);
        }
      }
      return;
    }
  } else {
    auto head_dim = params.d;
    if (head_dim <= 32) {
      run_mha_fwd_<half, 32>(params, stream);
    } else if (head_dim <= 64) {
      run_mha_fwd_<half, 64>(params, stream);
    } else if (head_dim <= 96) {
      run_mha_fwd_<half, 96>(params, stream);
    } else if (head_dim <= 128) {
      run_mha_fwd_<half, 128>(params, stream);
    } else if (head_dim <= 160) {
      run_mha_fwd_<half, 160>(params, stream);
    } else if (head_dim <= 192) {
      run_mha_fwd_<half, 192>(params, stream);
    } else if (head_dim <= 224) {
      run_mha_fwd_<half, 224>(params, stream);
    } else {
      run_mha_fwd_<half, 256>(params, stream);
    }
  }
}

Flash_fwd_params get_fwd_params(half* q_ptr, half* k_ptr, half* v_ptr, half* output_ptr,
                                int batch_size, int seqlen_q, int seqlen_k, int num_heads,
                                int num_heads_k, int head_dim, int q_batch_stride,
                                int k_batch_stride, int v_batch_stride, int o_batch_stride,
                                int q_head_stride, int k_head_stride, int v_head_stride,
                                int o_head_stride, int q_row_stride, int k_row_stride,
                                int v_row_stride, int o_row_stride, float softmax_scale,
                                bool is_causal, int window_size_left, int window_size_right) {
  CHECK(head_dim % 8 == 0);
  CHECK(head_dim <= 256);

  if (window_size_left >= seqlen_k) {
    window_size_left = -1;
  }
  if (window_size_right >= seqlen_k) {
    window_size_right = -1;
  }
  // causal=true is the same as causal=false in this case
  if (seqlen_q == 1) {
    is_causal = false;
  }
  if (is_causal) {
    window_size_right = 0;
  }

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_dim_rounded = round_multiple(head_dim, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = output_ptr;
  // Causal is the special case where window_size_right == 0 and window_size_left < 0.
  // Local is the more general case where window_size_right >= 0 or window_size_left >= 0.
  params.is_causal = window_size_left < 0 && window_size_right == 0;
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads / num_heads_k;
  params.seqlen_q = seqlen_q;
  params.seqlen_k = seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_dim;
  params.d_rounded = head_dim_rounded;
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;
  __half scale_softmax_log2_half = __float2half(params.scale_softmax_log2);
  __half2 scale_softmax_log2_half2 = __half2(scale_softmax_log2_half, scale_softmax_log2_half);
  params.scale_softmax_log2_half2 = reinterpret_cast<uint32_t&>(scale_softmax_log2_half2);

  params.q_batch_stride = q_batch_stride;
  params.q_head_stride = q_head_stride;
  params.q_row_stride = q_row_stride;
  params.k_batch_stride = k_batch_stride;
  params.k_head_stride = k_head_stride;
  params.k_row_stride = k_row_stride;
  params.v_batch_stride = v_batch_stride;
  params.v_head_stride = v_head_stride;
  params.v_row_stride = v_row_stride;
  params.o_batch_stride = o_batch_stride;
  params.o_head_stride = o_head_stride;
  params.o_row_stride = o_row_stride;

  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  params.tile_count_semaphore = new int;
  if (is_causal) {
    *(params.tile_count_semaphore) = 0;
  }

  return params;
}

void flash_attention_forward(half* q_ptr, half* k_ptr, half* v_ptr, half* output_ptr,
                             int batch_size, int seqlen_q, int seqlen_k, int num_heads,
                             int num_heads_k, int head_dim, int q_batch_stride, int k_batch_stride,
                             int v_batch_stride, int o_batch_stride, int q_head_stride,
                             int k_head_stride, int v_head_stride, int o_head_stride,
                             int q_row_stride, int k_row_stride, int v_row_stride, int o_row_stride,
                             float softmax_scale, bool is_causal, int window_size_left,
                             int window_size_right, cudaStream_t stream) {
  auto params = get_fwd_params(
      q_ptr, k_ptr, v_ptr, output_ptr, batch_size, seqlen_q, seqlen_k, num_heads, num_heads_k,
      head_dim, q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride, q_head_stride,
      k_head_stride, v_head_stride, o_head_stride, q_row_stride, k_row_stride, v_row_stride,
      o_row_stride, softmax_scale, is_causal, window_size_left, window_size_right);
  run(params, stream);
}

void flash_attention_var_len_forward(half* q_ptr, half* k_ptr, half* v_ptr, const int* cu_seqlens_q,
                                     const int* cu_seqlens_k, half* output_ptr, int batch_size,
                                     int max_seqlen_q, int max_seqlen_k, int num_heads,
                                     int num_heads_k, int head_dim, int q_head_stride,
                                     int k_head_stride, int v_head_stride, int o_head_stride,
                                     int q_row_stride, int k_row_stride, int v_row_stride,
                                     int o_row_stride, float softmax_scale, bool is_causal,
                                     int total_q, int total_k, int window_size_left,
                                     int window_size_right, cudaStream_t stream) {
  auto params = get_fwd_params(
      q_ptr, k_ptr, v_ptr, output_ptr, batch_size, max_seqlen_q, max_seqlen_k, num_heads,
      num_heads_k, head_dim, 0, 0, 0, 0,  // batch strides
      q_head_stride, k_head_stride, v_head_stride, o_head_stride, q_row_stride, k_row_stride,
      v_row_stride, o_row_stride, softmax_scale, is_causal, window_size_left, window_size_right);

  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.is_seqlens_k_cumulative = true;
  params.total_q = total_q;
  params.total_k = total_k;

  run(params, stream);
  delete params.tile_count_semaphore;
}

inline int num_splits_heuristic(int batch_nheads_mblocks, int num_SMs, int num_n_blocks,
                                int max_splits) {
  // If we have enough to almost fill the SMs, then just use 1 split
  if (batch_nheads_mblocks >= 0.8f * num_SMs) {
    return 1;
  }
  max_splits = std::min({max_splits, num_SMs, num_n_blocks});
  float max_efficiency = 0.f;
  std::vector<float> efficiency;
  efficiency.reserve(max_splits);
  auto ceildiv = [](int a, int b) { return (a + b - 1) / b; };
  // Some splits are not eligible. For example, if we have 64 blocks and choose 11 splits,
  // we'll have 6 * 10 + 4 blocks. If we choose 12 splits, we'll have 6 * 11 + (-2) blocks
  // (i.e. it's 11 splits anyway).
  // So we check if the number of blocks per split is the same as the previous num_splits.
  auto is_split_eligible = [&ceildiv, &num_n_blocks](int num_splits) {
    return num_splits == 1 ||
           ceildiv(num_n_blocks, num_splits) != ceildiv(num_n_blocks, num_splits - 1);
  };
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      efficiency.push_back(0.f);
    } else {
      float n_waves = float(batch_nheads_mblocks * num_splits) / num_SMs;
      float eff = n_waves / ceil(n_waves);
      // printf("num_splits = %d, eff = %f\n", num_splits, eff);
      if (eff > max_efficiency) {
        max_efficiency = eff;
      }
      efficiency.push_back(eff);
    }
  }
  for (int num_splits = 1; num_splits <= max_splits; num_splits++) {
    if (!is_split_eligible(num_splits)) {
      continue;
    }
    if (efficiency[num_splits - 1] >= 0.85 * max_efficiency) {
      // printf("num_splits chosen = %d\n", num_splits);
      return num_splits;
    }
  }
  return 1;
}

void run_splitkv(Flash_fwd_params params, cudaStream_t stream) {
  auto head_dim = params.d;

  if (head_dim <= 32) {
    run_mha_fwd_splitkv_dispatch<half, 32>(params, stream);
  } else if (head_dim <= 64) {
    run_mha_fwd_splitkv_dispatch<half, 64>(params, stream);
  } else if (head_dim <= 96) {
    run_mha_fwd_splitkv_dispatch<half, 96>(params, stream);
  } else if (head_dim <= 128) {
    run_mha_fwd_splitkv_dispatch<half, 128>(params, stream);
  } else if (head_dim <= 160) {
    run_mha_fwd_splitkv_dispatch<half, 160>(params, stream);
  } else if (head_dim <= 192) {
    run_mha_fwd_splitkv_dispatch<half, 192>(params, stream);
  } else if (head_dim <= 224) {
    run_mha_fwd_splitkv_dispatch<half, 224>(params, stream);
  } else {
    run_mha_fwd_splitkv_dispatch<half, 256>(params, stream);
  }
}

void set_splitkv_params(Flash_fwd_params& params, int32_t* block_table_ptr, int32_t* seqlens_k_ptr,
                        float* softmax_lse_accum_ptr, float* output_accum_ptr, int batch_size,
                        int seqlen_q, int seqlen_k, int num_heads, int head_dim, int num_blocks,
                        int block_size, int max_num_blocks_per_seq, int block_table_batch_stride,
                        int num_splits) {
  params.rotary_dim = 0;
  params.block_table = block_table_ptr;
  params.page_block_size = block_size;
  params.block_table_batch_stride = block_table_batch_stride;
  params.cu_seqlens_k = seqlens_k_ptr;
  params.is_seqlens_k_cumulative = false;

  int device;
  cudaGetDevice(&device);
  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  params.sm = major * 10 + minor;

  // This needs to match with run_mha_fwd_splitkv_dispatch
  const int block_n = head_dim <= 64 ? 256 : (head_dim <= 128 ? 128 : 64);
  const int num_n_blocks = (seqlen_k + block_n - 1) / block_n;

  // Technically kBlockM = 64 only for the splitKV kernels, not the standard kernel.
  // In any case we don't expect seqlen_q to be larger than 64 for inference.
  const int num_m_blocks = (seqlen_q + 64 - 1) / 64;
  params.num_splits = num_splits;

  if (num_splits < 1) {
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
    params.num_splits =
        num_splits_heuristic(batch_size * num_heads * num_m_blocks, sm_count, num_n_blocks, 128);
  }
  CHECK(params.num_splits <= 128);

  params.softmax_lseaccum_ptr = softmax_lse_accum_ptr;
  params.oaccum_ptr = output_accum_ptr;
}

void flash_attention_splitkv_paged_forward(
    half* q_ptr, half* kcache_ptr, half* vcache_ptr, int32_t* block_table_ptr,
    int32_t* seqlens_k_ptr, float* softmax_lse_accum_ptr, float* output_accum_ptr, half* output_ptr,
    int batch_size, int seqlen_q, int num_heads, int num_heads_k, int head_dim, int q_batch_stride,
    int k_batch_stride, int v_batch_stride, int o_batch_stride, int q_head_stride,
    int k_head_stride, int v_head_stride, int o_head_stride, int q_row_stride, int k_row_stride,
    int v_row_stride, int o_row_stride, int num_blocks, int block_size, int max_num_blocks_per_seq,
    int block_table_batch_stride, float softmax_scale, bool is_causal, int window_size_left,
    int window_size_right, int num_splits, cudaStream_t stream) {
  const int seqlen_k = max_num_blocks_per_seq * block_size;

  auto params = get_fwd_params(
      q_ptr, kcache_ptr, vcache_ptr, output_ptr, batch_size, seqlen_q, seqlen_k, num_heads,
      num_heads_k, head_dim, q_batch_stride, k_batch_stride, v_batch_stride, o_batch_stride,
      q_head_stride, k_head_stride, v_head_stride, o_head_stride, q_row_stride, k_row_stride,
      v_row_stride, o_row_stride, softmax_scale, is_causal, window_size_left, window_size_right);

  set_splitkv_params(params, block_table_ptr, seqlens_k_ptr, softmax_lse_accum_ptr,
                     output_accum_ptr, batch_size, seqlen_q, seqlen_k, num_heads, head_dim,
                     num_blocks, block_size, max_num_blocks_per_seq, block_table_batch_stride,
                     num_splits);

  run_splitkv(params, stream);
}

}  // namespace flash_attn