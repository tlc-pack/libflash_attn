#include "flash.h"
#include "flash_fwd_launch_template.h"

namespace flash_attn {


inline void _assert(bool result, const char* const file, int const line, std::string const& info = "")
{
  if (!result) {
    throw std::runtime_error(std::string("[FT][ERROR] ") + info + " Assertion fail: " + file + ":"
			     + std::to_string(line) + " \n");
  }
}

#define CHECK(val) _assert(val, __FILE__, __LINE__)

void run(Flash_fwd_params params, cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  int major, minor;
  cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(
        &minor, cudaDevAttrComputeCapabilityMinor, device);
  params.sm = major * 10 + minor;

  auto head_dim = params.d;

  if (head_dim <= 32) {
    run_mha_fwd_<half, 32>(params, stream);
  } else if (head_dim <= 64) {
    run_mha_fwd_<half, 64>(params, stream);
  } else if (head_dim <= 96) {
    run_mha_fwd_<half, 96>(params, stream);
  } else  if (head_dim <= 128) {
    run_mha_fwd_<half, 128>(params, stream);
  } else  if (head_dim <= 160) {
    run_mha_fwd_<half, 160>(params, stream);
  } else  if (head_dim <= 192) {
    run_mha_fwd_<half, 192>(params, stream);
  } else  if (head_dim <= 224) {
    run_mha_fwd_<half, 224>(params, stream);
  } else {
    run_mha_fwd_<half, 256>(params, stream);
  }
}

void flash_attention_forward(const half* q_ptr,
			     const half* k_ptr,
			     const half* v_ptr,
			     half* output_ptr,
			     int batch_size,
			     int seqlen_q,
			     int seqlen_k,
			     int num_heads,
			     int num_heads_k,
			     int head_dim,
			     int q_batch_stride,
			     int k_batch_stride,
			     int v_batch_stride,
			     int o_batch_stride,
			     int q_head_stride,
			     int k_head_stride,
			     int v_head_stride,
			     int o_head_stride,
			     int q_row_stride,
			     int k_row_stride,
			     int v_row_stride,
			     int o_row_stride,
			     float softmax_scale,
			     bool is_causal,
			     int window_size_left,
			     int window_size_right,
			     cudaStream_t stream) {
  CHECK(head_dim % 8 == 0);
  CHECK(head_dim <= 256);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_dim_rounded = round_multiple(head_dim, 32);
  const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

  Flash_fwd_params params;
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = output_ptr;
  params.is_causal = is_causal;
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

  run(params, stream);
}

void flash_attention_var_len_forward(const half *q_ptr,
				     const half *k_ptr,
                                     const half *v_ptr,
				     const int *cu_seqlens_q,
                                     const int *cu_seqlens_k,
				     half* output_ptr,
				     int batch_size,
				     int max_seqlen_q,
                                     int max_seqlen_k,
				     int num_heads,
				     int num_heads_k,
				     int head_dim,
				     int q_head_stride,
				     int k_head_stride,
				     int v_head_stride,
				     int o_head_stride,
				     int q_row_stride,
				     int k_row_stride,
				     int v_row_stride,
				     int o_row_stride,
				     float softmax_scale,
				     bool is_causal,
				     int window_size_left,
				     int window_size_right,
				     cudaStream_t stream) {
  CHECK(head_dim % 8 == 0);
  CHECK(head_dim <= 256);

  auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
  const int head_dim_rounded = round_multiple(head_dim, 32);
  const int seqlen_q_rounded = round_multiple(max_seqlen_q, 128);
  const int seqlen_k_rounded = round_multiple(max_seqlen_k, 128);

  Flash_fwd_params params;
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.cu_seqlens_q = cu_seqlens_q;
  params.cu_seqlens_k = cu_seqlens_k;
  params.o_ptr = output_ptr;
  params.is_causal = is_causal;
  params.b = batch_size;
  params.h = num_heads;
  params.h_k = num_heads_k;
  params.h_h_k_ratio = num_heads / num_heads_k;
  params.seqlen_q = max_seqlen_q;
  params.seqlen_k = max_seqlen_k;
  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d = head_dim;
  params.d_rounded = head_dim_rounded;
  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * M_LOG2E;

  params.q_head_stride = q_head_stride;
  params.q_row_stride = q_row_stride;
  params.k_head_stride = k_head_stride;
  params.k_row_stride = k_row_stride;
  params.v_head_stride = v_head_stride;
  params.v_row_stride = v_row_stride;
  params.o_head_stride = o_head_stride;
  params.o_row_stride = o_row_stride;

  params.window_size_left = window_size_left;
  params.window_size_right = window_size_right;

  run(params, stream);
}

} // namespace flash_attn