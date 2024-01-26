/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cuda_runtime.h>
#include "cutlass/numeric_types.h"
#include "cutlass/half.h"

namespace flash_attn {

using half = cutlass::half_t;

void flash_attention_forward(half* q_ptr,
			     half* k_ptr,
			     half* v_ptr,
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
			     int window_size_left = -1,
			     int window_size_right = -1,
			     cudaStream_t stream = nullptr);


void flash_attention_var_len_forward(half *q_ptr,
				     half *k_ptr,
                                     half *v_ptr,
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
				     int window_size_left = -1,
				     int window_size_right = -1,
				     cudaStream_t stream = nullptr);


void flash_attention_splitkv_paged_forward(half* q_ptr,
			                   half* kcache_ptr,
                       			   half* vcache_ptr,
					   int32_t* block_table_ptr,
					   int32_t* seqlens_k_ptr,
					   float* softmax_lse_accum_ptr,
					   float* output_accum_ptr,
                          	           half* output_ptr,
                      			   int batch_size,
                      			   int seqlen_q,
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
					   int num_blocks,
                                           int block_size,
					   int max_num_blocks_per_seq,
					   int block_table_batch_stride,
                    			   float softmax_scale,
                    			   bool is_causal,
                    			   int window_size_left = -1,
                    			   int window_size_right = -1,
					   int num_splits = 0,
                    			   cudaStream_t stream = nullptr);

} // namespace flash_attn
