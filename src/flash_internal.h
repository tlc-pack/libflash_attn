#pragma once

struct Qkv_params {
    using index_t = int64_t;
    // The QKV matrices.
    void *__restrict__ q_ptr;
    void *__restrict__ k_ptr;
    void *__restrict__ v_ptr;

    // The stride between rows of the Q, K and V matrices.
    index_t q_batch_stride;
    index_t k_batch_stride;
    index_t v_batch_stride;
    index_t q_row_stride;
    index_t k_row_stride;
    index_t v_row_stride;
    index_t q_head_stride;
    index_t k_head_stride;
    index_t v_head_stride;

    // The number of heads.
    int h, h_k;
    // In the case of multi-query and grouped-query attention (MQA/GQA), nheads_k could be
    // different from nheads (query).
    int h_h_k_ratio; // precompute h / h_k,
};

struct Flash_fwd_params : public Qkv_params {
    // The O matrix (output).
    void * __restrict__ o_ptr;
    void * __restrict__ oaccum_ptr;

    // The stride between rows of O.
    index_t o_batch_stride;
    index_t o_row_stride;
    index_t o_head_stride;

    // The pointer to the softmax sum.
    void * __restrict__ softmax_lseaccum_ptr;

    // The dimensions.
    int b, seqlen_q, seqlen_k, d, seqlen_q_rounded, seqlen_k_rounded, d_rounded;
    int seqlen_knew = 0;
    int rotary_dim = 0;


    // The scaling factors for the kernel.
    float scale_softmax;
    float scale_softmax_log2;

    // array of length b+1 holding starting offset of each sequence.
    const int* __restrict__ cu_seqlens_q = nullptr;
    const int* __restrict__ cu_seqlens_k = nullptr;

    // If provided, the actual length of each k sequence.
    int * __restrict__ seqused_k = nullptr;

    void * __restrict__ knew_ptr = nullptr;
    void * __restrict__ vnew_ptr = nullptr;

    // The stride between rows of the Q, K and V matrices.
    index_t knew_batch_stride = 0;
    index_t vnew_batch_stride = 0;
    index_t knew_row_stride = 0;
    index_t vnew_row_stride = 0;
    index_t knew_head_stride = 0;
    index_t vnew_head_stride = 0;

    // The cos and sin matrices for rotary embedding.
    void * __restrict__ rotary_cos_ptr = nullptr;
    void * __restrict__ rotary_sin_ptr = nullptr;

    // The indices to index into the KV cache.
    int * __restrict__ cache_batch_idx = nullptr;

    // Paged KV cache
    int * __restrict__ block_table;
    index_t block_table_batch_stride;
    int page_block_size;

    // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
    // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
    bool is_seqlens_k_cumulative = true;

    bool is_rotary_interleaved = false;

    int num_splits = 0;  // For split-KV version

    void * __restrict__ alibi_slopes_ptr = nullptr;
    index_t alibi_slopes_batch_stride;

    // Local window size
    int window_size_left = -1, window_size_right = -1;

    bool is_causal;
    int sm = 80;

    /****** new class members in hopper ******/
    int total_q, total_k;
    uint32_t scale_softmax_log2_half2;
    // int* __restrict__ blockmask;
    // The dropout probability (probability of keeping an activation).
    // float p_dropout;
    // uint8_t p_dropout_in_uint8_t;

    // Scale factor of 1 / (1 - p_dropout).
    // float rp_dropout;
    // float scale_softmax_rp_dropout;

    bool is_bf16 = false;
    bool is_e4m3 = false;

    int* __restrict__ tile_count_semaphore;
};

template<typename T, int Headdim> void run_mha_fwd_(Flash_fwd_params &params, cudaStream_t stream);
template<typename T, int Headdim> void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream);
