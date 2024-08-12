set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_90a,code=\\\"sm_90a,compute_90a\\\" \
                      ")
                    

add_library(flash_attn SHARED
  flash.cu
  flash_fwd_hdim64_fp16_sm90.cu
  flash_fwd_hdim64_bf16_sm90.cu
  flash_fwd_hdim128_fp16_sm90.cu
  flash_fwd_hdim128_bf16_sm90.cu
  flash_fwd_hdim256_fp16_sm90.cu
  flash_fwd_hdim256_bf16_sm90.cu
  flash_bwd_hdim64_fp16_sm90.cu
  flash_bwd_hdim96_fp16_sm90.cu
  flash_bwd_hdim128_fp16_sm90.cu
  flash_bwd_hdim64_bf16_sm90.cu
  flash_bwd_hdim96_bf16_sm90.cu
  flash_bwd_hdim128_bf16_sm90.cu
  flash_fwd_hdim64_e4m3_sm90.cu
  flash_fwd_hdim128_e4m3_sm90.cu
  flash_fwd_hdim256_e4m3_sm90.cu
)

set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "90a")
