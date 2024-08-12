set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

add_library(FA2_3 SHARED
  flash.cu

  flash_fwd_hdim128_fp16_sm80.cu
  flash_fwd_hdim160_fp16_sm80.cu
  flash_fwd_hdim192_fp16_sm80.cu
  flash_fwd_hdim224_fp16_sm80.cu
  flash_fwd_hdim256_fp16_sm80.cu
  flash_fwd_hdim32_fp16_sm80.cu
  flash_fwd_hdim64_fp16_sm80.cu
  flash_fwd_hdim96_fp16_sm80.cu

  flash_fwd_split_hdim128_fp16_sm80.cu
  flash_fwd_split_hdim160_fp16_sm80.cu
  flash_fwd_split_hdim192_fp16_sm80.cu
  flash_fwd_split_hdim224_fp16_sm80.cu
  flash_fwd_split_hdim256_fp16_sm80.cu
  flash_fwd_split_hdim32_fp16_sm80.cu
  flash_fwd_split_hdim64_fp16_sm80.cu
  flash_fwd_split_hdim96_fp16_sm80.cu
)

set_target_properties(FA2_3 PROPERTIES CUDA_ARCHITECTURES "80")
