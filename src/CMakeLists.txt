set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

include_directories(${CUTLASS_DIR}/include)
include_directories(../include)

add_library(flash_attn SHARED
  flash.cu
  flash_fwd_hdim128_fp16_sm80.cu
  flash_fwd_hdim160_fp16_sm80.cu
  flash_fwd_hdim192_fp16_sm80.cu
  flash_fwd_hdim224_fp16_sm80.cu
  flash_fwd_hdim256_fp16_sm80.cu
  flash_fwd_hdim32_fp16_sm80.cu
  flash_fwd_hdim64_fp16_sm80.cu
  flash_fwd_hdim96_fp16_sm80.cu
)

set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "80")
