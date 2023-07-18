set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 1 \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

include_directories(${CUTLASS_DIR}/include)

add_library(flash_attn SHARED flash_fwd_hdim128_fp16_sm80.cu)
