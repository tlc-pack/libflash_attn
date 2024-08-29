cmake_minimum_required(VERSION 3.12)

project(flash-attention LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 \
                      -gencode=arch=compute_90a,code=\\\"sm_90a,compute_90a\\\" \
                      ")

# ################ CUDA ################
find_package(CUDAToolkit REQUIRED)

include_directories("./cutlass/include")
include_directories("./include")
include_directories(${CUDAToolkit_INCLUDE_DIRS}) 


add_library(flash_attn SHARED
    src/flash.cu
  
    src/flash_fwd_hdim128_fp16_sm80.cu
    src/flash_fwd_hdim160_fp16_sm80.cu
    src/flash_fwd_hdim192_fp16_sm80.cu
    src/flash_fwd_hdim224_fp16_sm80.cu
    src/flash_fwd_hdim256_fp16_sm80.cu
    src/flash_fwd_hdim32_fp16_sm80.cu
    src/flash_fwd_hdim64_fp16_sm80.cu
    src/flash_fwd_hdim96_fp16_sm80.cu
    src/flash_fwd_split_hdim128_fp16_sm80.cu
    src/flash_fwd_split_hdim160_fp16_sm80.cu
    src/flash_fwd_split_hdim192_fp16_sm80.cu
    src/flash_fwd_split_hdim224_fp16_sm80.cu
    src/flash_fwd_split_hdim256_fp16_sm80.cu
    src/flash_fwd_split_hdim32_fp16_sm80.cu
    src/flash_fwd_split_hdim64_fp16_sm80.cu
    src/flash_fwd_split_hdim96_fp16_sm80.cu

    hopper/flash_fwd_hdim64_fp16_sm90.cu
    hopper/flash_fwd_hdim128_fp16_sm90.cu
    hopper/flash_fwd_hdim256_fp16_sm90.cu
    hopper/flash_fwd_hdim64_e4m3_sm90.cu
    hopper/flash_fwd_hdim128_e4m3_sm90.cu
    hopper/flash_fwd_hdim256_e4m3_sm90.cu
  )

set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "90a")