cmake_minimum_required(VERSION 3.12)

project(flash-attention LANGUAGES CXX CUDA)

find_program(GPU_dev_info nvidia-smi)
if(NOT GPU_dev_info)
    message(FATAL_ERROR "GPU driver not found.")
endif()

find_program(NVCC nvcc)
if(NOT NVCC)
    message(FATAL_ERROR "NVCC not found. Please make sure CUDA Toolkit is installed.")
endif()

execute_process(COMMAND ${GPU_dev_info} OUTPUT_VARIABLE GPU_dev_version)
string(REGEX MATCH "H100" GPU_H100 ${GPU_dev_version})

execute_process(COMMAND ${NVCC} --version OUTPUT_VARIABLE NVCC_VERSION_OUTPUT)
string(REGEX MATCH "([0-9]+\\.[0-9]+)" NVCC_VERSION ${NVCC_VERSION_OUTPUT})

if(NOT NVCC_VERSION)
    message(FATAL_ERROR "Failed to determine NVCC version.")
endif()

if(GPU_H100 AND NVCC_VERSION GREATER_EQUAL 12)
    set(compute_compacity "90a")
else()
    set(compute_compacity "80")
endif()

message("compute_compacity: ${compute_compacity}")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --use_fast_math -t 8 \
                    -gencode=arch=compute_${compute_compacity},code=\\\"sm_${compute_compacity},compute_${compute_compacity}\\\" \
                    ")



# ################ CUDA ################
find_package(CUDAToolkit REQUIRED)

include_directories("./cutlass/include")
include_directories("./include")
include_directories(${CUDAToolkit_INCLUDE_DIRS}) 


if(compute_compacity STREQUAL "90a")
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
else()
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
  )
endif()

set_target_properties(flash_attn PROPERTIES CUDA_ARCHITECTURES "${compute_compacity}")
