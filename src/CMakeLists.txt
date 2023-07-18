set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 8 \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

add_library(flash_attn SHARED fpA_intB_gemm.cu cutlass_heuristic.cc cutlass_preprocessors.cc)
