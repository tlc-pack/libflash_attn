project(flash-attention LANGUAGES CXX CUDA)

option(CUTLASS_DIR "The CUTLASS root directory")

add_subdirectory(src)
