// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.

#include "flash_fwd_launch_template.h"

namespace flash3 {
template <>
void run_mha_fwd_<cutlass::float_e4m3_t, 64>(Flash_fwd_params& params, cudaStream_t stream) {
  printf("run_mha_fwd_hdim64_fp8\n");
  run_mha_fwd_hdim64_fp8<cutlass::float_e4m3_t>(params, stream);
}
}  // namespace flash3
