#include "flash.h"
#include "flash_fwd_launch_template.h"

bool flash_attention_forward(const half* q_ptr,
			     const half* k_ptr,
			     const half* v_ptr,
			     half* output_ptr,
			     int head_dim,
			     cudaStream_t stream
			     ) {
  Flash_fwd_params params;
  params.q_ptr = q_ptr;
  params.k_ptr = k_ptr;
  params.v_ptr = v_ptr;
  params.o_ptr = output_ptr;

  if (head_dim == 32) {
    run_mha_fwd_<half, 32>(params, stream);
  } // else if (head_dim == 64) {
  //   run_mha_fwd_<half, 64>(params, stream);
  // } else if (head_dim == 96) {
  //   run_mha_fwd_<half, 96>(params, stream);
  // } else  if (head_dim == 128) {
  //   run_mha_fwd_<half, 128>(params, stream);
  // } else  if (head_dim == 160) {
  //   run_mha_fwd_<half, 160>(params, stream);
  // } else  if (head_dim == 192) {
  //   run_mha_fwd_<half, 192>(params, stream);
  // } else  if (head_dim == 224) {
  //   run_mha_fwd_<half, 224>(params, stream);
  // } else  if (head_dim == 256) {
  //   run_mha_fwd_<half, 256>(params, stream);
  // } else {
  //   return false;
  // }

  return true;
}
