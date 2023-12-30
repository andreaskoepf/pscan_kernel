#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "pscan_fwd_kernel.cuh"

template void pscan_fwd_cuda<float>(PScanParams &params, cudaStream_t stream);
template void pscan_fwd_cuda<double>(PScanParams &params, cudaStream_t stream);
