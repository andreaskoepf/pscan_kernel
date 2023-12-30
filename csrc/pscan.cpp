#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <vector>

#include "pscan.h"

template<typename input_t>
void pscan_fwd_cuda(PScanParams &params, cudaStream_t stream);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void set_pscan_params(
    PScanParams& params,
    const size_t batch,
    const size_t seqlen,
    const size_t dim,
    const torch::Tensor A, 
    const torch::Tensor X
) {
    params.batch = batch;
    params.seqlen = seqlen;
    params.dim = dim;
    params.A = A;
    params.X = X;
}


std::vector<torch::Tensor> pscan_scan_fwd(torch::Tensor A, torch::Tensor X) {
    CHECK_INPUT(A);
    CHECK_INPUT(X);

    at::cuda::CUDAGuard device_guard{(char)A.get_device()};
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    PScanParams pscan_params;

    const size_t batch = X.size(0);
    const size_t seqlen = X.size(1);
    const size_t dim = X.size(2);

    set_pscan_params(pscan_params, batch, seqlen, dim, A, X);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "pscan_fwd", [&] { 
        pscan_fwd_cuda<scalar_t>(pscan_params, stream);
    });

    return {A};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pscan_scan_fwd", &pscan_scan_fwd, "pscan scan forward");
    //m.def("bwd", &pscan_scan_bwd, "pscan scan backward");     // TBD
}

