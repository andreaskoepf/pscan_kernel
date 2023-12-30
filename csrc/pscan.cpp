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

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float)  {                                   \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

// #define DISPATCH_WTYPE_FLOAT_AND_HALF_AND_BF16(WTYPE, NAME, ...)                     \
//     if (WTYPE == at::ScalarType::Half) {                                             \
//         using weight_t = at::Half;                                                   \
//         __VA_ARGS__();                                                               \
//     } else if (WTYPE == at::ScalarType::BFloat16) {                                  \
//         using weight_t = at::BFloat16;                                               \
//         __VA_ARGS__();                                                               \
//     } else if (WTYPE == at::ScalarType::Float)  {                                    \
//         using weight_t = float;                                                      \
//         __VA_ARGS__();                                                               \
//     } else {                                                                         \
//         AT_ERROR(#NAME, " not implemented for weight type '", toString(WTYPE), "'"); \
//     }


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

    //std::cout << "hello" << std::endl;

    const size_t batch = A.size(0);
    const size_t seqlen = A.size(1);
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

