#pragma once

#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/CUDAContext.h>
#include "pscan.h"


inline int _cdiv(int x, int y) {
    return (x + y - 1) / y;
}


inline int nextPowerOfTwo(int x) {
	int power = 1;
	while (power < x) {
		power *= 2;
	}
	return power;
}


template<typename scalar_t>
__global__ void pscan_fwd_kernel(
    scalar_t* A,
    scalar_t* X,
    int strideXN,
    int strideXT,
    int strideXD,
    int strideAN,
    int strideAT,
    int batch,
    int seqlen,
    int dim,
    int powerOfTwo,
    int dimChunk
) {
    // allocated on invocation 
    extern __shared__ float temp[];

    float* dimBase = temp + powerOfTwo;

    const int batchId = blockIdx.x / (dim/dimChunk);
    const int dimOffset = (blockIdx.x % (dim/dimChunk)) * dimChunk;
    const int tid = threadIdx.x;

    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    if (tid < seqlen) {
        temp[ai] = A[batchId * strideAN + ai * strideAT];
        temp[bi] = A[batchId * strideAN + bi * strideAT];

        for (int i=0; i<dimChunk; ++i) {
            dimBase[ai*dimChunk + i] = X[batchId * strideXN + ai * strideXT + (dimOffset+i) * strideXD];
            dimBase[bi*dimChunk + i] = X[batchId * strideXN + bi * strideXT + (dimOffset+i) * strideXD];
        }
    } else {
        temp[ai] = 1;
        temp[bi] = 1;
    }

    // build sum in place up the tree
    for (int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();

        int bi = (tid + 1) * stride * 2 - 1;
        int ai = bi - stride;
        if (bi < seqlen) {
            for (int i=0; i<dimChunk; ++i) {
                dimBase[bi*dimChunk + i] += dimBase[ai*dimChunk + i] * temp[bi];
            }
            temp[bi] *= temp[ai];
        }
    }

    // traverse down tree & build scan
    for (int stride = powerOfTwo / 4; stride > 0; stride >>= 1) {
        __syncthreads();

        int ai = (tid + 1) * stride * 2 - 1;
        int bi = ai + stride;
        if (ai + stride < powerOfTwo) {
            for (int i=0; i<dimChunk; ++i) {
                dimBase[bi*dimChunk + i] += dimBase[ai*dimChunk + i] * temp[bi];
            }
            temp[bi] *= temp[ai];
        }
    }
	__syncthreads();

	if (tid < seqlen) {
        // store result in X
        for (int i=0; i<dimChunk; ++i) {
            X[batchId * strideXN + ai * strideXT + (dimOffset+i) * strideXD] = dimBase[ai*dimChunk + i];
            X[batchId * strideXN + bi * strideXT + (dimOffset+i) * strideXD] = dimBase[bi*dimChunk + i];
        }

        if (dimOffset == dim-dimChunk) {
            // store result in A
            A[batchId * strideAN + ai * strideAT] = temp[ai];
            A[batchId * strideAN + bi * strideAT] = temp[bi];
        }
	}
}


template<typename input_t>
void pscan_fwd_launch(PScanParams &params, cudaStream_t stream) {    

    assert(params.seqlen <= 2048);
    
    const int totalSharedMem = 48 * 1024;
    int powerOfTwo = nextPowerOfTwo(params.seqlen);

    int shared_mem_size = totalSharedMem;
    
    int dimChunk = (totalSharedMem - powerOfTwo * sizeof(input_t)) / (powerOfTwo * sizeof(input_t));
    while (params.dim % dimChunk != 0) {
        dimChunk -= 1;
    }
    
    //std::cout << dimChunk << std::endl;
    
    //int dimChunk = 8;
    //int dimChunk = 1;

    //std::cout << "N: " << params.batch << "; D: " << params.dim << std::endl;
    //std::cout << "stides: " << params.X.stride(0) << " " << params.X.stride(1) << " " << params.X.stride(2) << std::endl;

    int numThreads = _cdiv(params.seqlen, 2);
    int numBlocks = params.batch * params.dim / dimChunk;

    auto kernel = &pscan_fwd_kernel<input_t>;
    kernel<<<numBlocks, numThreads, shared_mem_size, stream>>>(
        params.A.data_ptr<input_t>(),
        params.X.data_ptr<input_t>(),
        params.X.stride(0), params.X.stride(1), params.X.stride(2),
        params.A.stride(0), params.A.stride(1),
        params.batch, params.seqlen, params.dim,
        powerOfTwo,
        dimChunk
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t>
void pscan_fwd_cuda(PScanParams &params, cudaStream_t stream) {
    pscan_fwd_launch<input_t>(params, stream);
}
