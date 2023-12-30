#pragma once

//#include <cuda_bf16.h>
//#include <cuda_fp16.h>

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
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
    int powerOfTwo
) {
    // allocated on invocation 
    extern __shared__ float temp[];

    const int batchId = blockIdx.x;
    const int dimOffset = blockIdx.y;
    const int tid = threadIdx.x;

    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    if (tid < seqlen) {
        temp[ai ] = A[batchId * strideAN + ai * strideAT];
        temp[bi ] = A[batchId * strideAN + bi * strideAT];
    } else {
        temp[ai] = 1;
        temp[bi] = 1;
    }

    for (int stride = 1; stride <= blockDim.x; stride *= 2)  {
        // build sum in place up the tree
        __syncthreads();

        int bi = (tid + 1) * stride * 2 - 1;
        if (bi < seqlen) {
            int ai = bi - stride;

            
            X[batchId * strideXN + bi * strideXT + dimOffset * strideXD] += 
                X[batchId * strideXN + ai * strideXT + dimOffset * strideXD] * temp[bi];
            

            temp[bi] *= temp[ai];
        }
    }

    // traverse down tree & build scan
    for (int stride = powerOfTwo / 4; stride > 0; stride >>= 1) {
        __syncthreads();
        int ai = (tid + 1) * stride * 2 - 1;
        if (ai + stride < powerOfTwo) {
            int bi = ai + stride;
            
            X[batchId * strideXN + bi * strideXT + dimOffset * strideXD] += 
                X[batchId * strideXN + ai * strideXT + dimOffset * strideXD] * temp[bi];

            temp[bi] *= temp[ai];
        }
    }
	__syncthreads();

	if (tid < seqlen) {
        if (dimOffset == 0) {
            // store result in A
            A[batchId * strideAN + ai * strideAT] = temp[ai];
            A[batchId * strideAN + bi * strideAT] = temp[bi];
        }
	}
}


const int THREADS_PER_BLOCK = 512;
const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;


template<typename input_t>
void pscan_fwd_launch(PScanParams &params, cudaStream_t stream) {    

    auto accessorA = params.A.packed_accessor32<input_t, 3, torch::DefaultPtrTraits>();
    auto accessorX = params.X.packed_accessor32<input_t, 3, torch::DefaultPtrTraits>();

    assert(params.seqlen <= ELEMENTS_PER_BLOCK);

    int numThreads = _cdiv(params.seqlen, 2);
    int powerOfTwo = nextPowerOfTwo(params.seqlen);

    dim3 grid(params.batch, params.dim);
    
    int shared_mem_size = 48 * 1024;

    std::cout << "stide" << params.X.stride(0) << " " << params.X.stride(1) << " " << params.X.stride(2) << std::endl;

    auto kernel = &pscan_fwd_kernel<input_t>;
    //kernel<<<grid, numThreads, shared_mem_size, stream>>>(
        kernel<<<grid, numThreads, shared_mem_size>>>(
        //accessorA, accessorX,
        params.A.data_ptr<input_t>(),
        params.X.data_ptr<input_t>(),
        params.X.stride(0), params.X.stride(1), params.X.stride(2),
        params.A.stride(0), params.A.stride(1),
        params.batch, params.seqlen, params.dim,
        powerOfTwo
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}


template<typename input_t>
void pscan_fwd_cuda(PScanParams &params, cudaStream_t stream) {
    pscan_fwd_launch<input_t>(params, stream);
}
