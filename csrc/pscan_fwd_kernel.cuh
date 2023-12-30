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


#define SHARED_MEMORY_BANKS 32
#define LOG_MEM_BANKS 5
//#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_MEM_BANKS)
#define CONFLICT_FREE_OFFSET(n) 0

template<typename scalar_t>
__global__ void pscan_fwd_kernel(
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> A,
    torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> X,
    int batch,
    int seqlen,
    int dim,
    int powerOfTwo,
    int dimChunk
) {
    // allocated on invocation 
    extern __shared__ float temp[];
    
    // float* dimBase = &temp[powerOfTwo + 32];

    //const int batchId = blockIdx.x;
    //const int dimBlock = blockIdx.y;
    const int batchId = 0;
    const int dimBlock = blockIdx.x;

    const int dimOffset = dimBlock * dimChunk;
    const int tid = threadIdx.x;

    int ai = 2 * tid;
    int bi = 2 * tid + 1;

    // float* dim_ai = dimBase + (dimChunk * ai);
    // float* dim_bi = dimBase + (dimChunk * bi);

    int bankOffsetA = 0; //CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = 0; //CONFLICT_FREE_OFFSET(bi);

    if (tid < seqlen) {
        temp[ai + bankOffsetA] = A[batchId][ai][0];
        temp[bi + bankOffsetB] = A[batchId][bi][0];

        // load dim chunk
        // for (int i=0; i < dimChunk; ++i) {
        //     // dim_ai[i] = X[batchId][ai][i+dimOffset];
        //     // dim_bi[i] = X[batchId][bi][i+dimOffset];
        //     dimBase[dimChunk * ai + i] = X[batchId][ai][i+dimOffset];
        //     dimBase[dimChunk * bi + i] = X[batchId][bi][i+dimOffset];
        // }
    } else {
        temp[ai + bankOffsetA] = 1;
        temp[bi + bankOffsetB] = 1;
    }

    for (int stride = 1; stride <= blockDim.x; stride *= 2)  {
        // build sum in place up the tree
        __syncthreads();
        int bi = (tid + 1) * stride * 2 - 1;
        if (bi < seqlen) {
            int ai = bi - stride;

            // float* dim_ai = dimBase + (dimChunk * ai);
            // float* dim_bi = dimBase + (dimChunk * bi);
            for (int i=0; i < dimChunk; ++i) {
                //X[batchId][bi][i+dimOffset] += X[batchId][ai][i+dimOffset] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
                X[batchId][bi][i+dimOffset] += X[batchId][ai][i+dimOffset] * temp[bi];

                //dimBase[(dimChunk * bi) + i] += dimBase[(dimChunk * ai) + i] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
                //dim_bi[i] += dim_ai[i] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
            }

            //temp[bi + CONFLICT_FREE_OFFSET(bi)] *= temp[ai + CONFLICT_FREE_OFFSET(ai)];
            temp[bi] *= temp[ai];
        }
    }

    // traverse down tree & build scan
    for (int stride = powerOfTwo / 4; stride > 0; stride >>= 1) {
        __syncthreads();
        int ai = (tid + 1) * stride * 2 - 1;
        if (ai + stride < powerOfTwo) {
            int bi = ai + stride;

            // float* dim_ai = dimBase + (dimChunk * ai);
            // float* dim_bi = dimBase + (dimChunk * bi);
            for (int i=0; i < dimChunk; ++i) {
                //X[batchId][bi][i+dimOffset] += X[batchId][ai][i+dimOffset] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
                X[batchId][bi][i+dimOffset] += X[batchId][ai][i+dimOffset] * temp[bi];

                //dimBase[(dimChunk * bi) + i] += dimBase[(dimChunk * ai) + i] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
                //dim_bi[i] += dim_ai[i] * temp[bi + CONFLICT_FREE_OFFSET(bi)];
            }

            //temp[bi+CONFLICT_FREE_OFFSET(bi)] *= temp[ai+CONFLICT_FREE_OFFSET(ai)];
            temp[bi] *= temp[ai];
        }
    }
	__syncthreads();

	if (tid < seqlen) {
        if (dimBlock == 0) {
            // store result in A
            A[batchId][ai][0] = temp[ai + bankOffsetA];
            A[batchId][bi][0] = temp[bi + bankOffsetB];
        }
        // store result in X
        // for (int i=0; i < dimChunk; ++i) {
        //     X[batchId][ai][i+dimOffset] = dimBase[(dimChunk * ai) + i];
        //     X[batchId][bi][i+dimOffset] = dimBase[(dimChunk * bi) + i];
        //     // X[batchId][ai][i+dimOffset] = dim_ai[i];
        //     // X[batchId][bi][i+dimOffset] = dim_bi[i];
        // }
	}
}


const int THREADS_PER_BLOCK = 512;
const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;


template<typename input_t>
void pscan_fwd_launch(PScanParams &params, cudaStream_t stream) {    

    auto accessorA = params.A.packed_accessor32<input_t, 3, torch::RestrictPtrTraits>();
    auto accessorX = params.X.packed_accessor32<input_t, 3, torch::RestrictPtrTraits>();

    assert(params.seqlen <= ELEMENTS_PER_BLOCK);

    int numThreads = _cdiv(params.seqlen, 2);
    int powerOfTwo = nextPowerOfTwo(params.seqlen);

    // default per block shared memory size limit: 48 KB
    //int dimBlocks = _cdiv(params.dim, 48 / sizeof(input_t) / 2);

    //int dimChunk = 4;
    //int dimBlocks = _cdiv(params.dim, dimChunk);
    //dim3 grid(params.batch, dimBlocks);
    
    int dimChunk = 1;
    //dim3 grid(params.batch, params.dim);
    dim3 grid(params.dim);
    
    int shared_mem_size = 48 * 1024;

    auto kernel = &pscan_fwd_kernel<input_t>;
    //kernel<<<grid, numThreads, shared_mem_size, stream>>>(
        kernel<<<grid, numThreads, shared_mem_size>>>(
        accessorA, accessorX,
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
