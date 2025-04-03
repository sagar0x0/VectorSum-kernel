#!POPCORN leaderboard vectorsum

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# Set architecture for L4 GPU (compute capability 8.9)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'

def custom_kernel(data: input_t) -> output_t:
    A = data.contiguous()
    assert A.is_cuda, "Input tensor must be on GPU"
    assert A.dim() == 1, "Input tensor must be 1-dimensional"
    assert A.dtype == torch.float32, "Input tensor must be float32"

    N = A.numel()
    if N == 0:
        return 0.0

    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    template <typename scalar_t>
    __inline__ __device__ scalar_t warpReduceSum(scalar_t val) {
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    template <typename scalar_t>
    __inline__ __device__ scalar_t blockReduceSum(scalar_t val) {
        static __shared__ scalar_t shared[32];
        int lane = threadIdx.x % 32;
        int wid = threadIdx.x / 32;

        val = warpReduceSum(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < (blockDim.x + 31)/32) ? shared[lane] : 0;
        if (wid == 0) val = warpReduceSum(val);
        return val;
    }

    template <typename scalar_t>
    __global__ void vector_sum_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ block_sums,
        const int N
    ) {
        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int globalIdx = bid * blockDim.x + tid;

        scalar_t threadSum = 0;

        #pragma unroll 4
        for (int i = globalIdx; i < N; i += gridDim.x * blockDim.x) {
            threadSum += input[i];
        }

        scalar_t blockSum = blockReduceSum(threadSum);
        if (tid == 0) {
            block_sums[bid] = blockSum;
        }
    }

    template <typename scalar_t>
    __global__ void final_reduce_kernel(
        const scalar_t* __restrict__ block_sums,
        scalar_t* __restrict__ output,
        int num_blocks
    ) {
        scalar_t sum = 0;
        for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
            sum += block_sums[i];
        }
        sum = blockReduceSum(sum);
        if (threadIdx.x == 0) {
            output[0] = sum;
        }
    }

    torch::Tensor vector_sum_cuda(torch::Tensor input) {
        TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(input.dim() == 1, "Input must be 1-dimensional");
        TORCH_CHECK(input.scalar_type() == torch::kFloat32, "Input must be float32");
        
        const int N = input.numel();
        if (N == 0) {
            return torch::zeros(1, torch::TensorOptions()
                              .dtype(input.dtype())
                              .device(input.device()));
        }

        const int blockSize = 256;
        const int numBlocks = min((N + blockSize - 1) / blockSize, 4096);

        auto block_sums = torch::zeros({numBlocks}, input.options());
        auto output = torch::zeros({1}, input.options());
        
        // First pass: compute block sums
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vector_sum_kernel", 
        ([&] 
            {
                vector_sum_kernel<scalar_t><<<numBlocks, blockSize>>>(
                    input.data_ptr<scalar_t>(),
                    block_sums.data_ptr<scalar_t>(),
                    N
                );
                
                // Second pass: reduce block sums
                final_reduce_kernel<scalar_t><<<1, blockSize>>>(
                    block_sums.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    numBlocks
                );
            }
        )
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        return output;
    }
    """

    cpp_source = """
    #include <torch/extension.h>
    torch::Tensor vector_sum_cuda(torch::Tensor input);
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("vector_sum_cuda", &vector_sum_cuda, "Vector sum reduction");
    }
    """

    module = load_inline(
        name='vector_sum',
        cpp_sources=cpp_source,
        cuda_sources=cuda_source,
        extra_cuda_cflags=['-arch=sm_89', '-O3', '--use_fast_math'],
        verbose=False
    )

    result = module.vector_sum_cuda(A)
    return result.squeeze()