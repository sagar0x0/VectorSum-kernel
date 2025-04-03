#!POPCORN leaderboard vectorsum

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# Set architecture for L4 GPU (compute capability 8.9)
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  # Critical for L4 optimization

def custom_kernel(data: input_t) -> output_t:
    """
    Custom implementation of vector sum using CUDA inline function.
    Args:
        data: Input tensor of shape (N,) with values from a normal distribution.
    Returns:
        Scalar value equal to the sum of all elements.
    """
    A = data.contiguous()  # Ensure contiguous memory access

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
    __global__ void vector_sum_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        const int N
    ) {
        // aligned dynamic allocation
        extern __shared__ __align__(sizeof(scalar_t)) char shared_mem_char[];
        scalar_t* sharedMem = reinterpret_cast<scalar_t*>(shared_mem_char);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int globalIdx = bid * blockDim.x + tid;
        const int numThreads = gridDim.x * blockDim.x;

        scalar_t threadSum = 0;

        using Vec4 = float4; // Use int4 for integer types
        const Vec4* vec_input = reinterpret_cast<const Vec4*>(input); // Cast ONCE

        // Vectorized loads grid-stride loop 
        // as each thread start at 0,4,8 cuz between values be added by thier prev threads
        for (int i = globalIdx ; i < N/4; i += numThreads) {
            Vec4 v = vec_input[i]; // Load 4 elements
            threadSum += v.x + v.y + v.z + v.w;
        }

        int remainder_start = (N / 4) * 4;
        for (int i = remainder_start + globalIdx; i < N; i += numThreads) {
            threadSum += input[i];
        }


        // Warp-level reduction
        threadSum = warpReduceSum(threadSum);

        // Store warp sums to shared memory
        if (tid % 32 == 0) {
            sharedMem[tid / 32] = threadSum;
        }
        __syncthreads();

        // First warp reduces partial sums
        if (tid < 32) {
            scalar_t val = (tid < (blockDim.x + 31) / 32) ? sharedMem[tid] : 0;
            scalar_t blockSum = warpReduceSum(val);
            if (tid == 0) {
                output[bid] = blockSum;
            }
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
        // no cap over the size 
        const int numBlocks = (N + blockSize - 1) / blockSize; 

        
        auto partial_sums = torch::zeros(numBlocks, torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(input.device()));

        dim3 grid(numBlocks);
        dim3 block(blockSize);
        
        // Calculate shared memory size based on block configuration
        size_t shared_mem_size = (blockSize / 32) * sizeof(float);

        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vector_sum_kernel", ([&] {
            vector_sum_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
                input.data_ptr<scalar_t>(),
                partial_sums.data_ptr<scalar_t>(),
                N
            );
        }));

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }


        return partial_sums;
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

    result = module.vector_sum_cuda(A.contiguous())
    return result.sum()   # Maintains GPU device (1 to 1024 sums as max blocksize=124)