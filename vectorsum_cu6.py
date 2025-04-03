#!POPCORN leaderboard vectorsum

import os
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

# for L4 optimization 
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.9'  

def custom_kernel(data: input_t) -> output_t:
    """
    Cuda implementation of vectorSum
    Args:
        data: Input tensor of shape (N,)
    Returns:
        Scalar value equal to the sum of all elements.
    """
    A = data.contiguous()  # contiguous memory access

    assert A.is_cuda, "Input tensor must be on GPU"
    assert A.dim() == 1, "Input tensor must be 1-dimensional"
    assert A.dtype == torch.float32, "Input tensor must be float32"

    N = A.numel()
    if N == 0:
        return torch.zeros((), dtype=A.dtype, device=A.device)  # Return scalar

    cuda_source = """
    #include <torch/extension.h>
    #include <cuda_runtime.h>

    // Warp-level reduction using shuffle instructions
    template <typename scalar_t>
    __inline__ __device__ scalar_t warpReduceSum(scalar_t val) {
        
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }


    // Multi-level reduction kernel
    template <typename scalar_t>
    __global__ void vector_sum_kernel(
        const scalar_t* __restrict__ input,
        scalar_t* __restrict__ output,
        const int N
    ) {
        // Shared memory for block-level reduction
        extern __shared__ __align__(sizeof(scalar_t)) char shared_mem_char[];
        scalar_t* sharedMem = reinterpret_cast<scalar_t*>(shared_mem_char);

        const int tid = threadIdx.x;
        const int bid = blockIdx.x;
        const int globalIdx = bid * blockDim.x + tid;

        scalar_t threadSum = 0;

        // Vectorized loads with grid-stride loop
        for (int i = globalIdx; i < N; i += gridDim.x * blockDim.x) {
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
                atomicAdd(output, blockSum);
            }
        }
    }


        

    torch::Tensor vector_sum_cuda(torch::Tensor input) {
        TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
        TORCH_CHECK(input.dim() == 1, "Input must be 1-dimensional");
        
        const int N = input.numel();
        if (N == 0) {
            // Return a scalar tensor
            return torch::zeros(1, torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(input.device()));
        }

        const int blockSize = 256;
        const int numBlocks = min((N + blockSize - 1) / blockSize, 4096); 

        auto output = torch::zeros({1}, input.options());
        
        // For very small inputs, use single-stage reduction
        if (numBlocks == 1) {
            dim3 grid(1);
            dim3 block(blockSize);
            size_t shared_mem_size = (blockSize / 32) * sizeof(float);
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vector_sum_kernel", ([&] {
                vector_sum_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    N
                );
            }));
        } 
        else {
            //

            dim3 grid1(numBlocks);
            dim3 block1(blockSize);
            size_t shared_mem_size1 = (blockSize / 32) * sizeof(float);
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "vector_sum_kernel", ([&] {
                vector_sum_kernel<scalar_t><<<grid1, block1, shared_mem_size1>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    N
                );
            }));
            
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        // Convert to scalar tensor (item())
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