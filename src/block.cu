#include "block.h"
#include <helper_cuda.h>
#include <device_launch_parameters.h>

////////////////////
/// class Blocks
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void ResetBlocksKernel(BlocksGPU blocks, int block_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < block_count) {
    blocks[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////

/// Life cycle
Blocks::Blocks() {
  block_count_  = 0;
}

Blocks::~Blocks() {
  Free();
}

void Blocks::Alloc(uint block_count) {
  checkCudaErrors(cudaMalloc(&gpu_data_, sizeof(Block) * block_count));
}

void Blocks::Free() {
  checkCudaErrors(cudaFree(gpu_data_));
}

void Blocks::Resize(uint block_count) {
  block_count_ = block_count;
  Alloc(block_count);
  Reset();
}

/// Reset
void Blocks::Reset() {
  const int threads_per_block = 64;

  if (block_count_ == 0) return;
  const dim3 grid_size((block_count_ + threads_per_block - 1)
                      / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetBlocksKernel<<<grid_size, block_size>>>(gpu_data_, block_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}