#include "core/block.h"
#include "helper_cuda.h"

#include <device_launch_parameters.h>

////////////////////
/// Device code
////////////////////
__global__
void BlocksResetKernel(BlockGPUMemory blocks, int block_count) {
  const uint block_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (block_idx < block_count) {
    blocks[block_idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
Blocks::Blocks() {
  block_count_ = 0;
  gpu_memory_ = NULL;
}

Blocks::Blocks(uint block_count) {
  Resize(block_count);
}

Blocks::~Blocks() {
  Free();
}

void Blocks::Alloc(uint block_count) {
  checkCudaErrors(cudaMalloc(&gpu_memory_, sizeof(Block) * block_count));
}

void Blocks::Free() {
  checkCudaErrors(cudaFree(gpu_memory_));
  gpu_memory_ = NULL;
}

void Blocks::Resize(uint block_count) {
  if (gpu_memory_) {
    Free();
  }
  block_count_ = block_count;
  Alloc(block_count);
  Reset();
}

void Blocks::Reset() {
  const uint threads_per_block = 64;

  if (block_count_ == 0) return;

  // NOTE: this block is the parallel unit in CUDA, not the data structure Block
  const uint blocks = (block_count_ + threads_per_block - 1) / threads_per_block;

  const dim3 grid_size(blocks, 1);
  const dim3 block_size(threads_per_block, 1);

  BlocksResetKernel<<<grid_size, block_size>>>(gpu_memory_, block_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}