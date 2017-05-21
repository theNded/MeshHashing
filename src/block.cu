#include "block.h"
#include <helper_cuda.h>

__global__
void ResetBlocksKernel(VoxelBlocksGPU blocks, int block_count) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < block_count) {
    blocks[idx].Clear();
  }
}

void VoxelBlocks::Alloc(uint block_count) {
  checkCudaErrors(cudaMalloc(&gpu_data_,
                             sizeof(VoxelBlock) * block_count));
}

void VoxelBlocks::Free() {
  checkCudaErrors(cudaFree(gpu_data_));

}

VoxelBlocks::VoxelBlocks() {
  block_count_ = 0;
}

VoxelBlocks::~VoxelBlocks() {
  Free();
}

void VoxelBlocks::Reset() {
  const int threads_per_block = 64;

  if (block_count_ == 0) return;
  const dim3 grid_size((block_count_ + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetBlocksKernel<<<grid_size, block_size>>>(gpu_data_, block_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void VoxelBlocks::Resize(uint block_count) {
  block_count_ = block_count;
  Alloc(block_count);
}