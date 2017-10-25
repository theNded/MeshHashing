#include "core/block_array.h"
#include "helper_cuda.h"

#include <device_launch_parameters.h>

////////////////////
/// Device code
////////////////////
__global__
void BlockArrayResetKernel(
    Block* blocks,
    int block_count
) {
  const uint block_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (block_idx < block_count) {
    blocks[block_idx].Clear();
  }
}

////////////////////
/// Host code
//////////////////////
__host__
BlockArray::BlockArray(uint block_count) {
  Resize(block_count);
}

//BlockArray::~BlockArray() {
//  Free();
//}

__host__
void BlockArray::Alloc(uint block_count) {
  if (! is_allocated_on_gpu_) {
    block_count_ = block_count;
    checkCudaErrors(cudaMalloc(&blocks_, sizeof(Block) * block_count));
    is_allocated_on_gpu_ = true;
  }
}

__host__
void BlockArray::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(blocks_));
    block_count_ = 0;
    blocks_ = NULL;
    is_allocated_on_gpu_ = false;
  }
}

__host__
void BlockArray::Resize(uint block_count) {
  if (is_allocated_on_gpu_) {
    Free();
  }
  Alloc(block_count);
  Reset();
}

__host__
void BlockArray::Reset() {
  const uint threads_per_block = 64;

  if (block_count_ == 0) return;

  // NOTE: this block is the parallel unit in CUDA, not the data structure Block
  const uint blocks = (block_count_ + threads_per_block - 1) / threads_per_block;

  const dim3 grid_size(blocks, 1);
  const dim3 block_size(threads_per_block, 1);

  BlockArrayResetKernel<<<grid_size, block_size>>>(blocks_, block_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}