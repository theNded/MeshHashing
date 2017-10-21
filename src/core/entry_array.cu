//
// Created by wei on 17-10-21.
//

#include "core/entry_array.h"
#include "helper_cuda.h"
#include <device_launch_parameters.h>

////////////////////
/// Device code
////////////////////
__global__
void EntryArrayResetKernel(HashEntry* entries, uint entry_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < entry_count) {
    entries[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
/// Life cycle
__host__
EntryArray::EntryArray() {
  entry_count_ = 0;

  entries_ = NULL;
  flags_ = NULL;
  counter_ = NULL;
}

__host__
EntryArray::EntryArray(uint entry_count) {
  Resize(entry_count);
}
//EntryArray::~EntryArray() {
//  Free();
//}

__host__
void EntryArray::Alloc(uint entry_count) {
  entry_count_ = entry_count;
  checkCudaErrors(cudaMalloc(&entries_, sizeof(HashEntry) * entry_count));
  checkCudaErrors(cudaMalloc(&counter_, sizeof(int)));
  checkCudaErrors(cudaMalloc(&flags_, sizeof(uchar) * entry_count));
}

__host__
void EntryArray::Free() {
  checkCudaErrors(cudaFree(entries_));
  checkCudaErrors(cudaFree(counter_));
  checkCudaErrors(cudaFree(flags_));
  entry_count_ = 0;
  entries_ = NULL;
  flags_ = NULL;
  counter_ = NULL;
}

__host__
void EntryArray::Resize(uint entry_count) {
  if (entries_ != NULL) {
    Free();
  }
  Alloc(entry_count);
  Reset();
}

__host__
void EntryArray::Reset() {
  const int threads_per_block = 64;
  const dim3 grid_size((entry_count_ + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  EntryArrayResetKernel<<<grid_size, block_size>>>(entries_, entry_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

__host__
uint EntryArray::count(){
  uint count;
  checkCudaErrors(cudaMemcpy(&count,
                             counter_,
                             sizeof(uint),
                             cudaMemcpyDeviceToHost));
  return count;
}

void EntryArray::reset_count() {
  checkCudaErrors(cudaMemset(counter_, 0, sizeof(uint)));
}