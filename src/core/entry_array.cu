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
void ResetCompactEntriesKernel(HashEntry* entries, uint entry_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < entry_count) {
    entries[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
/// Life cycle
EntryArray::EntryArray() {}
//EntryArray::~EntryArray() {
//  Free();
//}

void EntryArray::Alloc(uint entry_count) {
  checkCudaErrors(cudaMalloc(&entries,
                             sizeof(HashEntry) * entry_count));
  checkCudaErrors(cudaMalloc(&candidate_entry_counter,
                             sizeof(int)));
  checkCudaErrors(cudaMalloc(&entry_recycle_flags,
                             sizeof(int) * entry_count));
}

void EntryArray::Free() {
  checkCudaErrors(cudaFree(entries));
  checkCudaErrors(cudaFree(candidate_entry_counter));
  checkCudaErrors(cudaFree(entry_recycle_flags));
}

void EntryArray::Resize(uint entry_count) {
  entry_count_ = entry_count;
  Alloc(entry_count);
  Reset();
}

/// Reset
void EntryArray::Reset() {
  const int threads_per_block = 64;
  const dim3 grid_size((entry_count_ + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetCompactEntriesKernel<<<grid_size, block_size>>>(entries, entry_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

uint EntryArray::entry_count(){
  uint count;
  checkCudaErrors(cudaMemcpy(&count, candidate_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return count;
}

void EntryArray::reset_entry_count() {
  checkCudaErrors(cudaMemset(candidate_entry_counter,
                             0, sizeof(uint)));
}
