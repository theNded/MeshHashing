#include "hash_table.h"

/// Kernel functions
__global__
void ResetBucketMutexesKernel(HashTableGPU<Block> hash_table) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count) {
    hash_table.bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__global__
void ResetHeapKernel(HashTableGPU<Block> hash_table) {
  const HashParams& hash_params = kHashParams;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    hash_table.heap_counter[0] = hash_params.value_capacity - 1;	//points to the last element of the array
  }

  if (idx < hash_params.value_capacity) {
    hash_table.heap[idx] = hash_params.value_capacity - idx - 1;
    hash_table.values[idx].Clear();
  }
}

__global__
void ResetEntriesKernel(HashTableGPU<Block> hash_table) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count * HASH_BUCKET_SIZE) {
    hash_table.ClearHashEntry(hash_table.hash_entries[idx]);
    hash_table.ClearHashEntry(hash_table.compacted_hash_entries[idx]);
  }
}

/// Member functions
template <typename T>
HashTable<T>::HashTable() {}

template <typename T>
HashTable<T>::HashTable(const HashParams &params) {
  hash_params_ = params;
  Alloc(params);
  Reset();
}

template <typename T>
HashTable<T>::~HashTable() {
  Free();
}

template <typename T>
HashTableGPU<T>& HashTable<T>::gpu_data() {
  return gpu_data_;
}

template <typename T>
void HashTable<T>::Resize(const HashParams &params) {
  hash_params_ = params;
  Alloc(params);
  Reset();
}

template <typename T>
void HashTable<T>::Alloc(const HashParams &params) {
  checkCudaErrors(cudaMalloc(&gpu_data_.heap, sizeof(uint) * params.value_capacity));
  checkCudaErrors(cudaMalloc(&gpu_data_.heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.values, sizeof(T) * params.value_capacity));
  checkCudaErrors(cudaMalloc(&gpu_data_.hash_entry_remove_flags, sizeof(int) * params.entry_count));

  checkCudaErrors(cudaMalloc(&gpu_data_.hash_entries, sizeof(HashEntry) * params.entry_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.compacted_hash_entries, sizeof(HashEntry) * params.entry_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.compacted_hash_entry_counter, sizeof(int)));

  checkCudaErrors(cudaMalloc(&gpu_data_.bucket_mutexes, sizeof(int) * params.bucket_count));
  gpu_data_.is_on_gpu = true;
}

template <typename T>
void HashTable<T>::Free() {
  if (gpu_data_.is_on_gpu) {
    checkCudaErrors(cudaFree(gpu_data_.heap));
    checkCudaErrors(cudaFree(gpu_data_.heap_counter));
    checkCudaErrors(cudaFree(gpu_data_.values));
    checkCudaErrors(cudaFree(gpu_data_.hash_entry_remove_flags));

    checkCudaErrors(cudaFree(gpu_data_.hash_entries));
    checkCudaErrors(cudaFree(gpu_data_.compacted_hash_entries));
    checkCudaErrors(cudaFree(gpu_data_.compacted_hash_entry_counter));

    checkCudaErrors(cudaFree(gpu_data_.bucket_mutexes));
    gpu_data_.is_on_gpu = false;
  }
}

/// Member host functions calling kernels
// (__host__)
template <typename T>
void HashTable<T>::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((hash_params_.bucket_count + threads_per_block - 1)
                       /threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetBucketMutexesKernel<<<grid_size, block_size>>>(gpu_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

// (__host__)
template <typename T>
void HashTable<T>::Reset() {
  /// Reset mutexes
  ResetMutexes();

  {
    /// Reset entries
    const int threads_per_block = 64;
    const dim3 grid_size((HASH_BUCKET_SIZE * hash_params_.bucket_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    ResetEntriesKernel<<<grid_size, block_size>>>(gpu_data_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    const int threads_per_block = 64;
    const dim3 grid_size((hash_params_.value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    ResetHeapKernel<<<grid_size, block_size>>>(gpu_data_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

template class HashTable<Block>;