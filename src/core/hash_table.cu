#include <unordered_set>
#include <vector>
#include <list>
#include <glog/logging.h>
#include <device_launch_parameters.h>

#include "core/hash_table.h"

////////////////////
/// class HashTable
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void ResetBucketMutexesKernel(HashTableGPU hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *hash_table.bucket_count) {
    hash_table.bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__global__
void ResetHeapKernel(HashTableGPU hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *hash_table.value_capacity) {
    hash_table.heap[idx] = *hash_table.value_capacity - idx - 1;
  }
}

__global__
void ResetEntriesKernel(HashTableGPU hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < *hash_table.entry_count) {
    hash_table.entries[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////

/// Life cycle
HashTable::HashTable() {}

HashTable::HashTable(const HashParams &params) {
  hash_params_ = params;
  Alloc(params);
  Reset();
}

HashTable::~HashTable() {
  Free();
}

void HashTable::Alloc(const HashParams &params) {
  /// Parameters
  checkCudaErrors(cudaMalloc(&gpu_memory_.bucket_count, sizeof(uint)));
  checkCudaErrors(cudaMemcpy(gpu_memory_.bucket_count, &params.bucket_count,
                             sizeof(uint), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_memory_.bucket_size, sizeof(uint)));
  checkCudaErrors(cudaMemcpy(gpu_memory_.bucket_size, &params.bucket_size,
                             sizeof(uint), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_memory_.entry_count, sizeof(uint)));
  checkCudaErrors(cudaMemcpy(gpu_memory_.entry_count, &params.entry_count,
                             sizeof(uint), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_memory_.value_capacity, sizeof(uint)));
  checkCudaErrors(cudaMemcpy(gpu_memory_.value_capacity, &params.value_capacity,
                             sizeof(uint), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc(&gpu_memory_.linked_list_size, sizeof(uint)));
  checkCudaErrors(cudaMemcpy(gpu_memory_.linked_list_size, &params.linked_list_size,
                             sizeof(uint), cudaMemcpyHostToDevice));

  /// Values
  checkCudaErrors(cudaMalloc(&gpu_memory_.heap,
                             sizeof(uint) * params.value_capacity));
  checkCudaErrors(cudaMalloc(&gpu_memory_.heap_counter,
                             sizeof(uint)));

  /// Entries
  checkCudaErrors(cudaMalloc(&gpu_memory_.entries,
                             sizeof(HashEntry) * params.entry_count));

  /// Mutexes
  checkCudaErrors(cudaMalloc(&gpu_memory_.bucket_mutexes,
                             sizeof(int) * params.bucket_count));
}

void HashTable::Free() {
  checkCudaErrors(cudaFree(gpu_memory_.bucket_count));
  checkCudaErrors(cudaFree(gpu_memory_.bucket_size));
  checkCudaErrors(cudaFree(gpu_memory_.entry_count));
  checkCudaErrors(cudaFree(gpu_memory_.value_capacity));
  checkCudaErrors(cudaFree(gpu_memory_.linked_list_size));

  checkCudaErrors(cudaFree(gpu_memory_.heap));
  checkCudaErrors(cudaFree(gpu_memory_.heap_counter));

  checkCudaErrors(cudaFree(gpu_memory_.entries));

  checkCudaErrors(cudaFree(gpu_memory_.bucket_mutexes));
}

void HashTable::Resize(const HashParams &params) {
  hash_params_ = params;
  Alloc(params);
  Reset();
}
/// Reset
void HashTable::Reset() {
  /// Reset mutexes
  ResetMutexes();

  {
    /// Reset entries
    const int threads_per_block = 64;
    const dim3 grid_size((hash_params_.entry_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    ResetEntriesKernel<<<grid_size, block_size>>>(gpu_memory_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    uint heap_counter = hash_params_.value_capacity - 1;
    checkCudaErrors(cudaMemcpy(gpu_memory_.heap_counter, &heap_counter,
                               sizeof(uint),
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 64;
    const dim3 grid_size((hash_params_.value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    ResetHeapKernel<<<grid_size, block_size>>>(gpu_memory_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

void HashTable::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((hash_params_.bucket_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetBucketMutexesKernel<<<grid_size, block_size>>>(gpu_memory_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

////////////////////
/// class CandidateEntryPool
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void ResetCompactEntriesKernel(CandidateEntryPoolGPU hash_table, uint entry_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < entry_count) {
    hash_table.entries[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////

/// Life cycle
CandidateEntryPool::CandidateEntryPool() {}
CandidateEntryPool::~CandidateEntryPool() {
  Free();
}

void CandidateEntryPool::Alloc(uint entry_count) {
  checkCudaErrors(cudaMalloc(&gpu_memory_.entries,
                             sizeof(HashEntry) * entry_count));
  checkCudaErrors(cudaMalloc(&gpu_memory_.candidate_entry_counter,
                             sizeof(int)));
  checkCudaErrors(cudaMalloc(&gpu_memory_.entry_recycle_flags,
                             sizeof(int) * entry_count));
}

void CandidateEntryPool::Free() {
  checkCudaErrors(cudaFree(gpu_memory_.entries));
  checkCudaErrors(cudaFree(gpu_memory_.candidate_entry_counter));
  checkCudaErrors(cudaFree(gpu_memory_.entry_recycle_flags));
}

void CandidateEntryPool::Resize(uint entry_count) {
  entry_count_ = entry_count;
  Alloc(entry_count);
  Reset();
}

/// Reset
void CandidateEntryPool::Reset() {
  const int threads_per_block = 64;
  const dim3 grid_size((entry_count_ + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetCompactEntriesKernel<<<grid_size, block_size>>>(gpu_memory_, entry_count_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

uint CandidateEntryPool::entry_count(){
  uint count;
  checkCudaErrors(cudaMemcpy(&count, gpu_memory_.candidate_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return count;
}

void CandidateEntryPool::reset_entry_count() {
  checkCudaErrors(cudaMemset(gpu_memory_.candidate_entry_counter,
                             0, sizeof(uint)));
}

/// Member function: Others
void HashTable::Debug() {
  HashEntry *entries = new HashEntry[hash_params_.bucket_size * hash_params_.bucket_count];
  uint *heap = new uint[hash_params_.value_capacity];
  uint  heap_counter;

  checkCudaErrors(cudaMemcpy(&heap_counter, gpu_memory_.heap_counter, sizeof(uint), cudaMemcpyDeviceToHost));
  heap_counter++; //points to the first free entry: number of blocks is one more

  checkCudaErrors(cudaMemcpy(heap, gpu_memory_.heap,
                             sizeof(uint) * hash_params_.value_capacity,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(entries, gpu_memory_.entries,
                             sizeof(HashEntry) * hash_params_.bucket_size * hash_params_.bucket_count,
                             cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaMemcpy(values, gpu_memory_.values,
//                             sizeof(T) * hash_params_.value_capacity,
//                             cudaMemcpyDeviceToHost));

  LOG(INFO) << "GPU -> CPU data transfer finished";

  //Check for duplicates
  class Entry {
  public:
    Entry() {}
    Entry(int x_, int y_, int z_, int i_, int offset_, int ptr_) :
            x(x_), y(y_), z(z_), i(i_), offset(offset_), ptr(ptr_) {}
    ~Entry() {}

    bool operator< (const Entry &other) const {
      if (x == other.x) {
        if (y == other.y) {
          return z < other.z;
        } return y < other.y;
      } return x < other.x;
    }

    bool operator== (const Entry &other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    int x, y, z, i;
    int offset;
    int ptr;
  };

  /// Iterate over free heap
  std::unordered_set<uint> free_heap_index;
  std::vector<int> free_value_index(hash_params_.value_capacity, 0);
  for (uint i = 0; i < heap_counter; i++) {
    free_heap_index.insert(heap[i]);
    free_value_index[heap[i]] = FREE_ENTRY;
  }
  if (free_heap_index.size() != heap_counter) {
    LOG(ERROR) << "Heap check invalid";
  }

  uint not_free_entry_count = 0;
  uint not_locked_entry_count = 0;

  /// Iterate over entries
  std::list<Entry> l;
  uint entry_count = hash_params_.entry_count;
  for (uint i = 0; i < entry_count; i++) {
    if (entries[i].ptr != LOCK_ENTRY) {
      not_locked_entry_count++;
    }

    if (entries[i].ptr != FREE_ENTRY) {
      not_free_entry_count++;
      Entry occupied_entry(entries[i].pos.x, entries[i].pos.y, entries[i].pos.z,
                           i, entries[i].offset, entries[i].ptr);
      l.push_back(occupied_entry);

      if (free_heap_index.find(occupied_entry.ptr) != free_heap_index.end()) {
        LOG(ERROR) << "ERROR: ptr is on free heap, but also marked as an allocated entry";
      }
      free_value_index[entries[i].ptr] = LOCK_ENTRY;
    }
  }

  /// Iterate over values
  uint free_value_count = 0;
  uint not_free_value_count = 0;
  for (uint i = 0; i < hash_params_.value_capacity; i++) {
    if (free_value_index[i] == FREE_ENTRY) {
      free_value_count++;
    } else if (free_value_index[i] == LOCK_ENTRY) {
      not_free_value_count++;
    } else {
      LOG(ERROR) << "memory leak detected: neither free nor allocated";
      return;
    }
  }

  if (free_value_count + not_free_value_count == hash_params_.value_capacity)
    LOG(INFO) << "HEAP OK!";
  else {
    LOG(ERROR) << "HEAP CORRUPTED";
    return;
  }

  l.sort();
  size_t size_before = l.size();
  l.unique();
  size_t size_after = l.size();


  LOG(INFO) << "Duplicated entry count: " << size_before - size_after;
  LOG(INFO) << "Not locked entry count: " << not_locked_entry_count;
  LOG(INFO) << "Not free value count: " << not_free_value_count
            << "; free value count: " << free_value_count;
  LOG(INFO) << "not_free + free entry count: "
            << not_free_value_count + free_value_count;

  delete [] entries;
  //delete [] values;
  delete [] heap;
}