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
void HashTableResetBucketMutexesKernel(int *bucket_mutexes, uint bucket_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < bucket_count) {
    bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__global__
void HashTableResetHeapKernel(uint *heap, uint value_capacity) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < value_capacity) {
    heap[idx] = value_capacity - idx - 1;
  }
}

__global__
void HashTableResetEntriesKernel(HashEntry *entries, uint entry_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < entry_count) {
    entries[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////

/// Life cycle
HashTable::HashTable() {}

HashTable::HashTable(const HashParams &params) {
  Alloc(params);
  Reset();
}

//HashTable::~HashTable() {
//  Free();
//}

void HashTable::Alloc(const HashParams &params) {
  /// Parameters
  bucket_count = params.bucket_count;
  bucket_size = params.bucket_size;
  entry_count = params.entry_count;
  value_capacity = params.value_capacity;
  linked_list_size = params.linked_list_size;

  /// Values
  checkCudaErrors(cudaMalloc(&heap_, sizeof(uint) * params.value_capacity));
  checkCudaErrors(cudaMalloc(&heap_counter_, sizeof(uint)));

  /// Entries
  checkCudaErrors(cudaMalloc(&entries_, sizeof(HashEntry) * params.entry_count));

  /// Mutexes
  checkCudaErrors(cudaMalloc(&bucket_mutexes_, sizeof(int) * params.bucket_count));
}

void HashTable::Free() {
  checkCudaErrors(cudaFree(heap_));
  checkCudaErrors(cudaFree(heap_counter_));

  checkCudaErrors(cudaFree(entries_));
  checkCudaErrors(cudaFree(bucket_mutexes_));
}

void HashTable::Resize(const HashParams &params) {
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
    const dim3 grid_size((entry_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetEntriesKernel <<<grid_size, block_size>>>(entries_, entry_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    /// Reset allocated memory
    uint heap_counter_init = value_capacity - 1;
    checkCudaErrors(cudaMemcpy(heap_counter_, &heap_counter_init,
                               sizeof(uint),
                               cudaMemcpyHostToDevice));

    const int threads_per_block = 64;
    const dim3 grid_size((value_capacity + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    HashTableResetHeapKernel <<<grid_size, block_size>>>(heap_, value_capacity);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

void HashTable::ResetMutexes() {
  const int threads_per_block = 64;
  const dim3 grid_size((bucket_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  HashTableResetBucketMutexesKernel <<<grid_size, block_size>>>(bucket_mutexes_, bucket_count);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// Member function: Others
//void HashTable::Debug() {
//  HashEntry *entries = new HashEntry[hash_params_.bucket_size * hash_params_.bucket_count];
//  uint *heap_ = new uint[hash_params_.value_capacity];
//  uint  heap_counter_;
//
//  checkCudaErrors(cudaMemcpy(&heap_counter_, heap_counter_, sizeof(uint), cudaMemcpyDeviceToHost));
//  heap_counter_++; //points to the first free entry: number of blocks is one more
//
//  checkCudaErrors(cudaMemcpy(heap_, heap_,
//                             sizeof(uint) * hash_params_.value_capacity,
//                             cudaMemcpyDeviceToHost));
//  checkCudaErrors(cudaMemcpy(entries, entries,
//                             sizeof(HashEntry) * hash_params_.bucket_size * hash_params_.bucket_count,
//                             cudaMemcpyDeviceToHost));
////  checkCudaErrors(cudaMemcpy(values, values,
////                             sizeof(T) * hash_params_.value_capacity,
////                             cudaMemcpyDeviceToHost));
//
//  LOG(INFO) << "GPU -> CPU data transfer finished";
//
//  //Check for duplicates
//  class Entry {
//  public:
//    Entry() {}
//    Entry(int x_, int y_, int z_, int i_, int offset_, int ptr_) :
//            x(x_), y(y_), z(z_), i(i_), offset(offset_), ptr(ptr_) {}
//    ~Entry() {}
//
//    bool operator< (const Entry &other) const {
//      if (x == other.x) {
//        if (y == other.y) {
//          return z < other.z;
//        } return y < other.y;
//      } return x < other.x;
//    }
//
//    bool operator== (const Entry &other) const {
//      return x == other.x && y == other.y && z == other.z;
//    }
//
//    int x, y, z, i;
//    int offset;
//    int ptr;
//  };
//
//  /// Iterate over free heap_
//  std::unordered_set<uint> free_heap_index;
//  std::vector<int> free_value_index(hash_params_.value_capacity, 0);
//  for (uint i = 0; i < heap_counter_; i++) {
//    free_heap_index.insert(heap_[i]);
//    free_value_index[heap_[i]] = FREE_ENTRY;
//  }
//  if (free_heap_index.size() != heap_counter_) {
//    LOG(ERROR) << "heap_ check invalid";
//  }
//
//  uint not_free_entry_count = 0;
//  uint not_locked_entry_count = 0;
//
//  /// Iterate over entries
//  std::list<Entry> l;
//  uint entry_count = hash_params_.count;
//  for (uint i = 0; i < count; i++) {
//    if (entries[i].ptr != LOCK_ENTRY) {
//      not_locked_entry_count++;
//    }
//
//    if (entries[i].ptr != FREE_ENTRY) {
//      not_free_entry_count++;
//      Entry occupied_entry(entries[i].pos.x, entries[i].pos.y, entries[i].pos.z,
//                           i, entries[i].offset, entries[i].ptr);
//      l.push_back(occupied_entry);
//
//      if (free_heap_index.find(occupied_entry.ptr) != free_heap_index.end()) {
//        LOG(ERROR) << "ERROR: ptr is on free heap_, but also marked as an allocated entry";
//      }
//      free_value_index[entries[i].ptr] = LOCK_ENTRY;
//    }
//  }
//
//  /// Iterate over values
//  uint free_value_count = 0;
//  uint not_free_value_count = 0;
//  for (uint i = 0; i < hash_params_.value_capacity; i++) {
//    if (free_value_index[i] == FREE_ENTRY) {
//      free_value_count++;
//    } else if (free_value_index[i] == LOCK_ENTRY) {
//      not_free_value_count++;
//    } else {
//      LOG(ERROR) << "memory leak detected: neither free nor allocated";
//      return;
//    }
//  }
//
//  if (free_value_count + not_free_value_count == hash_params_.value_capacity)
//    LOG(INFO) << "heap_ OK!";
//  else {
//    LOG(ERROR) << "heap_ CORRUPTED";
//    return;
//  }
//
//  l.sort();
//  size_t size_before = l.size();
//  l.unique();
//  size_t size_after = l.size();
//
//
//  LOG(INFO) << "Duplicated entry count: " << size_before - size_after;
//  LOG(INFO) << "Not locked entry count: " << not_locked_entry_count;
//  LOG(INFO) << "Not free value count: " << not_free_value_count
//            << "; free value count: " << free_value_count;
//  LOG(INFO) << "not_free + free entry count: "
//            << not_free_value_count + free_value_count;
//
//  delete [] entries;
//  //delete [] values;
//  delete [] heap_;
//}