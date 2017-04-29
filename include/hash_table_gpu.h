//
// Created by wei on 17-3-12.
//

/// HashTable for VoxelHashing
// TODO(wei): put util functions (geometry transform) into another .h

/// Header both used for .cu and .cc
/// To correctly use this header,
/// 1. include it in some .cu file,
/// 2. compile it by nvcc and generate a library,
/// 3. link it to an executable
#ifndef VH_HASH_DATA_H
#define VH_HASH_DATA_H

#include "common.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include "core.h"
#include "geometry_util.h"
#include "hash_param.h"
#include "sensor_data.h"

/// constant.cu
extern __constant__ HashParams kHashParams;
extern void SetConstantHashParams(const HashParams& hash_params);

// TODO(wei): make a templated class to adapt to more types
template <typename ValueType>
class HashTableGPU {
public:
  /// Hash VALUE part
  uint      *heap;               /// index to free values
  uint      *heap_counter;       /// single element; used as an atomic counter (points to the next free block)
  ValueType *values;             /// pre-allocated and managed by heap manually to avoid expensive malloc
  int       *hash_entry_remove_flags; /// used in garbage collection

  /// Hash KEY part
  HashEntry *hash_entries;                 /// hash entries that stores pointers to sdf values
  HashEntry *compacted_hash_entries;       /// allocated for parallel computation
  int       *compacted_hash_entry_counter; /// atomic counter to add compacted entries atomically
                                           /// == occupied_block_count
  /// Misc
  int       *bucket_mutexes;     /// binary flag per hash bucket; used for allocation to atomically lock a bucket
  bool       is_on_gpu;          /// the class be be used on both cpu and gpu

  ///////////////
  // Host part //
  ///////////////
  __device__ __host__
  HashTableGPU() {
    heap = NULL;
    heap_counter = NULL;
    values = NULL;
    hash_entry_remove_flags = NULL;

    hash_entries = NULL;
    compacted_hash_entries = NULL;
    compacted_hash_entry_counter = NULL;

    bucket_mutexes = NULL;
    is_on_gpu = false;
  }

  __host__
  void Alloc(const HashParams &params, bool is_data_on_gpu = true) {
    is_on_gpu = is_data_on_gpu;
    if (is_on_gpu) {
      checkCudaErrors(cudaMalloc(&heap, sizeof(uint) * params.value_capacity));
      checkCudaErrors(cudaMalloc(&heap_counter, sizeof(uint)));
      checkCudaErrors(cudaMalloc(&values, sizeof(ValueType) * params.value_capacity));
      checkCudaErrors(cudaMalloc(&hash_entry_remove_flags, sizeof(int) * params.entry_count));

      checkCudaErrors(cudaMalloc(&hash_entries, sizeof(HashEntry) * params.entry_count));
      checkCudaErrors(cudaMalloc(&compacted_hash_entries, sizeof(HashEntry) * params.entry_count));
      checkCudaErrors(cudaMalloc(&compacted_hash_entry_counter, sizeof(int)));

      checkCudaErrors(cudaMalloc(&bucket_mutexes, sizeof(int) * params.bucket_count));
    } else {
      heap               = new uint[params.value_capacity];
      heap_counter       = new uint[1];
      values             = new ValueType[params.value_capacity];
      hash_entry_remove_flags = new int[params.entry_count];

      hash_entries                 = new HashEntry[params.entry_count];
      compacted_hash_entries       = new HashEntry[params.entry_count];
      compacted_hash_entry_counter = new int[1];

      bucket_mutexes = new int[params.bucket_count];
    }
  }

  __host__
  void Free() {
    if (is_on_gpu) {
      checkCudaErrors(cudaFree(heap));
      checkCudaErrors(cudaFree(heap_counter));
      checkCudaErrors(cudaFree(values));
      checkCudaErrors(cudaFree(hash_entry_remove_flags));

      checkCudaErrors(cudaFree(hash_entries));
      checkCudaErrors(cudaFree(compacted_hash_entries));
      checkCudaErrors(cudaFree(compacted_hash_entry_counter));

      checkCudaErrors(cudaFree(bucket_mutexes));
    } else {
      if (heap)               delete[] heap;
      if (heap_counter)       delete[] heap_counter;
      if (values)             delete[] values;
      if (hash_entry_remove_flags) delete[] hash_entry_remove_flags;

      if (hash_entries)                 delete[] hash_entries;
      if (compacted_hash_entries)       delete[] compacted_hash_entries;
      if (compacted_hash_entry_counter) delete[] compacted_hash_entry_counter;

      if (bucket_mutexes) delete[] bucket_mutexes;
    }

    heap               = NULL;
    heap_counter       = NULL;
    values             = NULL;
    hash_entry_remove_flags = NULL;

    hash_entries                 = NULL;
    compacted_hash_entries       = NULL;
    compacted_hash_entry_counter = NULL;

    bucket_mutexes = NULL;
  }

  /////////////////
  // Device part //
  /////////////////
#ifdef __CUDACC__
  /// There are 3 kinds of positions (pos)
  /// 1. world pos, unit: meter
  /// 2. voxel pos, unit: voxel (typically 0.004m)
  /// 3. block pos, unit: block (typically 8 voxels)

  //! see Teschner et al. (but with correct prime values)
  __device__
  uint HashBucketForBlockPos(const int3& block_pos) const {
    const int p0 = 73856093;
    const int p1 = 19349669;
    const int p2 = 83492791;

    int res = ((block_pos.x * p0) ^ (block_pos.y * p1) ^ (block_pos.z * p2)) % kHashParams.bucket_count;
    if (res < 0) res += kHashParams.bucket_count;
    return (uint) res;
  }


  ////////////////////////////////////////
  /// Access
  __device__
    void ClearHashEntry(uint id) {
      ClearHashEntry(hash_entries[id]);
  }

  __device__
  void ClearHashEntry(HashEntry& hash_entry) {
    hash_entry.pos    = make_int3(0);
    hash_entry.offset = 0;
    hash_entry.ptr    = FREE_ENTRY;
  }


  __device__
  bool IsBlockAllocated(const int3& block_pos, const HashEntry& hash_entry) const {
    return block_pos.x == hash_entry.pos.x
        && block_pos.y == hash_entry.pos.y
        && block_pos.z == hash_entry.pos.z
        && hash_entry.ptr != FREE_ENTRY;
  }


  __device__
  HashEntry GetEntry(const int3& block_pos) const {
    uint bucket_idx             = HashBucketForBlockPos(block_pos);
    uint bucket_first_entry_idx = bucket_idx * HASH_BUCKET_SIZE;

    HashEntry entry;
    entry.pos    = block_pos;
    entry.offset = 0;
    entry.ptr    = FREE_ENTRY;

    for (uint i = 0; i < HASH_BUCKET_SIZE; ++i) {
      HashEntry curr_entry = hash_entries[i + bucket_first_entry_idx];
      if (IsBlockAllocated(block_pos, curr_entry)) {
        return curr_entry;
      }
    }

    /// The last entry is visted twice, but its OK
#ifdef HANDLE_COLLISIONS
    const uint bucket_last_entry_idx = bucket_first_entry_idx + HASH_BUCKET_SIZE - 1;
    int i = bucket_last_entry_idx;
    HashEntry curr_entry;

    #pragma unroll 1
    for (uint iter = 0; iter < kHashParams.linked_list_size; ++iter) {
      curr_entry = hash_entries[i];

      if (IsBlockAllocated(block_pos, curr_entry)) {
        return curr_entry;
      }
      if (curr_entry.offset == 0) { // should never reach here
        break;
      }
      i = bucket_last_entry_idx + curr_entry.offset;
      i %= (kHashParams.entry_count); // avoid overflow
    }
#endif
    return entry;
  }

  __device__
  uint AllocHeap() {
    uint addr = atomicSub(&heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    return heap[addr];
  }
  __device__
  void FreeHeap(uint ptr) {
    uint addr = atomicAdd(&heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    heap[addr + 1] = ptr;
  }

  //pos in SDF block coordinates
  __device__
  void AllocEntry(const int3& pos) {
    uint bucket_idx             = HashBucketForBlockPos(pos);				//hash bucket
    uint bucket_first_entry_idx = bucket_idx * HASH_BUCKET_SIZE;	//hash position

    int empty_entry_idx = -1;
    for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr_entry = hash_entries[i];
      if (IsBlockAllocated(pos, curr_entry)) {
        return;
      }

      /// wei: should not break and alloc before a thorough searching is over
      if (empty_entry_idx == -1 && curr_entry.ptr == FREE_ENTRY) {
        empty_entry_idx = i;
      }
    }

#ifdef HANDLE_COLLISIONS
    const uint bucket_last_entry_idx = bucket_first_entry_idx + HASH_BUCKET_SIZE - 1;
    uint i = bucket_last_entry_idx;
    int offset = 0;
    for (uint iter = 0; iter < kHashParams.linked_list_size; ++iter) {
      HashEntry& curr_entry = hash_entries[i];
      if (IsBlockAllocated(pos, curr_entry)) {
        return;
      }
      if (curr_entry.offset == 0) {
        break;
      }
      i = (bucket_last_entry_idx + curr_entry.offset) % kHashParams.entry_count;
    }
#endif

    if (empty_entry_idx != -1) {
      int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
      if (lock != LOCK_ENTRY) {
        HashEntry& entry = hash_entries[empty_entry_idx];
        entry.pos    = pos;
        entry.offset = NO_OFFSET;
        entry.ptr    = AllocHeap();	//memory alloc
      }
      return;
    }

#ifdef HANDLE_COLLISIONS
    i = bucket_last_entry_idx;
    offset = 0;

    #pragma  unroll 1
    for (uint iter = 0; iter < kHashParams.linked_list_size; ++iter) {
      offset ++;
      if ((offset % HASH_BUCKET_SIZE) == 0) continue;

      i = (bucket_last_entry_idx + offset) % kHashParams.entry_count;

      HashEntry& curr_entry = hash_entries[i];

      if (curr_entry.ptr == FREE_ENTRY) {	//this is the first free entry
        int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          // Not use reference in order to avoid lock ?
          HashEntry& bucket_last_entry = hash_entries[bucket_last_entry_idx];
          uint alloc_bucket_idx = i / HASH_BUCKET_SIZE;

          lock = atomicExch(&bucket_mutexes[alloc_bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            HashEntry& entry = hash_entries[i];
            entry.pos    = pos;
            entry.offset = bucket_last_entry.offset; // pointer assignment in linked list
            entry.ptr    = AllocHeap();	//memory alloc

            // Not sure if it is ok to directly assign to reference
            bucket_last_entry.offset = offset;
            hash_entries[bucket_last_entry_idx] = bucket_last_entry;
          }
        }
        return;	//bucket was already locked
      }
    }
#endif
  }


  //! deletes a hash entry position for a given block_pos index (returns true uppon successful deletion; otherwise returns false)
  __device__
  bool DeleteEntry(const int3& block_pos) {
    uint bucket_idx = HashBucketForBlockPos(block_pos);	//hash bucket
    uint bucket_first_entry_idx = bucket_idx * HASH_BUCKET_SIZE;		//hash position

    for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr = hash_entries[i];
      if (IsBlockAllocated(block_pos, curr)) {

#ifndef HANDLE_COLLISIONS
        FreeHeap(curr.ptr);
        ClearHashEntry(i);
        return true;
#else
        // Deal with linked list: curr = curr->next
        if (curr.offset != 0) {
          int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            FreeHeap(curr.ptr);
            int next_idx = (i + curr.offset) % (kHashParams.entry_count);
            hash_entries[i] = hash_entries[next_idx];
            ClearHashEntry(next_idx);
            return true;
          } else {
            return false;
          }
        } else {
          FreeHeap(curr.ptr);
          ClearHashEntry(i);
          return true;
        }
#endif
      }
    }

#ifdef HANDLE_COLLISIONS
    // Init with linked list traverse
    const uint bucket_last_entry_idx = bucket_first_entry_idx + HASH_BUCKET_SIZE - 1;
    int i = bucket_last_entry_idx;
    HashEntry& curr = hash_entries[i];

    int prev_idx = i;
    i = (bucket_last_entry_idx + curr.offset) % kHashParams.entry_count;

    #pragma unroll 1
    for (uint iter = 0; iter < kHashParams.linked_list_size; ++iter) {
      curr = hash_entries[i];

      if (IsBlockAllocated(block_pos, curr)) {
        int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          FreeHeap(curr.ptr);
          ClearHashEntry(i);
          HashEntry prev = hash_entries[prev_idx];
          prev.offset = curr.offset;
          hash_entries[prev_idx] = prev;
          return true;
        } else {
          return false;
        }
      }

      if (curr.offset == 0) {	//we have found the end of the list
        return false;	//should actually never happen because we need to find that guy before
      }

      prev_idx = i;
      i = (bucket_last_entry_idx + curr.offset) % kHashParams.entry_count;
    }
#endif	// HANDLE_COLLSISION
    return false;
  }

#endif	//CUDACC

};

//typedef HashTableGPU<Block> HashTable;

#endif