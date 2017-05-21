//
// Created by wei on 17-4-28.
//
// Host (CPU) wrapper for the hash table data allocated on the GPU
// Host operations: alloc, free, and reset

#ifndef VH_HASH_TABLE_H
#define VH_HASH_TABLE_H

#include "common.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include "core.h"
#include "geometry_util.h"
#include "params.h"

struct __ALIGN__(8) HashEntry {
  int3	pos;		   // hash position (lower left corner of SDFBlock))
  int		ptr;	     // pointer into heap to SDFBlock
  uint	offset;		 // offset for collisions

  // uint padding

  __device__
  void operator=(const struct HashEntry& e) {
    ((long long*)this)[0] = ((const long long*)&e)[0];
    ((long long*)this)[1] = ((const long long*)&e)[1];
    ((int*)this)[4]       = ((const int*)&e)[4];
  }

  __device__
  void Clear() {
    pos    = make_int3(0);
    offset = 0;
    ptr    = FREE_ENTRY;
  }
};

struct HashTableGPU {
  /// Hash VALUE part
  uint      *heap;                    /// index to free values
  uint      *heap_counter;            /// single element; used as an atomic counter (points to the next free block)
  int       *hash_entry_remove_flags; /// used in garbage collection

  /// Hash KEY part
  HashEntry *hash_entries;                 /// hash entries that stores pointers to sdf values
  HashEntry *compacted_hash_entries;       /// allocated for parallel computation
  int       *compacted_hash_entry_counter; /// atomic counter to add compacted entries atomically
  /// == occupied_block_count
  /// Misc
  int       *bucket_mutexes;     /// binary flag per hash bucket; used for allocation to atomically lock a bucket

  /// Parameters
  uint      *bucket_count;
  uint      *bucket_size;
  uint      *entry_count;
  uint      *value_capacity;
  uint      *linked_list_size;

  bool       is_on_gpu;          /// the class be be used on both cpu and gpu

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

    int res = ((block_pos.x * p0) ^ (block_pos.y * p1) ^ (block_pos.z * p2))
            % (*bucket_count);
    if (res < 0) res += (*bucket_count);
    return (uint) res;
  }


  ////////////////////////////////////////
  /// Access
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
    uint bucket_first_entry_idx = bucket_idx * (*bucket_size);

    HashEntry entry;
    entry.pos    = block_pos;
    entry.offset = 0;
    entry.ptr    = FREE_ENTRY;

    for (uint i = 0; i < (*bucket_size); ++i) {
      HashEntry curr_entry = hash_entries[i + bucket_first_entry_idx];
      if (IsBlockAllocated(block_pos, curr_entry)) {
        return curr_entry;
      }
    }

    /// The last entry is visted twice, but its OK
#ifdef HANDLE_COLLISIONS
    const uint bucket_last_entry_idx = bucket_first_entry_idx + (*bucket_size) - 1;
    int i = bucket_last_entry_idx;
    HashEntry curr_entry;

    #pragma unroll 1
    for (uint iter = 0; iter < *linked_list_size; ++iter) {
      curr_entry = hash_entries[i];

      if (IsBlockAllocated(block_pos, curr_entry)) {
        return curr_entry;
      }
      if (curr_entry.offset == 0) { // should never reach here
        break;
      }
      i = bucket_last_entry_idx + curr_entry.offset;
      i %= (*entry_count); // avoid overflow
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
    uint bucket_first_entry_idx = bucket_idx * (*bucket_size);	//hash position

    int empty_entry_idx = -1;
    for (uint j = 0; j < (*bucket_size); j++) {
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
    const uint bucket_last_entry_idx = bucket_first_entry_idx + (*bucket_size) - 1;
    uint i = bucket_last_entry_idx;
    int offset = 0;
    for (uint iter = 0; iter < *linked_list_size; ++iter) {
      HashEntry& curr_entry = hash_entries[i];
      if (IsBlockAllocated(pos, curr_entry)) {
        return;
      }
      if (curr_entry.offset == 0) {
        break;
      }
      i = (bucket_last_entry_idx + curr_entry.offset) % (*entry_count);
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
    for (uint iter = 0; iter < *linked_list_size; ++iter) {
      offset ++;
      if ((offset % (*bucket_size)) == 0) continue;

      i = (bucket_last_entry_idx + offset) % (*entry_count);

      HashEntry& curr_entry = hash_entries[i];

      if (curr_entry.ptr == FREE_ENTRY) {	//this is the first free entry
        int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          // Not use reference in order to avoid lock ?
          HashEntry& bucket_last_entry = hash_entries[bucket_last_entry_idx];
          uint alloc_bucket_idx = i / (*bucket_size);

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
    uint bucket_first_entry_idx = bucket_idx * (*bucket_size);		//hash position

    for (uint j = 0; j < (*bucket_size); j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr = hash_entries[i];
      if (IsBlockAllocated(block_pos, curr)) {

#ifndef HANDLE_COLLISIONS
        FreeHeap(curr.ptr);
        hash_entries[i].Clear();
        return true;
#else
        // Deal with linked list: curr = curr->next
        if (curr.offset != 0) {
          int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            FreeHeap(curr.ptr);
            int next_idx = (i + curr.offset) % (*entry_count);
            hash_entries[i] = hash_entries[next_idx];
            hash_entries[next_idx].Clear();
            return true;
          } else {
            return false;
          }
        } else {
          FreeHeap(curr.ptr);
          hash_entries[i].Clear();
          return true;
        }
#endif
      }
    }

#ifdef HANDLE_COLLISIONS
    // Init with linked list traverse
    const uint bucket_last_entry_idx = bucket_first_entry_idx + (*bucket_size) - 1;
    int i = bucket_last_entry_idx;
    HashEntry& curr = hash_entries[i];

    int prev_idx = i;
    i = (bucket_last_entry_idx + curr.offset) % (*entry_count);

    #pragma unroll 1
    for (uint iter = 0; iter < *linked_list_size; ++iter) {
      curr = hash_entries[i];

      if (IsBlockAllocated(block_pos, curr)) {
        int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          FreeHeap(curr.ptr);
          hash_entries[i].Clear();
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
      i = (bucket_last_entry_idx + curr.offset) % (*entry_count);
    }
#endif	// HANDLE_COLLSISION
    return false;
  }
#endif	//CUDACC

};

/// Generally, the template should be implemented entirely in a header
/// However, we need CUDA code that has to be in .cu
/// Hence, we separate the declaration and implementation
/// And specifically instantiate it with @typename Block in the .cu
class HashTable {
private:
  HashTableGPU gpu_data_;
  HashParams hash_params_;

  void Alloc(const HashParams &params);
  void Free();

public:
  HashTable();
  HashTable(const HashParams &params);
  ~HashTable();

  uint compacted_entry_count();
  void Resize(const HashParams &params);
  void Reset();
  void ResetMutexes();

  void CollectAllEntries();

  void Debug();

  HashTableGPU& gpu_data() {
    return gpu_data_;
  }
};

#endif //VH_HASH_TABLE_H
