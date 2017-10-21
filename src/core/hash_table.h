//
// Created by wei on 17-4-28.
//

#ifndef VH_HASH_TABLE_H
#define VH_HASH_TABLE_H

#include "helper_cuda.h"
#include "helper_math.h"

#include "core/common.h"
#include "core/params.h"
#include "core/hash_entry.h"
#include "geometry/coordinate_utils.h"

class HashTable {
public:
  /// Parameters
  uint      bucket_count;
  uint      bucket_size;
  uint      entry_count;
  uint      value_capacity;
  uint      linked_list_size;

  __host__ HashTable();
  __host__ explicit HashTable(const HashParams &params);
  // ~HashTable();
  __host__ void Alloc(const HashParams &params);
  __host__ void Free();

  __host__ void Resize(const HashParams &params);
  __host__ void Reset();
  __host__ void ResetMutexes();

  __host__ __device__ HashEntry& entry(uint i) {
    return entries_[i];
  }
  //__host__ void Debug();

  /////////////////
  // Device part //

private:
  // @param array
  uint      *heap_;             /// index to free values
  // @param read-write element
  uint      *heap_counter_;     /// single element; used as an atomic counter (points to the next free block)

  // @param array
  HashEntry *entries_;          /// hash entries that stores pointers to sdf values
  // @param array
  int       *bucket_mutexes_;   /// binary flag per hash bucket; used for allocation to atomically lock a bucket

#ifdef __CUDACC__
public:
  __device__
  HashEntry GetEntry(const int3& pos) const {
    uint bucket_idx             = HashBucketForBlockPos(pos);
    uint bucket_first_entry_idx = bucket_idx * bucket_size;

    HashEntry entry;
    entry.pos    = pos;
    entry.offset = 0;
    entry.ptr    = FREE_ENTRY;

    for (uint i = 0; i < bucket_size; ++i) {
      HashEntry curr_entry = entries_[i + bucket_first_entry_idx];
      if (IsPosAllocated(pos, curr_entry)) {
        return curr_entry;
      }
    }

    /// The last entry is visted twice, but it's OK
#ifdef HANDLE_COLLISIONS
    const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
    int i = bucket_last_entry_idx;

    #pragma unroll 1
    for (uint iter = 0; iter < linked_list_size; ++iter) {
      HashEntry curr_entry = entries_[i];

      if (IsPosAllocated(pos, curr_entry)) {
        return curr_entry;
      }
      if (curr_entry.offset == 0) {
        break;
      }
      i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
    }
#endif
    return entry;
  }

  //pos in SDF block coordinates
  __device__
  void AllocEntry(const int3& pos) {
    uint bucket_idx             = HashBucketForBlockPos(pos);		//hash bucket
    uint bucket_first_entry_idx = bucket_idx * bucket_size;	//hash position

    /// 1. Try GetEntry, meanwhile collect an empty entry potentially suitable
    int empty_entry_idx = -1;
    for (uint j = 0; j < bucket_size; j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr_entry = entries_[i];
      if (IsPosAllocated(pos, curr_entry)) {
        return;
      }

      /// wei: should not break and alloc before a thorough searching is over:
      if (empty_entry_idx == -1 && curr_entry.ptr == FREE_ENTRY) {
        empty_entry_idx = i;
      }
    }

#ifdef HANDLE_COLLISIONS
    const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
    uint i = bucket_last_entry_idx;
    for (uint iter = 0; iter < linked_list_size; ++iter) {
      HashEntry curr_entry = entries_[i];

      if (IsPosAllocated(pos, curr_entry)) {
        return;
      }
      if (curr_entry.offset == 0) {
        break;
      }
      i = (bucket_last_entry_idx + curr_entry.offset) % (entry_count);
    }
#endif

    /// 2. NOT FOUND, Allocate
    if (empty_entry_idx != -1) {
      int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
      if (lock != LOCK_ENTRY) {
        HashEntry& entry = entries_[empty_entry_idx];
        entry.pos    = pos;
        entry.ptr    = Alloc();
        entry.offset = NO_OFFSET;
      }
      return;
    }

#ifdef HANDLE_COLLISIONS
    i = bucket_last_entry_idx;
    int offset = 0;

    #pragma  unroll 1
    for (uint iter = 0; iter < linked_list_size; ++iter) {
      offset ++;
      if ((offset % bucket_size) == 0) continue;

      i = (bucket_last_entry_idx + offset) % (entry_count);

      HashEntry& curr_entry = entries_[i];

      if (curr_entry.ptr == FREE_ENTRY) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          HashEntry& bucket_last_entry = entries_[bucket_last_entry_idx];
          uint alloc_bucket_idx = i / bucket_size;

          lock = atomicExch(&bucket_mutexes_[alloc_bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            HashEntry& entry = entries_[i];
            entry.pos    = pos;
            entry.offset = bucket_last_entry.offset; // pointer assignment in linked list
            entry.ptr    = Alloc();	//memory alloc

            // Not sure if it is ok to directly assign to reference
            bucket_last_entry.offset = offset;
            entries_[bucket_last_entry_idx] = bucket_last_entry;
          }
        }
        return;	//bucket was already locked
      }
    }
#endif
  }

  //! deletes a hash entry position for a given pos index
  // returns true uppon successful deletion; otherwise returns false
  __device__
  bool FreeEntry(const int3& pos) {
    uint bucket_idx = HashBucketForBlockPos(pos);	//hash bucket
    uint bucket_first_entry_idx = bucket_idx * bucket_size;		//hash position

    for (uint j = 0; j < bucket_size; j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr = entries_[i];
      if (IsPosAllocated(pos, curr)) {

#ifndef HANDLE_COLLISIONS
        Free(curr.ptr);
        entries_[i].Clear();
        return true;
#else
        // Deal with linked list: curr = curr->next
        if (curr.offset != 0) {
          int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            Free(curr.ptr);
            int next_idx = (i + curr.offset) % (entry_count);
            entries_[i] = entries_[next_idx];
            entries_[next_idx].Clear();
            return true;
          } else {
            return false;
          }
        } else {
          Free(curr.ptr);
          entries_[i].Clear();
          return true;
        }
#endif
      }
    }

#ifdef HANDLE_COLLISIONS
    // Init with linked list traverse
    const uint bucket_last_entry_idx = bucket_first_entry_idx + bucket_size - 1;
    int i = bucket_last_entry_idx;
    HashEntry& curr = entries_[i];

    int prev_idx = i;
    i = (bucket_last_entry_idx + curr.offset) % (entry_count);

    #pragma unroll 1
    for (uint iter = 0; iter < linked_list_size; ++iter) {
      curr = entries_[i];

      if (IsPosAllocated(pos, curr)) {
        int lock = atomicExch(&bucket_mutexes_[bucket_idx], LOCK_ENTRY);
        if (lock != LOCK_ENTRY) {
          Free(curr.ptr);
          entries_[i].Clear();
          HashEntry prev = entries_[prev_idx];
          prev.offset = curr.offset;
          entries_[prev_idx] = prev;
          return true;
        } else {
          return false;
        }
      }

      if (curr.offset == 0) {	//we have found the end of the list
        return false;	//should actually never happen because we need to find that guy before
      }

      prev_idx = i;
      i = (bucket_last_entry_idx + curr.offset) % (entry_count);
    }
#endif	// HANDLE_COLLSISION
    return false;
  }

private:
  //! see Teschner et al. (but with correct prime values)
  __device__
  uint HashBucketForBlockPos(const int3& pos) const {
    const int p0 = 73856093;
    const int p1 = 19349669;
    const int p2 = 83492791;

    int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
              % bucket_count;
    if (res < 0) res += bucket_count;
    return (uint) res;
  }

  __device__
  bool IsPosAllocated(const int3& pos, const HashEntry& hash_entry) const {
    return pos.x == hash_entry.pos.x
        && pos.y == hash_entry.pos.y
        && pos.z == hash_entry.pos.z
        && hash_entry.ptr != FREE_ENTRY;
  }

  __device__
  uint Alloc() {
    uint addr = atomicSub(&heap_counter_[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("Memory nearly exhausted! %d -> %d\n", addr, heap_[addr]);
    }
    return heap_[addr];
  }

  __device__
  void Free(uint ptr) {
    uint addr = atomicAdd(&heap_counter_[0], 1);
    heap_[addr + 1] = ptr;
  }
#endif
};

#endif //VH_HASH_TABLE_H
