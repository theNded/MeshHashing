//
// Created by wei on 17-3-12.
//

/// HashTable for VoxelHashing

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
#include "hash_param.h"
#include "sensor_data.h"

#define HANDLE_COLLISIONS

#define HASH_BUCKET_SIZE  10
#define SDF_BLOCK_SIZE    8

#define LOCK_ENTRY -1
#define FREE_ENTRY -2
#define NO_OFFSET   0

/// constant.cu
extern void UpdateConstantHashParams(const HashParams &params);
extern __constant__ HashParams kHashParams;

struct HashTable {
  /// Hash VALUE part
  uint      *heap;               /// index to free blocks
  uint      *heap_counter;       /// single element; used as an atomic counter (points to the next free block)
  Voxel     *blocks;             /// pre-allocated and managed by heap manually to avoid expensive malloc
  int       *block_remove_flags; /// used in garbage collection

  /// Hash KEY part
  HashEntry *hash_entries;                 /// hash entries that stores pointers to sdf blocks
  HashEntry *compacted_hash_entries;       /// allocated for parallel computation
  int       *compacted_hash_entry_counter; /// atomic counter to add compacted entries atomically

  /// Misc
  int       *bucket_mutexes;     /// binary flag per hash bucket; used for allocation to atomically lock a bucket
  bool       is_on_gpu;          /// the class be be used on both cpu and gpu

  ///////////////
  // Host part //
  ///////////////
  __host__
  void updateParams(const HashParams &params) {
    if (is_on_gpu) {
      UpdateConstantHashParams(params);
    }
  }

  __device__ __host__
  HashTable() {
    heap = NULL;
    heap_counter = NULL;
    blocks = NULL;
    block_remove_flags = NULL;

    hash_entries = NULL;
    compacted_hash_entries = NULL;
    compacted_hash_entry_counter = NULL;

    bucket_mutexes = NULL;
    is_on_gpu = false;
  }

  __host__
  void allocate(const HashParams &params, bool is_data_on_gpu = true) {
    is_on_gpu = is_data_on_gpu;
    if (is_on_gpu) {
      checkCudaErrors(cudaMalloc(&heap, sizeof(uint) * params.block_count));
      checkCudaErrors(cudaMalloc(&heap_counter, sizeof(uint)));
      checkCudaErrors(cudaMalloc(&blocks, sizeof(Voxel) * params.voxel_count));
      checkCudaErrors(cudaMalloc(&block_remove_flags, sizeof(int) * params.entry_count));

      checkCudaErrors(cudaMalloc(&hash_entries, sizeof(HashEntry) * params.entry_count));
      checkCudaErrors(cudaMalloc(&compacted_hash_entries, sizeof(HashEntry) * params.entry_count));
      checkCudaErrors(cudaMalloc(&compacted_hash_entry_counter, sizeof(int)));

      checkCudaErrors(cudaMalloc(&bucket_mutexes, sizeof(int) * params.bucket_count));
    } else {
      heap               = new uint[params.block_count];
      heap_counter       = new uint[1];
      blocks             = new Voxel[params.voxel_count];
      block_remove_flags = new int[params.entry_count];

      hash_entries                 = new HashEntry[params.entry_count];
      compacted_hash_entries       = new HashEntry[params.entry_count];
      compacted_hash_entry_counter = new int[1];

      bucket_mutexes = new int[params.bucket_count];
    }

    updateParams(params);
  }

  __host__
  void free() {
    if (is_on_gpu) {
      checkCudaErrors(cudaFree(heap));
      checkCudaErrors(cudaFree(heap_counter));
      checkCudaErrors(cudaFree(blocks));
      checkCudaErrors(cudaFree(block_remove_flags));

      checkCudaErrors(cudaFree(hash_entries));
      checkCudaErrors(cudaFree(compacted_hash_entries));
      checkCudaErrors(cudaFree(compacted_hash_entry_counter));

      checkCudaErrors(cudaFree(bucket_mutexes));
    } else {
      if (heap)               delete[] heap;
      if (heap_counter)       delete[] heap_counter;
      if (blocks)             delete[] blocks;
      if (block_remove_flags) delete[] block_remove_flags;

      if (hash_entries)                 delete[] hash_entries;
      if (compacted_hash_entries)       delete[] compacted_hash_entries;
      if (compacted_hash_entry_counter) delete[] compacted_hash_entry_counter;

      if (bucket_mutexes) delete[] bucket_mutexes;
    }

    heap               = NULL;
    heap_counter       = NULL;
    blocks             = NULL;
    block_remove_flags = NULL;

    hash_entries                 = NULL;
    compacted_hash_entries       = NULL;
    compacted_hash_entry_counter = NULL;

    bucket_mutexes = NULL;
  }

  __host__
  HashTable CopyToCPU(const HashParams &params) const {
    HashTable hash_table;
    hash_table.allocate(params, false);

    checkCudaErrors(cudaMemcpy(hash_table.heap, heap,
                               sizeof(uint) * params.block_count,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hash_table.heap_counter, heap_counter,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hash_table.blocks, blocks,
                               sizeof(Voxel) * params.voxel_count,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hash_table.block_remove_flags, block_remove_flags,
                               sizeof(int) * params.entry_count,
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(hash_table.hash_entries, hash_entries,
                               sizeof(HashEntry) * params.entry_count,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hash_table.compacted_hash_entries, compacted_hash_entries,
                               sizeof(HashEntry) * params.entry_count,
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(hash_table.compacted_hash_entry_counter, compacted_hash_entry_counter,
                               sizeof(unsigned int),
                               cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaMemcpy(hash_table.bucket_mutexes, bucket_mutexes,
                               sizeof(int) * params.bucket_count,
                               cudaMemcpyDeviceToHost));

    // TODO MATTHIAS look at this (i.e,. when does memory get destroyed ;
    // if it's in the destructer it would kill everything here
    return hash_table;
  }

  /////////////////
  // Device part //
  /////////////////
#ifdef __CUDACC__
  __device__
  const HashParams& params() const {
    return kHashParams;
  }

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

  __device__
  void combineVoxel(const Voxel &in, const Voxel& update, Voxel &out) const {
    float3 c_in     = make_float3(in.color.x, in.color.y, in.color.z);
    float3 c_update = make_float3(update.color.x, update.color.y, update.color.z);

    float3 c_out = 0.5f * c_in + 0.5f * c_update;

    out.color = make_uchar3(c_out.x + 0.5f, c_out.y + 0.5f, c_out.z + 0.5f);
    out.sdf = (in.sdf * (float)in.weight + update.sdf * (float)update.weight)
            / ((float)in.weight + (float)update.weight);
    out.weight = min(kHashParams.weight_upper_bound, (uint)in.weight + (uint)update.weight);
  }


  //! returns the truncation of the SDF for a given distance value
  __device__
  float getTruncation(float z) const {
    return kHashParams.truncation_distance
         + kHashParams.truncation_distance_scale * z;
  }


  ///////////////////////////////////////////////////
  /// Transforms

  /// float is only used to do interpolation
  /// Semantic: A pos To B pos; A, B in {world, voxel, block}
  __device__
  float3 WorldToVoxelf(const float3& world_pos) const	{
    return world_pos / kHashParams.voxel_size;
  }
  __device__
  int3 WorldToVoxeli(const float3& world_pos) const {
    const float3 p = world_pos / kHashParams.voxel_size;
    return make_int3(p + make_float3(sign(p)) * 0.5f);
  }

  __device__
  int3 VoxelToBlock(int3 voxel_pos) const {
    if (voxel_pos.x < 0) voxel_pos.x -= SDF_BLOCK_SIZE-1;
    if (voxel_pos.y < 0) voxel_pos.y -= SDF_BLOCK_SIZE-1;
    if (voxel_pos.z < 0) voxel_pos.z -= SDF_BLOCK_SIZE-1;

    return make_int3(
      voxel_pos.x / SDF_BLOCK_SIZE,
      voxel_pos.y / SDF_BLOCK_SIZE,
      voxel_pos.z / SDF_BLOCK_SIZE);
  }

  // corner voxel with smallest xyz
  __device__
  int3 BlockToVoxel(const int3& block_pos) const {
    return block_pos * SDF_BLOCK_SIZE;
  }

  __device__
  float3 VoxelToWorld(const int3& voxel_pos) const {
    return make_float3(voxel_pos) * kHashParams.voxel_size;
  }

  __device__
  float3 BlockToWorld(const int3& block_pos) const {
    return VoxelToWorld(BlockToVoxel(block_pos));
  }

  __device__
  int3 WorldToBlock(const float3& world_pos) const {
    return VoxelToBlock(WorldToVoxeli(world_pos));
  }

  // TODO: wei, a better implementation?
  __device__
  bool IsBlockInCameraFrustum(const int3& block_pos) {
    float3 world_pos = VoxelToWorld(BlockToVoxel(block_pos)) + kHashParams.voxel_size * 0.5f * (SDF_BLOCK_SIZE - 1.0f);
    return DepthCameraData::isInCameraFrustumApprox(kHashParams.m_rigidTransformInverse, world_pos);
  }

  /// Idx means local idx inside a block \in [0, 511]
  __device__
  uint3 IdxToVoxelLocalPos(uint idx) const	{
    uint x = idx % SDF_BLOCK_SIZE;
    uint y = (idx % (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE)) / SDF_BLOCK_SIZE;
    uint z = idx / (SDF_BLOCK_SIZE * SDF_BLOCK_SIZE);
    return make_uint3(x, y, z);
  }

  //! computes the linearized index of a local virtual voxel pos; pos in [0;7]^3
  __device__
  uint VoxelLocalPosToIdx(const int3& voxel_local_pos) const {
    return
      voxel_local_pos.z * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE +
      voxel_local_pos.y * SDF_BLOCK_SIZE +
      voxel_local_pos.x;
  }

  __device__
  int VoxelPosToIdx(const int3& voxel_pos) const	{
    int3 voxel_local_pos = make_int3(
      voxel_pos.x % SDF_BLOCK_SIZE,
      voxel_pos.y % SDF_BLOCK_SIZE,
      voxel_pos.z % SDF_BLOCK_SIZE);

    if (voxel_local_pos.x < 0) voxel_local_pos.x += SDF_BLOCK_SIZE;
    if (voxel_local_pos.y < 0) voxel_local_pos.y += SDF_BLOCK_SIZE;
    if (voxel_local_pos.z < 0) voxel_local_pos.z += SDF_BLOCK_SIZE;

    return VoxelLocalPosToIdx(voxel_local_pos);
  }

  __device__
  int WorldPosToIdx(const float3& world_pos) const	{
    int3 voxel_pos = WorldToVoxeli(world_pos);
    return VoxelPosToIdx(voxel_pos);
  }

  ////////////////////////////////////////
  /// Access
  __device__
  HashEntry GetHashEntryForWorldPos(const float3& world_pos) const	{
    int3 block_pos = WorldToBlock(world_pos);
    return GetHashEntryForBlockPos(block_pos);
  }

  __device__
    void DeleteHashEntry(uint id) {
      DeleteHashEntry(hash_entries[id]);
  }

  __device__
  void DeleteHashEntry(HashEntry& hash_entry) {
    hash_entry.pos    = make_int3(0);
    hash_entry.offset = 0;
    hash_entry.ptr    = FREE_ENTRY;
  }

  __device__
  bool IsBlockAllocated(const float3& world_pos) const {
    HashEntry hash_entry = GetHashEntryForWorldPos(world_pos);
    return (hash_entry.ptr != FREE_ENTRY);
  }

  __device__
  bool IsBlockAllocated(const int3& block_pos, const HashEntry& hash_entry) const {
    return block_pos.x == hash_entry.pos.x
        && block_pos.y == hash_entry.pos.y
        && block_pos.z == hash_entry.pos.z
        && hash_entry.ptr != FREE_ENTRY;
  }

  __device__
  void DeleteVoxel(Voxel& v) const {
    v.color  = make_uchar3(0, 0, 0);
    v.weight = 0;
    v.sdf    = 0.0f;
  }
  __device__
  void DeleteVoxel(uint id) {
    DeleteVoxel(blocks[id]);
  }

  __device__
  Voxel GetVoxel(const float3& world_pos) const	{
    HashEntry hash_entry = GetHashEntryForWorldPos(world_pos);
    Voxel v;
    if (hash_entry.ptr == FREE_ENTRY) {
      DeleteVoxel(v);
    } else {
      int3 voxel_pos = WorldToVoxeli(world_pos);
      v = blocks[hash_entry.ptr + VoxelPosToIdx(voxel_pos)];
    }
    return v;
  }

  __device__
  Voxel GetVoxel(const int3& voxel_pos) const	{
    HashEntry hash_entry = GetHashEntryForBlockPos(VoxelToBlock(voxel_pos));
    Voxel v;
    if (hash_entry.ptr == FREE_ENTRY) {
      DeleteVoxel(v);
    } else {
      v = blocks[hash_entry.ptr + VoxelPosToIdx(voxel_pos)];
    }
    return v;
  }

  __device__
  void SetVoxel(const int3& voxel_pos, Voxel& voxel) const {
    HashEntry hash_entry = GetHashEntryForBlockPos(VoxelToBlock(voxel_pos));
    if (hash_entry.ptr != FREE_ENTRY) {
      blocks[hash_entry.ptr + VoxelPosToIdx(voxel_pos)] = voxel;
    }
  }

  __device__
  HashEntry GetHashEntryForBlockPos(const int3& block_pos) const {
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
	void AllocBlock(const int3& pos) {
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
				entry.ptr    = AllocHeap() * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;	//memory alloc
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
						entry.ptr    = AllocHeap() * SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;	//memory alloc

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
  bool DeleteHashEntryElement(const int3& block_pos) {
    uint bucket_idx = HashBucketForBlockPos(block_pos);	//hash bucket
    uint bucket_first_entry_idx = bucket_idx * HASH_BUCKET_SIZE;		//hash position

    for (uint j = 0; j < HASH_BUCKET_SIZE; j++) {
      uint i = j + bucket_first_entry_idx;
      const HashEntry& curr = hash_entries[i];
      if (IsBlockAllocated(block_pos, curr)) {

#ifndef HANDLE_COLLISIONS
        const uint block_size = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
        FreeHeap(curr.ptr / block_size);
        DeleteHashEntry(i);
        return true;
#else
        // Deal with linked list: curr = curr->next
        if (curr.offset != 0) {
          int lock = atomicExch(&bucket_mutexes[bucket_idx], LOCK_ENTRY);
          if (lock != LOCK_ENTRY) {
            const uint block_size = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
            FreeHeap(curr.ptr / block_size);
            int next_idx = (i + curr.offset) % (kHashParams.entry_count);
            hash_entries[i] = hash_entries[next_idx];
            DeleteHashEntry(next_idx);
            return true;
          } else {
            return false;
          }
        } else {
          const uint block_size = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
          FreeHeap(curr.ptr / block_size);
          DeleteHashEntry(i);
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
          const uint block_size = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
          FreeHeap(curr.ptr / block_size);
          DeleteHashEntry(i);
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

  /// (wei): reported bug on GitHub, wait for responce)
  //  TODO MATTHIAS check the atomics in this function
  //!inserts a hash entry without allocating any memory: used by streaming:
  __device__
  bool insertHashEntry(HashEntry entry) {
		uint h  = HashBucketForBlockPos(entry.pos); //hash bucket
		uint hp = h * HASH_BUCKET_SIZE;	//hash position

    for (uint j = 0; j < HASH_BUCKET_SIZE; ++j) {
      uint i = j + hp;
      int lock = atomicCAS(&hash_entries[i].ptr, FREE_ENTRY, LOCK_ENTRY);
      if (lock == FREE_ENTRY) {
        hash_entries[i] = entry;
        return true;
      }
    }

#ifdef HANDLE_COLLISIONS
    const uint idxLastEntryInBucket = hp + HASH_BUCKET_SIZE - 1;
    uint i = idxLastEntryInBucket;

    #pragma  unroll 1
    for (uint maxIter = 0; maxIter < kHashParams.linked_list_size; ++maxIter) {
      //curr = GetHashEntry(hash, i);
      HashEntry curr = hash_entries[i];	//TODO MATTHIAS do by reference
      if (curr.offset == 0) break;									//we have found the end of the list
      i = idxLastEntryInBucket + curr.offset;							//go to next element in the list
      i %= (HASH_BUCKET_SIZE * kHashParams.bucket_count);	//check for overflow
    }

    uint maxIter = 0;
    int offset = 0;
    #pragma  unroll 1
    while (maxIter < kHashParams.linked_list_size) {													//linear search for free entry
      offset++;
      uint i = (idxLastEntryInBucket + offset) % (HASH_BUCKET_SIZE * kHashParams.bucket_count);	//go to next hash element
      if ((offset % HASH_BUCKET_SIZE) == 0) continue;										//cannot insert into a last bucket element (would conflict with other linked lists)

      int prevWeight = 0;
      //InterlockedCompareExchange(hash[3*i+2], FREE_ENTRY, LOCK_ENTRY, prevWeight);		//check for a free entry
      uint* hash_entriesUI = (uint*)hash_entries;
      prevWeight = prevWeight = atomicCAS(&hash_entriesUI[3*idxLastEntryInBucket+1], (uint)FREE_ENTRY, (uint)LOCK_ENTRY);
      if (prevWeight == FREE_ENTRY) {														//if free entry found set prev->next = curr & curr->next = prev->next
        //[allow_uav_condition]
        //while(hash[3*idxLastEntryInBucket+2] == LOCK_ENTRY); // expects setHashEntry to set the ptr last, required because pos.z is packed into the same value -> prev->next = curr -> might corrput pos.z

        HashEntry lastEntryInBucket = hash_entries[idxLastEntryInBucket];			//get prev (= lastEntry in Bucket)

        int newOffsetPrev = (offset << 16) | (lastEntryInBucket.pos.z & 0x0000ffff);	//prev->next = curr (maintain old z-pos)
        int oldOffsetPrev = 0;
        //InterlockedExchange(hash[3*idxLastEntryInBucket+1], newOffsetPrev, oldOffsetPrev);	//set prev offset atomically
        uint* hash_entriesUI = (uint*)hash_entries;
        oldOffsetPrev = prevWeight = atomicExch(&hash_entriesUI[3*idxLastEntryInBucket+1], newOffsetPrev);
        entry.offset = oldOffsetPrev >> 16;													//remove prev z-pos from old offset

        //setHashEntry(hash, i, entry);														//sets the current hashEntry with: curr->next = prev->next
        hash_entries[i] = entry;
        return true;
      }

      maxIter++;
    }
#endif

    return false;
  }

  //////////
  // Histogram (no collision traversal)
  __device__
  unsigned int getNumHashEntriesPerBucket(unsigned int bucketID) {
    unsigned int h = 0;
    for (uint i = 0; i < HASH_BUCKET_SIZE; i++) {
      if (hash_entries[bucketID*HASH_BUCKET_SIZE+i].ptr != FREE_ENTRY) {
        h++;
      }
    }
    return h;
  }

  // Histogram (collisions traversal only)
  __device__
  unsigned int getNumHashLinkedList(unsigned int bucketID) {
    unsigned int listLen = 0;

#ifdef HANDLE_COLLISIONS
    const uint idxLastEntryInBucket = (bucketID+1)*HASH_BUCKET_SIZE - 1;
    unsigned int i = idxLastEntryInBucket;	//start with the last entry of the current bucket
    //int offset = 0;
    HashEntry curr;	curr.offset = 0;
    //traverse list until end: memorize idx at list end and memorize offset from last element of bucket to list end

    unsigned int maxIter = 0;
    uint g_MaxLoopIterCount = kHashParams.linked_list_size;
    #pragma unroll 1
    while (maxIter < g_MaxLoopIterCount) {
      //offset = curr.offset;
      //curr = GetHashEntry(g_Hash, i);
      curr = hash_entries[i];

      if (curr.offset == 0) {	//we have found the end of the list
        break;
      }
      i = idxLastEntryInBucket + curr.offset;		//go to next element in the list
      i %= (HASH_BUCKET_SIZE * kHashParams.bucket_count);	//check for overflow
      listLen++;

      maxIter++;
    }
#endif
    return listLen;
  }


#endif	//CUDACC

};

#endif