//
// Created by wei on 17-4-5.
//

#ifndef VOXEL_HASHING_MAP_H
#define VOXEL_HASHING_MAP_H

#include "hash_table.h"
#include "sensor.h"
#include "sensor_data.h"

/// CUDA functions
/// @hash_table  is used for CUDA computation
/// @hash_params is used for kernel management by the host (CPU)
extern uint GenerateCompressedHashEntriesCudaHost(
        HashTableGPU<Block>& hash_table, const HashParams& hash_params,
        float4x4 c_T_w
);

/// Garbage collection
extern void StarveOccupiedVoxelsCudaHost(
        HashTableGPU<Block>& hash_table, const HashParams& hash_params
);
extern void CollectInvalidBlockInfoCudaHost(
        HashTableGPU<Block>& hash_table, const HashParams& hash_params
);
extern void RecycleInvalidBlockCudaHost(
        HashTableGPU<Block>& hash_table, const HashParams& hash_params
);

class Map {
public:

  Map(const HashParams& hash_params);
  ~Map();

  void Reset();
  void GenerateCompressedHashEntries(float4x4 c_T_w);
  void RecycleInvalidBlocks();

  HashTableGPU<Block> &hash_table() {
    return hash_table_.gpu_data();
  }

  HashParams &hash_params() {
    return hash_params_;
  }
  uint& frame_count() {
    return integrated_frame_count_;
  }
  const uint& occupied_block_count() const {
    return occupied_block_count_;
  }

//! debug only!
  unsigned int getHeapFreeCount();
  void debugHash();

  HashTable<Block> hash_table_;

private:
  HashParams hash_params_;
  uint integrated_frame_count_;
  uint occupied_block_count_;

};


#endif //VOXEL_HASHING_MAP_H
