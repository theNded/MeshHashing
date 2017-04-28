//
// Created by wei on 17-4-5.
//

#ifndef VOXEL_HASHING_MAP_H
#define VOXEL_HASHING_MAP_H

#include "hash_table_gpu.h"
#include "sensor.h"
#include "sensor_data.h"

/// CUDA functions
/// @hash_table  is used for CUDA computation
/// @hash_params is used for kernel management by the host (CPU)
extern void ResetCudaHost(
        HashTable& hash_table, const HashParams& hash_params
);
extern void ResetBucketMutexesCudaHost(
        HashTable& hash_table, const HashParams& hash_params
);
extern uint GenerateCompressedHashEntriesCudaHost(
        HashTable& hash_table, const HashParams& hash_params,
        float4x4 c_T_w
);

/// Garbage collection
extern void StarveOccupiedVoxelsCudaHost(
        HashTable& hash_table, const HashParams& hash_params
);
extern void CollectInvalidBlockInfoCudaHost(
        HashTable& hash_table, const HashParams& hash_params
);
extern void RecycleInvalidBlockCudaHost(
        HashTable& hash_table, const HashParams& hash_params
);

class Map {
public:

  Map(const HashParams& hash_params);
  ~Map();

  void Reset();
  void GenerateCompressedHashEntries(float4x4 c_T_w);
  void RecycleInvalidBlocks();

  HashTable &hash_table() {
    return hash_table_;
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

private:
  HashParams hash_params_;
  HashTable  hash_table_;
  uint integrated_frame_count_;
  uint occupied_block_count_;

};


#endif //VOXEL_HASHING_MAP_H
