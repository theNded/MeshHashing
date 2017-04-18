//
// Created by wei on 17-4-5.
//

#ifndef VOXEL_HASHING_MAP_H
#define VOXEL_HASHING_MAP_H

#include "hash_table.h"
#include "sensor.h"
#include "sensor_data.h"

/// CUDA functions
extern void ResetCudaHost(HashTable& hash_table,
                          const HashParams& hash_params);
extern void ResetBucketMutexesCudaHost(HashTable& hash_table,
                                       const HashParams& hash_params);
extern void AllocBlocksCudaHost(
        HashTable& hash_table,
        const HashParams& hash_params,
        const SensorData& sensor_data,
        const SensorParams& sensor_params,
        const float4x4& w_T_c,
        const unsigned int* d_bitMask);
extern unsigned int GenerateCompressedHashEntriesCudaHost(
        HashTable& hash_table,
        const HashParams& hash_params,
        float4x4 c_T_w);

/// Garbage collection
extern void StarveOccupiedVoxelsCudaHost(HashTable& hash_table, const HashParams& hash_params);
extern void CollectInvalidBlockInfoCudaHost(HashTable& hash_table, const HashParams& hash_params);
extern void RecycleInvalidBlockCudaHost(HashTable& hash_table, const HashParams& hash_params);

class Map {
public:

  Map(const HashParams& hash_params);
  ~Map();

  void Reset();
  void AllocBlocks(Sensor* sensor);
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

//! debug only!
  unsigned int getHeapFreeCount();
  void debugHash();

private:
  HashParams hash_params_;
  HashTable  hash_table_;
  uint integrated_frame_count_;
};


#endif //VOXEL_HASHING_MAP_H
