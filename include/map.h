//
// Created by wei on 17-4-5.
//

#ifndef VOXEL_HASHING_MAP_H
#define VOXEL_HASHING_MAP_H

#include "hash_table.h"
#include "sensor.h"

class Map {

private:
  HashParams   hash_params_;

  uint integrated_frame_count_;
  uint occupied_block_count_;

  /// Garbage collection
  void StarveOccupiedVoxels();
  void CollectInvalidBlockInfo();
  void RecycleInvalidBlock();

public:
  Map(const HashParams& hash_params);
  ~Map();

  void Reset();
  void Recycle();

  void CompactHashEntries(float4x4 c_T_w);

  HashTableGPU<Block> &hash_table() {
    return hash_table_.gpu_data();
  }
  uint& frame_count() {
    return integrated_frame_count_;
  }
  const uint& occupied_block_count() const {
    return occupied_block_count_;
  }

  SensorParams sensor_params_;

  HashTable<Block> hash_table_;
};


#endif //VOXEL_HASHING_MAP_H
