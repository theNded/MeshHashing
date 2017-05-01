//
// Created by wei on 17-4-5.
//
// Map: managing HashTable<VoxelBlock> and might be other structs later

#ifndef VH_MAP_H
#define VH_MAP_H

#include "hash_table.h"
#include "sensor.h"

class Map {
private:
  HashTable<VoxelBlock> hash_table_;
  uint integrated_frame_count_;

  /// Garbage collection
  void StarveOccupiedVoxels();
  void CollectInvalidBlockInfo();
  void RecycleInvalidBlock();

public:
  Map(const HashParams& hash_params);
  ~Map();

  void Reset();
  void Recycle();

  /// Only classes with Kernel function should call it
  /// The other part of the hash_table should be hidden
  HashTable<VoxelBlock> &hash_table() {
    return hash_table_;
  }
  HashTableGPU<VoxelBlock> &gpu_data() {
    return hash_table_.gpu_data();
  }
  uint& frame_count() {
    return integrated_frame_count_;
  }
  void Debug() {
    hash_table_.Debug();
  }
};


#endif //VH_MAP_H
