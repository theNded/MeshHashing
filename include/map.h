//
// Created by wei on 17-4-5.
//

#ifndef VOXEL_HASHING_MAP_H
#define VOXEL_HASHING_MAP_H

#include "hash_table.h"
#include "sensor_data.h"

/// CUDA functions
extern void resetCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void resetHashBucketMutexCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void allocCUDA(HashTable& hash_table, const HashParams& hash_params, const SensorData& sensor_data, const SensorParams& depthCameraParams, const unsigned int* d_bitMask);

extern unsigned int compactifyHashAllInOneCUDA(HashTable& hash_table, const HashParams& hash_params);


/// Garbage collection
extern void starveVoxelsKernelCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void garbageCollectIdentifyCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void garbageCollectFreeCUDA(HashTable& hash_table, const HashParams& hash_params);

class Map {
private:
  HashParams hash_params_;
  HashTable  hash_table_;

  uint integrated_frame_count_;

public:
  Map(const HashParams& hash_params);
  ~Map();
  void Reset();

  void AllocBlocks(const SensorData &sensor_data,
                   const SensorParams& sensor_params);
  void GenerateCompressedHashEntries();
  void RecycleInvalidBlocks();

  HashTable &hash_table() {
    return hash_table_;
  }
};


#endif //VOXEL_HASHING_MAP_H
