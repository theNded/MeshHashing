//
// Created by wei on 17-4-5.
//

#include "map.h"

Map::Map(const HashParams &hash_params) {
  hash_params_ = hash_params;
  hash_table_.Alloc(hash_params_);

  Reset();
}

Map::~Map() {
  hash_table_.Free();
}

void Map::Reset() {
  integrated_frame_count_ = 0;

  hash_params_.m_rigidTransform.setIdentity();
  hash_params_.m_rigidTransformInverse.setIdentity();
  hash_params_.occupied_block_count = 0;

  hash_table_.updateParams(hash_params_);
  resetCUDA(hash_table_, hash_params_);
}

void Map::AllocBlocks(const SensorData &sensor_data,
                      const SensorParams& sensor_params) {
  resetHashBucketMutexCUDA(hash_table_, hash_params_);
  // TODO(wei): add bit_mask
  allocCUDA(hash_table_, hash_params_,
            sensor_data, sensor_params, NULL);
  // TODO(wei): change it here
}

void Map::GenerateCompressedHashEntries() {
  hash_params_.occupied_block_count = compactifyHashAllInOneCUDA(hash_table_,
                                                                 hash_params_);
  //this version uses atomics over prefix sums, which has a much better performance
  std::cout << "Occupied Blocks: " << hash_params_.occupied_block_count << std::endl;
  hash_table_.updateParams(hash_params_);  //make sure numOccupiedBlocks is updated on the GPU
}

void Map::RecycleInvalidBlocks() {
  bool garbage_collect = true;         /// false
  int garbage_collect_starve = 15;      /// 15
  if (garbage_collect) {

    if (integrated_frame_count_ > 0 && integrated_frame_count_ % garbage_collect_starve == 0) {
      starveVoxelsKernelCUDA(hash_table_, hash_params_);
    }

    garbageCollectIdentifyCUDA(hash_table_, hash_params_);
    resetHashBucketMutexCUDA(hash_table_, hash_params_);  //needed if linked lists are enabled -> for memeory deletion
    garbageCollectFreeCUDA(hash_table_, hash_params_);
  }
}