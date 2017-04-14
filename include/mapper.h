//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_HASH_TABLE_MANAGER_H
#define MRF_VH_HASH_TABLE_MANAGER_H

#include "common.h"

#include "geometry_util.h"
#include "hash_param.h"
#include "hash_table.h"
#include "map.h"
#include "sensor_data.h"
#include "sensor_param.h"

/// ! FUSION PART !
extern void integrateDepthMapCUDA(HashTable& hash_table, const HashParams& hash_params, const SensorData& sensor_data, const SensorParams& depthCameraParams);
extern void bindInputDepthColorTextures(const SensorData& sensor_data);

/// CUDA / C++ shared class
class Mapper {
public:
  Mapper(Map *voxel_map);
  ~Mapper();


  /// Set input (image)
  void bindDepthCameraTextures(const SensorData& sensor_data);

  /// SDF fusion
  void integrate(const float4x4& lastRigidTransform, const SensorData& sensor_data,
                 const SensorParams& depthCameraParams, unsigned int* d_bitMask);

  /// Set pose
  void setLastRigidTransform(const float4x4& lastRigidTransform);

  //! debug only!
  unsigned int getHeapFreeCount();
  void debugHash();

private:
  void integrateDepthMap(const SensorData& sensor_data, const SensorParams& depthCameraParams);

  Map *map_;
};

#endif //MRF_VH_HASH_TABLE_MANAGER_H
