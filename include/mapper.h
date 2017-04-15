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
#include "sensor.h"

/// ! FUSION PART !
extern void integrateDepthMapCUDA(HashTable& hash_table, const HashParams& hash_params,
                                  const SensorData& sensor_data, const SensorParams& sensor_params,
                                  float4x4 c_T_w);
extern void bindInputDepthColorTextures(const SensorData& sensor_data);

/// CUDA / C++ shared class
class Mapper {
public:
  Mapper();
  ~Mapper();

  /// Set input (image)
  void bindDepthCameraTextures(const SensorData& sensor_data);

  /// SDF fusion
  void integrate(Map* map, Sensor *sensor, unsigned int* d_bitMask);

private:
  void integrateDepthMap(Map* map, Sensor* sensor);
};

#endif //MRF_VH_HASH_TABLE_MANAGER_H
