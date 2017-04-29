//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_HASH_TABLE_MANAGER_H
#define MRF_VH_HASH_TABLE_MANAGER_H

#include "common.h"

#include "geometry_util.h"
#include "params.h"
#include "hash_table_gpu.h"
#include "map.h"
#include "sensor.h"


/// CUDA / C++ shared class
class Mapper {
public:
  Mapper();
  ~Mapper();

  /// Set input (image)
  /// Should bind only once
  void BindSensorDataToTexture(const SensorData& sensor_data);

  void Integrate(Map* map, Sensor *sensor, unsigned int* is_streamed_mask);

private:
  void IntegrateDepthMap(Map* map, Sensor* sensor);
  void AllocBlocks(Map* map, Sensor* sensor);
};

#endif //MRF_VH_HASH_TABLE_MANAGER_H
