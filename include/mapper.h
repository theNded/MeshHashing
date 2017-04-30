//
// Created by wei on 17-3-16.
//

#ifndef VH_MAPPER_H
#define VH_MAPPER_H

#include "common.h"

#include "geometry_util.h"
#include "params.h"
#include "hash_table_gpu.h"
#include "map.h"
#include "sensor.h"

class Mapper {
private:
  void IntegrateDepthMap(Map* map, Sensor* sensor);
  void AllocBlocks(Map* map, Sensor* sensor);

public:
  Mapper();
  ~Mapper();

  void Integrate(Map* map, Sensor *sensor, unsigned int* is_streamed_mask);
};

#endif //VH_MAPPER_H
