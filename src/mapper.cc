//
// Created by wei on 17-3-16.
//

#include "mapper.h"
#include "sensor.h"
#include "sensor_data.h"
#include <unordered_set>
#include <vector>
#include <list>

Mapper::Mapper() {}
Mapper::~Mapper() {}

void Mapper::BindSensorDataToTexture(const SensorData &sensor_data) {
  BindSensorDataToTextureCudaHost(sensor_data);
}

void Mapper::Integrate(Map *map, Sensor* sensor,
                       unsigned int *d_bitMask) {

  //make the rigid transform available on the GPU
  //map->hash_table().updateParams(map->hash_params());
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  map->AllocBlocks(sensor);
  /// DIFFERENT: d_bitMask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  map->GenerateCompressedHashEntries(sensor->c_T_w());
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  IntegrateDepthMap(map, sensor);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  map->RecycleInvalidBlocks();

  map->integrated_frame_count_++;
}

void Mapper::IntegrateDepthMap(Map *map, Sensor *sensor) {
  IntegrateCudaHost(map->hash_table(), map->hash_params(),
                        sensor->getSensorData(), sensor->getSensorParams(),
                        sensor->c_T_w());
}