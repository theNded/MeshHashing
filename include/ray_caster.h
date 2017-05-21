//
// Created by wei on 17-3-17.
//

#ifndef VH_RAY_CASTER_H
#define VH_RAY_CASTER_H

#include "common.h"

#include "map.h"
#include "params.h"
#include "sensor.h"

struct RayCasterSample {
  float  sdf;
  float  t;
  uint   weight;
};

struct RayCasterDataGPU {
  float  *depth_image;
  float4 *vertex_image;
  float4 *normal_image;
  float4 *color_image;
};

class RayCaster {
private:
  RayCasterDataGPU gpu_data_;
  RayCasterParams  ray_caster_params_;

public:
  RayCaster(const RayCasterParams& params);
  ~RayCaster(void);

  void Cast(Map& map, const float4x4& c_T_w);

  const RayCasterDataGPU& gpu_data() {
    return gpu_data_;
  }
  const RayCasterParams& ray_caster_params() const {
    return ray_caster_params_;
  }
};

#endif //VH_RAY_CASTER_H
