//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_H
#define MRF_VH_RAY_CASTER_H

#include "common.h"

#include "map.h"
#include "params.h"
#include "sensor.h"


struct RayCasterSample {
  float sdf;
  float t;
  uint weight;
};

struct RayCasterData {
  float  *depth_image_;
  float4 *vertex_image_;
  float4 *normal_image_;
  float4 *color_image_;
};


class RayCaster {
public:
  RayCaster(const RayCasterParams& params);
  ~RayCaster(void);

  void Cast(Map* map, const float4x4& c_T_w);


  const RayCasterData& ray_caster_data() {
    return ray_caster_data_;
  }
  const RayCasterParams& ray_caster_params() const {
    return ray_caster_params_;
  }

private:
  RayCasterParams ray_caster_params_;
  RayCasterData ray_caster_data_;

};

#endif //MRF_VH_RAY_CASTER_H
