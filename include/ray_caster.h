//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_H
#define MRF_VH_RAY_CASTER_H

#include "common.h"

#include "map.h"
#include "ray_caster_data.h"
#include "ray_caster_param.h"
#include "sensor_data.h"

/// CUDA functions
extern void CastCudaHost(const HashTable& hash_table, const RayCasterData &rayCastData, const RayCasterParams &rayCastParams);

class RayCaster {
public:
  RayCaster(const RayCasterParams& params);
  ~RayCaster(void);

  void Cast(Map* map, const float4x4& c_T_w);

  const RayCasterData& ray_caster_data(void) {
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
