//
// Created by wei on 17-3-17.
//

#include "ray_caster.h"
#include "ray_caster_data.h"
#include "hash_table_gpu.h"

RayCaster::RayCaster(const RayCasterParams& params) {
  ray_caster_params_ = params;
  ray_caster_data_.Alloc(ray_caster_params_);
}

RayCaster::~RayCaster(void) {
  ray_caster_data_.Free();
}

/// Major function, extract surface and normal from the volumes
void RayCaster::Cast(Map* map, const float4x4& c_T_w) {
  const float4x4 w_T_c = c_T_w.getInverse();

  CastCudaHost(map->hash_table(), ray_caster_data_, ray_caster_params_, c_T_w, w_T_c);
}