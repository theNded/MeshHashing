//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_PARAM_H
#define MRF_VH_RAY_CASTER_PARAM_H

#include "common.h"

#include <helper_math.h>

struct __ALIGN__(16) RayCasterParams {
  float4x4 intrinsics;               /// Intrinsic matrix
  float4x4 intrinsics_inverse;

  uint width;                /// 640
  uint height;               /// 480

  float min_raycast_depth;
  float max_raycast_depth;
  float raycast_step;                /// 0.8f * SDF_Truncation

  float sample_sdf_threshold;        /// 50.5f * s_rayIncrement
  float sdf_threshold;               /// 50.0f * s_rayIncrement
  bool  enable_gradients;

  uint dummy0;
};
#endif //MRF_VH_RAY_CASTER_PARAM_H
