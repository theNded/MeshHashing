//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_CAMERA_PARAM_H
#define MRF_VH_CAMERA_PARAM_H

#include "common.h"

/// We may generate a virtual camera for LiDAR
/// where many points are null
struct __ALIGN__(16) SensorParams {
  float fx;              /// Set manually
  float fy;
  float cx;
  float cy;

  uint width;            /// 640
  uint height;           /// 480

  float min_depth_range; /// 0.5f
  float max_depth_range; /// 5.0f, might need modify for LiDAR
};


#endif //MRF_VH_CAMERA_PARAM_H
