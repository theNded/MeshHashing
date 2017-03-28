//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_CAMERA_PARAM_H
#define MRF_VH_CAMERA_PARAM_H

#include "common.h"

/// We may generate a virtual camera for LiDAR
/// where many points are null
struct __ALIGN__(16) DepthCameraParams {
  float fx;                    /// Set manually
  float fy;
  float mx;
  float my;

  unsigned int m_imageWidth;   /// 640
  unsigned int m_imageHeight;  /// 480

  float m_sensorDepthWorldMin; /// 5.0f
  float m_sensorDepthWorldMax; /// 0.5f, might need modify for LiDAR
};


#endif //MRF_VH_CAMERA_PARAM_H
