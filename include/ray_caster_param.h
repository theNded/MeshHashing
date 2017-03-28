//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_PARAM_H
#define MRF_VH_RAY_CASTER_PARAM_H

#include "common.h"

#include <helper_math.h>

struct __ALIGN__(16) RayCastParams {
  float4x4 m_viewMatrix;               /// World to Cam, c_T_w
  float4x4 m_viewMatrixInverse;
  float4x4 m_intrinsics;               /// Intrinsic matrix
  float4x4 m_intrinsicsInverse;

  unsigned int m_width;                /// 640
  unsigned int m_height;               /// 480

	/// Seems deprecated
  unsigned int m_numOccupiedSDFBlocks;
  unsigned int m_maxNumVertices;
  int m_splatMinimum;

  float m_minDepth;
  float m_maxDepth;
  float m_rayIncrement;                /// 0.8f * SDF_Truncation
  float m_thresSampleDist;             /// 50.5f * s_rayIncrement
  float m_thresDist;                   /// 50.0f * s_rayIncrement
  bool  m_useGradients;

  uint dummy0;
};
#endif //MRF_VH_RAY_CASTER_PARAM_H
