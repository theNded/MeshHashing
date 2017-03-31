//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_H
#define MRF_VH_RAY_CASTER_H

#include "common.h"

#include "hash_table.h"
#include "ray_caster_data.h"
#include "ray_caster_param.h"
#include "sensor_data.h"

/// CUDA functions
extern void renderCS(const HashTable& hashData, const RayCastData &rayCastData, const DepthCameraData &cameraData, const RayCastParams &rayCastParams);

extern void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height, const DepthCameraData& depthCameraData);

/// CUDA / C++ shared class
class CUDARayCastSDF {
public:
  CUDARayCastSDF(const RayCastParams& params);
  ~CUDARayCastSDF(void);

  void render(const HashTable& hashData, const HashParams& hashParams, const DepthCameraData& cameraData, const float4x4& lastRigidTransform);

  const RayCastData& getRayCastData(void) {
    return m_data;
  }
  const RayCastParams& getRayCastParams() const {
    return m_params;
  }

  // debugging
  void convertToCameraSpace(const DepthCameraData& cameraData);

private:

  void create(const RayCastParams& params);
  void destroy(void);

  RayCastParams m_params;
  /// float *d_depth in CUDA
  RayCastData m_data;

};

#endif //MRF_VH_RAY_CASTER_H
