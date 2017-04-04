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
extern void renderCS(const HashTable& hash_table, const RayCasterData &rayCastData, const SensorData &cameraData, const RayCastParams &rayCastParams);

extern void computeNormals(float4* d_output, float4* d_input, unsigned int width, unsigned int height);
extern void convertDepthFloatToCameraSpaceFloat4(float4* d_output, float* d_input, float4x4 intrinsicsInv, unsigned int width, unsigned int height, const SensorData& sensor_data);

/// CUDA / C++ shared class
class RayCaster {
public:
  RayCaster(const RayCastParams& params);
  ~RayCaster(void);

  void render(const HashTable& hash_table, const HashParams& hash_params, const SensorData& cameraData, const float4x4& lastRigidTransform);

  const RayCasterData& getRayCasterData(void) {
    return m_data;
  }
  const RayCastParams& getRayCastParams() const {
    return m_params;
  }

  // debugging
  void convertToCameraSpace(const SensorData& cameraData);

private:

  void create(const RayCastParams& params);
  void destroy(void);

  RayCastParams m_params;
  /// float *d_depth in CUDA
  RayCasterData m_data;

};

#endif //MRF_VH_RAY_CASTER_H
