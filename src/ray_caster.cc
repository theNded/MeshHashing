//
// Created by wei on 17-3-17.
//

#include "ray_caster.h"
#include "ray_caster_data.h"
#include "hash_table.h"

CUDARayCastSDF::CUDARayCastSDF(const RayCastParams& params) {
  create(params);
}

CUDARayCastSDF::~CUDARayCastSDF(void) {
  destroy();
}

void CUDARayCastSDF::create(const RayCastParams& params) {
  m_params = params;
  m_data.allocate(m_params);
}

void CUDARayCastSDF::destroy(void) {
  m_data.free();
}

/// Major function, extract surface and normal from the volumes
void CUDARayCastSDF::render(const HashTable& HashTable, const HashParams& hashParams,
                            const DepthCameraData& cameraData, const float4x4& lastRigidTransform) {

  m_params.m_viewMatrix = lastRigidTransform;
  m_params.m_viewMatrixInverse = m_params.m_viewMatrix.getInverse();
  m_data.updateParams(m_params);

  renderCS(HashTable, m_data, cameraData, m_params);

  //convertToCameraSpace(cameraData);
  //if (!m_params.m_useGradients) {
//    computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
//  }
}

//
//void CUDARayCastSDF::convertToCameraSpace(const DepthCameraData& cameraData) {
//  convertDepthFloatToCameraSpaceFloat4(m_data.d_depth4, m_data.d_depth, m_params.m_intrinsicsInverse, m_params.m_width, m_params.m_height, cameraData);
//
//  if(!m_params.m_useGradients) {
//    computeNormals(m_data.d_normals, m_data.d_depth4, m_params.m_width, m_params.m_height);
//  }
//}
