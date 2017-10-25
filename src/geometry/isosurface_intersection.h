//
// Created by wei on 17-10-25.
//

#ifndef GEOMETRY_ISOSURFACE_INTERSECTION_H
#define GEOMETRY_ISOSURFACE_INTERSECTION_H

#include "core/common.h"
#include "geometry/spatial_query.h"

__device__
/// sdf_near: -, sdf_far: +
inline float LinearIntersection(float t_near, float t_far,
                                float sdf_near, float sdf_far) {
  return t_near + (sdf_near / (sdf_near - sdf_far)) * (t_far - t_near);
}

// d0 near, d1 far
__device__
/// Iteratively
inline bool BisectionIntersection(
    const float3 &world_cam_pos,
    const float3 &world_cam_dir,
    float sdf_near, float t_near,
    float sdf_far, float t_far,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper& geometry_helper,
    float &t, uchar3 &color) {
  float l = t_near, r = t_far, m = (l + r) * 0.5f;
  float l_sdf = sdf_near, r_sdf = sdf_far, m_sdf;

  const uint kIterations = 3;
#pragma unroll 1
  for (uint i = 0; i < kIterations; i++) {
    m = LinearIntersection(l, r, l_sdf, r_sdf);
    if (!GetSpatialValue(world_cam_pos + m * world_cam_dir,
                         blocks,
                         hash_table,
                         geometry_helper,
                         m_sdf, color))
      return false;

    if (l_sdf * m_sdf > 0.0) {
      l = m;
      l_sdf = m_sdf;
    } else {
      r = m;
      r_sdf = m_sdf;
    }
  }
  t = m;
  return true;
}

#endif //MESH_HASHING_ISOSURFACE_INTERSECTION_H
