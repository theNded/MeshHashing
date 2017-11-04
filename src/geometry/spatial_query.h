#ifndef GEOMETRY_SPATIAL_QUERY_H
#define GEOMETRY_SPATIAL_QUERY_H

#include <matrix.h>
#include "geometry_helper.h"

#include "core/hash_table.h"
#include "core/block_array.h"
#include "geometry/voxel_query.h"

__device__
inline float frac(float val) {
  return (val - floorf(val));
}

__device__
inline float3 frac(const float3 &val) {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

// TODO: simplify this code
// @function with tri-linear interpolation
__device__
inline bool GetSpatialValue(
    const float3 &pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    Voxel* voxel
) {
  const float offset = geometry_helper.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(geometry_helper.WorldToVoxelf(pos));

  Voxel voxel_query;
  float sdf = 0.0f;
  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float a = 0.0f;
  float b = 0.0f;
  float radius = 0.0f;

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    float3 mask = make_float3((i & 4) > 0, (i & 2) > 0, (i & 1) > 0);
    // 0 --> 1 - r, 1 --> r
    float3 r = (make_float3(1.0f) - mask) * (make_float3(1.0) - ratio)
               + (mask) * ratio;
    bool valid = GetVoxelValue(pos_corner + mask * offset, blocks, hash_table,
                               geometry_helper, &voxel_query);
    if (! valid) return false;
    float w = r.x * r.y * r.z;
    sdf += w * voxel_query.sdf;
    colorf += w * make_float3(voxel_query.color);
    a += w * voxel_query.a;
    b += w * voxel_query.b;
    radius += w * sqrtf(1.0f / voxel_query.weight);
    // TODO: Interpolation of stats
  }

  voxel->sdf = sdf;
  voxel->color = make_uchar3(colorf.x, colorf.y, colorf.z);
  voxel->a = a;
  voxel->b = b;
  voxel->weight = 1.0f / squaref(radius);
  return true;
}

__device__
inline float3 GetSpatialSDFGradient(
    const float3 &pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  const float3 grad_masks[3] = {{0.5, 0, 0}, {0, 0.5, 0}, {0, 0, 0.5}};
  const float3 offset = make_float3(geometry_helper.voxel_size);

  bool valid = true;
  float sdfp[3], sdfn[3];
  Voxel voxel_query;
#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    float3 dpos = grad_masks[i] * offset;
    valid = valid && GetSpatialValue(pos - dpos, blocks, hash_table,
                                     geometry_helper, &voxel_query);
    sdfn[i] = voxel_query.sdf;
    valid = valid && GetSpatialValue(pos + dpos, blocks, hash_table,
                                     geometry_helper, &voxel_query);
    sdfp[i] = voxel_query.sdf;
  }

  float3 grad = make_float3((sdfp[0] - sdfn[0]) / offset.x,
                            (sdfp[1] - sdfn[1]) / offset.y,
                            (sdfp[2] - sdfn[2]) / offset.z);
  float l = length(grad);
  if (l == 0.0f || ! valid) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  return grad / l;
}

#endif