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
    float &sdf,
    uchar3 &color
#ifdef STATS
,
Stat  &stats,
#endif
) {
  const float offset = geometry_helper.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(geometry_helper.WorldToVoxelf(pos));

  float w;
  Voxel v;

  sdf = 0.0f;
  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;

#ifdef STATS
  stats.Clear();
#endif

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    float3 mask = make_float3((i&4)>0, (i&2)>0, (i&1)>0);
    // 0 --> 1 - r, 1 --> r
    float3 r = (make_float3(1.0f) - mask) * (make_float3(1.0) - ratio)
             + (mask) * ratio;
    v = GetVoxel(pos_corner + mask * offset, blocks, hash_table, geometry_helper);
    if (v.weight < EPSILON) return false;
    v_color = make_float3(v.color.x, v.color.y, v.color.z);
    w = r.x * r.y * r.z;
    sdf += w * v.sdf;
    colorf += w * v_color;
    // TODO: Interpolation of stats
  }

  color = make_uchar3(colorf.x, colorf.y, colorf.z);
  return true;
}

__device__
inline float3 GetSpatialGradient(
    const float3 &pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  const float voxelSize = geometry_helper.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  /// negative
  float distn00;
  uchar3 colorn00;
  GetSpatialValue(pos - make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         blocks,
                         hash_table,
                         geometry_helper,
                         distn00, colorn00);
  float dist0n0;
  uchar3 color0n0;
  GetSpatialValue(pos - make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         blocks,
                         hash_table,
                         geometry_helper,
                         dist0n0, color0n0);

  float dist00n;
  uchar3 color00n;
  GetSpatialValue(pos - make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         blocks,
                         hash_table,
                         geometry_helper,
                         dist00n, color00n);

  /// positive
  float distp00;
  uchar3 colorp00;
  GetSpatialValue(pos + make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         blocks,
                         hash_table,
                         geometry_helper,
                         distp00, colorp00);
  float dist0p0;
  uchar3 color0p0;
  GetSpatialValue(pos + make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         blocks,
                         hash_table,
                         geometry_helper,
                         dist0p0, color0p0);

  float dist00p;
  uchar3 color00p;
  GetSpatialValue(pos + make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         blocks,
                         hash_table,
                         geometry_helper,
                         dist00p, color00p);

  float3 grad = make_float3((distp00 - distn00) / offset.x,
                            (dist0p0 - dist0n0) / offset.y,
                            (dist00p - dist00n) / offset.z);

  float l = length(grad);
  if (l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return grad / l;
}

#endif