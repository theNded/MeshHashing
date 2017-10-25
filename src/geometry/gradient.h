#include <matrix.h>
#include "geometry_helper.h"

#include "core/hash_table.h"
#include "core/block_array.h"
#include "geometry/spatial_query.h"

__device__
inline float frac(float val) {
  return (val - floorf(val));
}

__device__
inline float3 frac(const float3 &val) {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

// TODO: simplify this code
/// Interpolation of statistics involved
__device__
inline bool TrilinearInterpolation(const HashTable &hash_table,
                                   const BlockArray &blocks,
                                   const float3 &pos,
                                   float &sdf,
#ifdef STATS
                                   Stat  &stats,
#endif
                                   uchar3 &color,
                                   GeometryHelper& geoemtry_helper) {
  const float offset = geoemtry_helper.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(geoemtry_helper.WorldToVoxelf(pos));

  float w;
  Voxel v;

  sdf = 0.0f;
#ifdef STATS
  stats.Clear();
#endif

  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    float3 mask = make_float3((i&4)>0, (i&2)>0, (i&1)>0);
    // 0 --> 1 - r, 1 --> r
    float3 r = (make_float3(1.0f) - mask) * (make_float3(1.0) - ratio)
             + (mask) * ratio;
    v = GetVoxel(hash_table, blocks, pos_corner + mask * offset, geoemtry_helper);
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
/// sdf_near: -, sdf_far: +
inline float LinearIntersection(float t_near, float t_far,
                                float sdf_near, float sdf_far) {
  return t_near + (sdf_near / (sdf_near - sdf_far)) * (t_far - t_near);
}

// d0 near, d1 far
__device__
/// Iteratively
inline bool BisectionIntersection(const HashTable &hash_table,
                                  const BlockArray &blocks,
                                  const float3 &world_cam_pos,
                                  const float3 &world_cam_dir,
                                  float sdf_near, float t_near,
                                  float sdf_far, float t_far,
                                  float &t, uchar3 &color,
                                  GeometryHelper& geoemtry_helper) {
  float l = t_near, r = t_far, m = (l + r) * 0.5f;
  float l_sdf = sdf_near, r_sdf = sdf_far, m_sdf;

  const uint kIterations = 3;
#pragma unroll 1
  for (uint i = 0; i < kIterations; i++) {
    m = LinearIntersection(l, r, l_sdf, r_sdf);
    if (!TrilinearInterpolation(hash_table, blocks,
                                world_cam_pos + m * world_cam_dir,
                                m_sdf, color, geoemtry_helper))
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

__device__
inline float3 GradientAtPoint(const HashTable &hash_table,
                              const BlockArray &blocks,
                              const float3 &pos,
                              GeometryHelper& geoemtry_helper) {
  const float voxelSize = geoemtry_helper.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  /// negative
  float distn00;
  uchar3 colorn00;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         distn00, colorn00,
                         geoemtry_helper);
  float dist0n0;
  uchar3 color0n0;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         dist0n0, color0n0,
                         geoemtry_helper);
  float dist00n;
  uchar3 color00n;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         dist00n, color00n,
                         geoemtry_helper);

  /// positive
  float distp00;
  uchar3 colorp00;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         distp00, colorp00,
                         geoemtry_helper);
  float dist0p0;
  uchar3 color0p0;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         dist0p0, color0p0,
                         geoemtry_helper);
  float dist00p;
  uchar3 color00p;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         dist00p, color00p,
                         geoemtry_helper);

  float3 grad = make_float3((distp00 - distn00) / offset.x,
                            (dist0p0 - dist0n0) / offset.y,
                            (dist00p - dist00n) / offset.z);

  float l = length(grad);
  if (l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return grad / l;
}