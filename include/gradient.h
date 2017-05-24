#include <matrix.h>
#include <geometry_util.h>

#include "hash_table.h"
#include "block.h"

__device__
inline float frac(float val) {
  return (val - floorf(val));
}
__device__
inline float3 frac(const float3& val)  {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

// TODO(wei): refine it
__device__
inline Voxel GetVoxel(const HashTableGPU& hash_table,
                       VoxelBlock *blocks, float3 world_pos) {
  HashEntry hash_entry = hash_table.GetEntry(WorldToBlock(world_pos));
  Voxel v;
  if (hash_entry.ptr == FREE_ENTRY) {
    v.Clear();
  } else {
    int3 voxel_pos = WorldToVoxeli(world_pos);
    int i = VoxelPosToIdx(voxel_pos);
    v = blocks[hash_entry.ptr].voxels[i];
  }
  return v;
}

__device__
inline bool TrilinearInterpolation(const HashTableGPU& hash_table,
                            VoxelBlock *blocks,
                            const float3& pos,
                            float& sdf, uchar3& color) {
  const float offset = kSDFParams.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(WorldToVoxelf(pos));

  float w;
  Voxel v;

  sdf = 0.0f;
  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;

  /// 000
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(0.0f, 0.0f, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*(1.0f-ratio.y)*(1.0f-ratio.z);
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 001
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(0.0f, 0.0f, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*(1.0f-ratio.y)*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 010
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(0.0f, offset, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*ratio.y *(1.0f-ratio.z);
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 011
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(0.0f, offset, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = (1.0f-ratio.x)*ratio.y*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  /// 100
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(offset, 0.0f, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*(1.0f-ratio.y)*(1.0f-ratio.z);
  sdf    +=	w * v.sdf;
  colorf += w * v_color;

  /// 101
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(offset, 0.0f, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*(1.0f-ratio.y)*ratio.z;
  sdf    +=	w * v.sdf;
  colorf += w	* v_color;

  /// 110
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(offset, offset, 0.0f));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*ratio.y*(1.0f-ratio.z);
  sdf    +=	w * v.sdf;
  colorf += w * v_color;

  /// 111
  v = GetVoxel(hash_table, blocks, pos_corner+make_float3(offset, offset, offset));
  if(v.weight == 0) return false;
  v_color = make_float3(v.color.x, v.color.y, v.color.z);
  w = ratio.x*ratio.y*ratio.z;
  sdf    += w * v.sdf;
  colorf += w * v_color;

  color = make_uchar3(colorf.x, colorf.y, colorf.z);
  return true;
}

__device__
/// sdf_near: -, sdf_far: +
inline float LinearIntersection(float t_near, float t_far, float sdf_near, float sdf_far) {
  return t_near + (sdf_near / (sdf_near - sdf_far)) * (t_far - t_near);
}

// d0 near, d1 far
__device__
/// Iteratively
inline bool BisectionIntersection(const HashTableGPU& hash_table,
                           VoxelBlock* blocks,
                           const float3& world_pos_camera_origin,
                           const float3& world_dir,
                           float sdf_near, float t_near,
                           float sdf_far,  float t_far,
                           float& t, uchar3& color) {
  float l = t_near, r = t_far, m = (l + r) * 0.5f;
  float l_sdf = sdf_near, r_sdf = sdf_far, m_sdf;

  const uint kIterations = 3;
#pragma unroll 1
  for(uint i = 0; i < kIterations; i++) {
    m = LinearIntersection(l, r, l_sdf, r_sdf);
    if(!TrilinearInterpolation(hash_table, blocks,
                               world_pos_camera_origin + m * world_dir,
                               m_sdf, color))
      return false;

    if (l_sdf * m_sdf > 0.0) {
      l = m; l_sdf = m_sdf;
    } else {
      r = m; r_sdf = m_sdf;
    }
  }
  t = m;
  return true;
}

__device__
inline float3 GradientAtPoint(const HashTableGPU& hash_table,
                               VoxelBlocksGPU blocks,
                               const float3& pos) {
  const float voxelSize = kSDFParams.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  /// negative
  float distn00; uchar3 colorn00;
  TrilinearInterpolation(hash_table, blocks, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f),
                         distn00, colorn00);
  float dist0n0; uchar3 color0n0;
  TrilinearInterpolation(hash_table, blocks, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f),
                         dist0n0, color0n0);
  float dist00n; uchar3 color00n;
  TrilinearInterpolation(hash_table, blocks, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z),
                         dist00n, color00n);

  /// positive
  float distp00; uchar3 colorp00;
  TrilinearInterpolation(hash_table, blocks, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f),
                         distp00, colorp00);
  float dist0p0; uchar3 color0p0;
  TrilinearInterpolation(hash_table, blocks, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f),
                         dist0p0, color0p0);
  float dist00p; uchar3 color00p;
  TrilinearInterpolation(hash_table, blocks, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z),
                         dist00p, color00p);

  float3 grad = make_float3((distp00-distn00)/offset.x,
                            (dist0p0-dist0n0)/offset.y,
                            (dist00p-dist00n)/offset.z);

  float l = length(grad);
  if(l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return grad/l;
}