#include <matrix.h>
#include "coordinate_utils.h"

#include "../core/hash_table.h"
#include "../core/block.h"

__device__
inline float frac(float val) {
  return (val - floorf(val));
}

__device__
inline float3 frac(const float3 &val) {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

// TODO(wei): refine it
__device__
inline Voxel GetVoxel(const HashTableGPU &hash_table,
                      const BlockGPUMemory &blocks,
                      const float3 world_pos,
                      CoordinateConverter& converter) {
  HashEntry hash_entry = hash_table.GetEntry(converter.WorldToBlock(world_pos));
  Voxel v;
  if (hash_entry.ptr == FREE_ENTRY) {
    v.ClearSDF();
  } else {
    int3 voxel_pos = converter.WorldToVoxeli(world_pos);
    int i = converter.VoxelPosToIdx(voxel_pos);
    v = blocks[hash_entry.ptr].voxels[i];
  }
  return v;
}

// TODO: put a dummy here
__device__
inline float GetSDF(const HashTableGPU& hash_table,
                    BlockGPUMemory&          blocks,
                    const HashEntry&    curr_entry,
                    const uint3         voxel_local_pos,
                    float &weight,
                    CoordinateConverter& converter) {
  float sdf = 0.0; weight = 0;
  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = converter.VoxelLocalPosToIdx(voxel_local_pos);
    Voxel& v = blocks[curr_entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return 0;
    uint i = converter.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);

    Voxel &v = blocks[entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  }

  return sdf;
}

__device__
inline Voxel& GetVoxelRef(const HashTableGPU& hash_table,
                     BlockGPUMemory&          blocks,
                     const HashEntry&    curr_entry,
                     const uint3         voxel_local_pos,
                          CoordinateConverter& converter) {

  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = converter.VoxelLocalPosToIdx(voxel_local_pos);
    return blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef: should never reach here!\n");
    }
    uint i = converter.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);
    return blocks[entry.ptr].voxels[i];
  }
}

// TODO: simplify this code
/// Interpolation of statistics involved
__device__
inline bool TrilinearInterpolation(const HashTableGPU &hash_table,
                                   const BlockGPUMemory &blocks,
                                   const float3 &pos,
                                   float &sdf,
                                   Stat  &stats,
                                   uchar3 &color,
                                   CoordinateConverter& converter) {
  const float offset = converter.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(converter.WorldToVoxelf(pos));

  float w;
  Voxel v;

  sdf = 0.0f;
  stats.Clear();

  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    float3 mask = make_float3((i&4)>0, (i&2)>0, (i&1)>0);
    // 0 --> 1 - r, 1 --> r
    float3 r = (make_float3(1.0f) - mask) * (make_float3(1.0) - ratio)
             + (mask) * ratio;
    v = GetVoxel(hash_table, blocks, pos_corner + mask * offset, converter);
    if (v.weight < EPSILON) return false;
    v_color = make_float3(v.color.x, v.color.y, v.color.z);
    w = r.x * r.y * r.z;
    sdf += w * v.sdf;
    colorf += w * v_color;
    // Interpolation of stats
  }

  color = make_uchar3(colorf.x, colorf.y, colorf.z);
  return true;
}

__device__
inline bool TrilinearInterpolation(const HashTableGPU &hash_table,
                                   const BlockGPUMemory &blocks,
                                   const float3 &pos,
                                   float &sdf,
                                   uchar3 &color,
                                   CoordinateConverter& converter) {
  const float offset = converter.voxel_size;
  const float3 pos_corner = pos - 0.5f * offset;
  float3 ratio = frac(converter.WorldToVoxelf(pos));

  float w;
  Voxel v;

  sdf = 0.0f;
  float3 colorf = make_float3(0.0f, 0.0f, 0.0f);
  float3 v_color;

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    float3 mask = make_float3((i&4)>0, (i&2)>0, (i&1)>0);
    // 0 --> 1 - r, 1 --> r
    float3 r = (make_float3(1.0f) - mask) * (make_float3(1.0) - ratio)
               + (mask) * ratio;
    v = GetVoxel(hash_table, blocks, pos_corner + mask * offset, converter);
    if (v.weight < EPSILON) return false;
    v_color = make_float3(v.color.x, v.color.y, v.color.z);
    w = r.x * r.y * r.z;
    sdf += w * v.sdf;
    colorf += w * v_color;
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
inline bool BisectionIntersection(const HashTableGPU &hash_table,
                                  const BlockGPUMemory &blocks,
                                  const float3 &world_cam_pos,
                                  const float3 &world_cam_dir,
                                  float sdf_near, float t_near,
                                  float sdf_far, float t_far,
                                  float &t, uchar3 &color,
                                  CoordinateConverter& converter) {
  float l = t_near, r = t_far, m = (l + r) * 0.5f;
  float l_sdf = sdf_near, r_sdf = sdf_far, m_sdf;

  const uint kIterations = 3;
#pragma unroll 1
  for (uint i = 0; i < kIterations; i++) {
    m = LinearIntersection(l, r, l_sdf, r_sdf);
    if (!TrilinearInterpolation(hash_table, blocks,
                                world_cam_pos + m * world_cam_dir,
                                m_sdf, color, converter))
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
inline float3 GradientAtPoint(const HashTableGPU &hash_table,
                              const BlockGPUMemory &blocks,
                              const float3 &pos,
                              CoordinateConverter& converter) {
  const float voxelSize = converter.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  /// negative
  float distn00;
  uchar3 colorn00;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         distn00, colorn00,
                         converter);
  float dist0n0;
  uchar3 color0n0;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         dist0n0, color0n0,
                         converter);
  float dist00n;
  uchar3 color00n;
  TrilinearInterpolation(hash_table, blocks,
                         pos - make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         dist00n, color00n,
                         converter);

  /// positive
  float distp00;
  uchar3 colorp00;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.5f * offset.x, 0.0f, 0.0f),
                         distp00, colorp00,
                         converter);
  float dist0p0;
  uchar3 color0p0;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.0f, 0.5f * offset.y, 0.0f),
                         dist0p0, color0p0,
                         converter);
  float dist00p;
  uchar3 color00p;
  TrilinearInterpolation(hash_table, blocks,
                         pos + make_float3(0.0f, 0.0f, 0.5f * offset.z),
                         dist00p, color00p,
                         converter);

  float3 grad = make_float3((distp00 - distn00) / offset.x,
                            (dist0p0 - dist0n0) / offset.y,
                            (dist00p - dist00n) / offset.z);

  float l = length(grad);
  if (l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return grad / l;
}