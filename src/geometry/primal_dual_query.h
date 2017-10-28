//
// Created by wei on 17-10-26.
//

#ifndef MESH_HASHING_PRIMAL_DUAL_QUERY_H
#define MESH_HASHING_PRIMAL_DUAL_QUERY_H

#include "core/common.h"
#include "core/block_array.h"
#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "helper_math.h"

// TODO: fix these awful conversions and coordinates
__device__
inline bool GetPrimalDualValue(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    Voxel* voxel
) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  int3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    const Voxel &v = blocks[curr_entry.ptr].voxels[i];
    voxel->x = v.x;
    voxel->p = v.p;
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY) return false;
    uint i = geometry_helper.VectorizeOffset(offset);
    const Voxel &v = blocks[entry.ptr].voxels[i];
    voxel->x = v.x;
    voxel->p = v.p;
  }
  return true;
}

__device__
inline bool GetPrimalGradientDualDivergence(
    const HashEntry &curr_entry,
    const int3 voxel_local_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float3& primal_gradient,
    float& dual_divergence
) {
  bool valid = false;

  float primal000;
  float3 dual000;
  valid = GetPrimalDualValue(curr_entry,
                             voxel_local_pos + make_int3(0, 0, 0),
                             blocks,
                             hash_table,
                             geometry_helper,
                             primal000, dual000);
  if (! valid) return false;
  dual_divergence = dual000.x + dual000.y + dual000.z;

  /// negative
  float primaln00;
  float3 dualn00;
  valid = GetPrimalDualValue(curr_entry,
                                  voxel_local_pos + make_int3(-1, 0, 0),
                                  blocks,
                                  hash_table,
                                  geometry_helper,
                                  primaln00, dualn00);
  if (! valid) return false;

  float primal0n0;
  float3 dual0n0;
  valid = GetPrimalDualValue(curr_entry,
                             voxel_local_pos + make_int3(0, -1, 0),
                             blocks,
                             hash_table,
                             geometry_helper,
                             primal0n0, dual0n0);
  if (! valid) return false;

  float primal00n;
  float3 dual00n;
  valid = GetPrimalDualValue(curr_entry,
                                  voxel_local_pos + make_int3(0, 0, -1),
                                  blocks,
                                  hash_table,
                                  geometry_helper,
                                  primal00n, dual00n);
  if (! valid) return false;

  /// Positive
  float primalp00;
  float3 dualp00;
  valid = GetPrimalDualValue(curr_entry,
                             voxel_local_pos + make_int3(-1, 0, 0),
                             blocks,
                             hash_table,
                             geometry_helper,
                             primalp00, dualp00);
  if (! valid) return false;

  float primal0p0;
  float3 dual0p0;
  valid = GetPrimalDualValue(curr_entry,
                             voxel_local_pos + make_int3(0, -1, 0),
                             blocks,
                             hash_table,
                             geometry_helper,
                             primal0p0, dual0p0);
  if (! valid) return false;

  float primal00p;
  float3 dual00p;
  valid = GetPrimalDualValue(curr_entry,
                             voxel_local_pos + make_int3(0, 0, -1),
                             blocks,
                             hash_table,
                             geometry_helper,
                             primal00p, dual00p);
  if (! valid) return false;

  primal_gradient = make_float3(primalp00 - primaln00,
                                primal0p0 - primal0n0,
                                primal00p - primal00n);

  return false;
}


#endif //MESH_HASHING_PRIMAL_DUAL_QUERY_H
