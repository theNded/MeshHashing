//
// Created by wei on 17-10-26.
//

#ifndef MESH_HASHING_PRIMAL_DUAL_QUERY_H
#define MESH_HASHING_PRIMAL_DUAL_QUERY_H

#include "core/common.h"
#include "core/block_array.h"
#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "geometry/voxel_query.h"
#include "helper_math.h"

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
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

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
inline float GetDualDivergence(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  Voxel voxel_query;
  bool valid = GetVoxelValue(curr_entry, voxel_pos,
                             blocks, hash_table,
                             geometry_helper, &voxel_query);
  if (! valid) return 0;
  return (voxel_query.p.x + voxel_query.p.y + voxel_query.p.z);
}

__device__
inline float3 GetPrimalGradient(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  const int3 grad_offsets[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

  bool valid = true;
  Voxel voxel_query;
  float primalp[3], primaln[3];
#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    valid = valid && GetPrimalDualValue(curr_entry, voxel_pos + grad_offsets[i],
                                        blocks, hash_table,
                                        geometry_helper, &voxel_query);
    primalp[i] = voxel_query.x;
    valid = valid && GetPrimalDualValue(curr_entry, voxel_pos - grad_offsets[i],
                                        blocks, hash_table,
                                        geometry_helper, &voxel_query);
    primaln[i] = voxel_query.x;
    if (! valid) break;
  }

  float3 primal_gradient = make_float3(primalp[0] - primaln[0],
                                       primalp[1] - primaln[1],
                                       primalp[2] - primaln[2]);
  if (! valid) return make_float3(0.0f);
  return primal_gradient;
}


#endif //MESH_HASHING_PRIMAL_DUAL_QUERY_H
