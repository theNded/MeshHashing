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
    Voxel* voxel, // primal
    PrimalDualVariables* primal_dual_variables // dual
) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    const Block& block = blocks[curr_entry.ptr];
    *voxel = block.voxels[i];
    *primal_dual_variables = block.primal_dual_variables[i];
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY)
      return false;
    uint i = geometry_helper.VectorizeOffset(offset);
    const Block& block = blocks[entry.ptr];
    *voxel = block.voxels[i];
    *primal_dual_variables = block.primal_dual_variables[i];
  }
  return true;
}

__device__
inline bool GetSDFGradient(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float3* primal_gradient
) {
  const int3 grad_offsets[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

  Voxel voxel_query;
  PrimalDualVariables primal_dual_variable_query;
  bool valid = GetPrimalDualValue(curr_entry, voxel_pos,
                                  blocks, hash_table,
                                  geometry_helper,
                                  &voxel_query,
                                  &primal_dual_variable_query);
  if (! valid
      || voxel_query.weight < EPSILON
      || !primal_dual_variable_query.mask) {
    printf("GetSDFGradinet: Invalid Center\n");
  }

  float primal = voxel_query.sdf;
  float primalp[3];
#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    valid = GetPrimalDualValue(curr_entry, voxel_pos + grad_offsets[i],
                               blocks, hash_table,
                               geometry_helper,
                               &voxel_query,
                               &primal_dual_variable_query);
    if (! valid
        || voxel_query.weight < EPSILON
        || !primal_dual_variable_query.mask) {
      *primal_gradient = make_float3(0);
      return false;
    }
    primalp[i] = voxel_query.sdf;
  }

  *primal_gradient = make_float3(primalp[0] - primal,
                                 primalp[1] - primal,
                                 primalp[2] - primal);
  return true;
}

__device__
inline bool GetDualDivergence(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float *dual_divergence
) {
  const int3 grad_offsets[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

  Voxel voxel_query;
  PrimalDualVariables primal_dual_variable_query;
  bool valid = GetPrimalDualValue(curr_entry, voxel_pos,
                                  blocks, hash_table,
                                  geometry_helper,
                                  &voxel_query,
                                  &primal_dual_variable_query);
  if (! valid
      || voxel_query.weight < EPSILON
      || !primal_dual_variable_query.mask) {
    printf("GetDualDivergence: Invalid Center\n");
    return false;
  }

  float3 dual = primal_dual_variable_query.p;
  float3 dualn[3];
#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    valid = valid && GetPrimalDualValue(curr_entry, voxel_pos - grad_offsets[i],
                                        blocks, hash_table,
                                        geometry_helper,
                                        &voxel_query,
                                        &primal_dual_variable_query);
    dualn[i] = primal_dual_variable_query.p;
    if (! valid
        || voxel_query.weight < EPSILON
        || !primal_dual_variable_query.mask) {
      *dual_divergence = 0;
      return false;
    }
  }

  *dual_divergence = (dualn[0] - dual).x
                     + (dualn[1] - dual).y
                     + (dualn[2] - dual).z;
  return true;
}

__device__
inline bool GetPrimalGradient(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float3* primal_gradient
) {
  const int3 grad_offsets[3] = {{1,0,0}, {0,1,0}, {0,0,1}};

  Voxel voxel_query;
  PrimalDualVariables primal_dual_variable_query;
  float primalp[3], primal;
  bool valid = GetPrimalDualValue(curr_entry, voxel_pos,
                                  blocks, hash_table,
                                  geometry_helper,
                                  &voxel_query,
                                  &primal_dual_variable_query);
  if (! valid
      || voxel_query.weight < EPSILON
      || !primal_dual_variable_query.mask) {
    printf("GetPrimalGradient: Invalid Center\n");
    return false;
  }
  primal = primal_dual_variable_query.sdf_bar;

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    valid = valid && GetPrimalDualValue(curr_entry, voxel_pos + grad_offsets[i],
                                        blocks, hash_table,
                                        geometry_helper,
                                        &voxel_query,
                                        &primal_dual_variable_query);
    primalp[i] = primal_dual_variable_query.sdf_bar;
    if (! valid
        || voxel_query.weight < EPSILON
        || !primal_dual_variable_query.mask) {
      *primal_gradient = make_float3(0);
      return false;
    }
  }

  *primal_gradient = make_float3(primalp[0] - primal,
                                 primalp[1] - primal,
                                 primalp[2] - primal);
  return true;
}


#endif //MESH_HASHING_PRIMAL_DUAL_QUERY_H
