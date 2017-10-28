//
// Created by wei on 17-10-25.
//

#ifndef GEOMETRY_VOXEL_QUERY_H
#define GEOMETRY_VOXEL_QUERY_H

#include <matrix.h>
#include "geometry_helper.h"

#include "core/hash_table.h"
#include "core/block_array.h"


// TODO(wei): refine it
// function:
// at @param world_pos
// get Voxel in @param blocks
// with the help of @param hash_table and geometry_helper
__device__
inline Voxel GetVoxel(
    const float3 world_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  int3 voxel_pos = geometry_helper.WorldToVoxeli(world_pos);
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);
  HashEntry entry = hash_table.GetEntry(block_pos);

  Voxel v;
  if (entry.ptr == FREE_ENTRY) {
    v.ClearSDF();
  } else {
    uint i = geometry_helper.VectorizeOffset(offset);
    v = blocks[entry.ptr].voxels[i];
  }
  return v;
}

// function:
// block-pos @param curr_entry -> voxel-pos @param voxel_local_pos
// get Voxel in @param blocks
// with the help of @param hash_table and geometry_helper
__device__
inline Voxel &GetVoxelRef(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef: should never reach here!\n");
    }
    uint i = geometry_helper.VectorizeOffset(offset);
    return blocks[entry.ptr].voxels[i];
  }
}

// TODO: put a dummy here
// function:
// block-pos @param curr_entry -> voxel-pos @param voxel_local_pos
// get SDF in @param blocks
// with the help of @param hash_table and geometry_helper
__device__
inline void GetVoxelValue(
    const HashEntry &curr_entry,
    const int3 voxel_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float &sdf,
    float &weight) {
  sdf = 0.0;
  weight = 0;
  int3 block_pos = geometry_helper.VoxelToBlock(voxel_pos);
  uint3 offset = geometry_helper.VoxelToOffset(block_pos, voxel_pos);

  if (curr_entry.pos == block_pos) {
    uint i = geometry_helper.VectorizeOffset(offset);
    const Voxel &v = blocks[curr_entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  } else {
    HashEntry entry = hash_table.GetEntry(block_pos);
    if (entry.ptr == FREE_ENTRY) return;
    uint i = geometry_helper.VectorizeOffset(offset);
    const Voxel &v = blocks[entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  }
}

#endif //MESH_HASHING_SPATIAL_QUERY_H
