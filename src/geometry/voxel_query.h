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
  HashEntry hash_entry = hash_table.GetEntry(geometry_helper.WorldToBlock(world_pos));
  Voxel v;
  if (hash_entry.ptr == FREE_ENTRY) {
    v.ClearSDF();
  } else {
    int3 voxel_pos = geometry_helper.WorldToVoxeli(world_pos);
    int i = geometry_helper.VoxelPosToIdx(voxel_pos);
    v = blocks[hash_entry.ptr].voxels[i];
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
    const uint3 voxel_local_pos,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper
) {

  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = geometry_helper.VoxelLocalPosToIdx(voxel_local_pos);
    return blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef: should never reach here!\n");
    }
    uint i = geometry_helper.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);
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
    const uint3 voxel_local_pos,
    const BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    float &sdf,
    float &weight) {
  sdf = 0.0;
  weight = 0;
  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = geometry_helper.VoxelLocalPosToIdx(voxel_local_pos);
    const Voxel &v = blocks[curr_entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return;
    uint i = geometry_helper.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);

    const Voxel &v = blocks[entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  }
}

#endif //MESH_HASHING_SPATIAL_QUERY_H
