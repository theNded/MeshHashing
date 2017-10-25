//
// Created by wei on 17-10-25.
//

#ifndef MESH_HASHING_SPATIAL_QUERY_H
#define MESH_HASHING_SPATIAL_QUERY_H

#include <matrix.h>
#include "geometry_helper.h"

#include "core/hash_table.h"
#include "core/block_array.h"


// TODO(wei): refine it
__device__
inline Voxel GetVoxel(const HashTable &hash_table,
                      const BlockArray &blocks,
                      const float3 world_pos,
                      GeometryHelper& geoemtry_helper) {
  HashEntry hash_entry = hash_table.GetEntry(geoemtry_helper.WorldToBlock(world_pos));
  Voxel v;
  if (hash_entry.ptr == FREE_ENTRY) {
    v.ClearSDF();
  } else {
    int3 voxel_pos = geoemtry_helper.WorldToVoxeli(world_pos);
    int i = geoemtry_helper.VoxelPosToIdx(voxel_pos);
    v = blocks[hash_entry.ptr].voxels[i];
  }
  return v;
}

// TODO: put a dummy here
__device__
inline float GetSDF(const HashTable& hash_table,
                    const BlockArray&          blocks,
                    const HashEntry&    curr_entry,
                    const uint3         voxel_local_pos,
                    float &weight,
                    GeometryHelper& geoemtry_helper) {
  float sdf = 0.0; weight = 0;
  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = geoemtry_helper.VoxelLocalPosToIdx(voxel_local_pos);
    const Voxel& v = blocks[curr_entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return 0;
    uint i = geoemtry_helper.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);

    const Voxel &v = blocks[entry.ptr].voxels[i];
    sdf = v.sdf;
    weight = v.weight;
  }

  return sdf;
}

__device__
inline Voxel& GetVoxelRef(const HashTable& hash_table,
                          BlockArray&          blocks,
                          const HashEntry&    curr_entry,
                          const uint3         voxel_local_pos,
                          GeometryHelper& geoemtry_helper) {

  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = geoemtry_helper.VoxelLocalPosToIdx(voxel_local_pos);
    return blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetVoxelRef: should never reach here!\n");
    }
    uint i = geoemtry_helper.VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);
    return blocks[entry.ptr].voxels[i];
  }
}

#endif //MESH_HASHING_SPATIAL_QUERY_H
