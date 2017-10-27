//
// Created by wei on 17-10-23.
//

#include "core/common.h"
#include "core/entry_array.h"

#include "visualization/bounding_box.h"
#include "visualization/extract_bounding_box.h"

#include "geometry/geometry_helper.h"

#include "helper_cuda.h"

__device__
const static int3 kEdgeOffsets[24] = {
    {0, 0, 0}, {0, 0, 1},
    {0, 0, 1}, {1, 0, 1},
    {1, 0, 1}, {1, 0, 0},
    {1, 0, 0}, {0, 0, 0},

    {0, 1, 0}, {0, 1, 1},
    {0, 1, 1}, {1, 1, 1},
    {1, 1, 1}, {1, 1, 0},
    {1, 1, 0}, {0, 1, 0},

    {0, 0, 0}, {0, 1, 0},
    {0, 0, 1}, {0, 1, 1},
    {1, 0, 1}, {1, 1, 1},
    {1, 0, 0}, {1, 1, 0}
};

__global__
void ExtractBoundingBoxKernel(
    EntryArray    candidate_entries,
    BoundingBox   bounding_box,
    GeometryHelper geometry_helper) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  HashEntry& entry = candidate_entries[idx];

  int3 voxel_base_pos   = geometry_helper.BlockToVoxel(entry.pos);
  float3 world_base_pos = geometry_helper.VoxelToWorld(voxel_base_pos)
                          - make_float3(0.5f) * geometry_helper.voxel_size;

  float s = geometry_helper.voxel_size * BLOCK_SIDE_LENGTH;
  int addr = atomicAdd(bounding_box.vertex_counter(), 24);
  for (int i = 0; i < 24; i ++) {
    bounding_box.vertices()[addr + i]
        = world_base_pos + s * make_float3(kEdgeOffsets[i]);
  }
}

void ExtractBoundingBox(EntryArray& candidate_entries,
                        BoundingBox& bounding_box,
                        GeometryHelper& geometry_helper) {
  bounding_box.Reset();
  int occupied_block_count = candidate_entries.count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size((occupied_block_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    ExtractBoundingBoxKernel <<< grid_size, block_size >>> (
        candidate_entries, bounding_box, geometry_helper);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}
