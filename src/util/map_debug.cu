#include "../engine/map.h"
#include <glog/logging.h>

//////////
/// Get bounding boxes
//////////
/// Assume this operation is following
/// CollectInFrustumBlockArray or
/// CollectAllBlockArray
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
void GetBoundingBoxKernel(
        EntryArray candidate_entries,
        BBoxGPU             bboxes,
        CoordinateConverter converter) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  HashEntry& entry = candidate_entries[idx];

  int3 voxel_base_pos   = converter.BlockToVoxel(entry.pos);
  float3 world_base_pos = converter.VoxelToWorld(voxel_base_pos)
                          - make_float3(0.5f) * converter.voxel_size;

  float s = converter.voxel_size * BLOCK_SIDE_LENGTH;
  int addr = atomicAdd(bboxes.vertex_counter, 24);
  for (int i = 0; i < 24; i ++) {
    bboxes.vertices[addr + i] = world_base_pos + s * make_float3(kEdgeOffsets[i]);
  }
}


void Map::GetBoundingBoxes() {
  bbox_.Reset();

  int occupied_block_count = candidate_entries_.entry_count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size((occupied_block_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    GetBoundingBoxKernel <<< grid_size, block_size >>> (
            candidate_entries_,
                    bbox_.gpu_memory(),
                coordinate_converter_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
  LOG(INFO) << bbox_.vertex_count();
}