#include "map.h"
#include "mc_tables.h"
#include "gradient.h"

#include <glog/logging.h>
#include <curand_kernel.h>

/// Planar fitting
__global__
void PlaneFittingKernel(
    HashTableGPU        hash_table,
    CompactHashTableGPU compact_hash_table,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    float3              camera_pos) {

  // Step 1: vertices collection
  __shared__ int            vertex_count;
  __shared__ curandState_t  state;
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  if (threadIdx.x == 0) {
    vertex_count = 0;
    curand_init((unsigned long long)clock() + blockIdx.x, 0, 0, &state);
  }
  __syncthreads();

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = VoxelToWorld(voxel_pos);

  Cube &this_cube = blocks[entry.ptr].cubes[local_idx];

  for (int i = 0; i < 3; ++i) {
    if (this_cube.vertex_ptrs[i] != FREE_PTR) {
      int addr = atomicAdd(&vertex_count, 1);
      blocks[entry.ptr].vertices[addr] = mesh.vertices[this_cube.vertex_ptrs[i]].pos;
    };
  }

  __syncthreads();
  if (vertex_count < 3) {
    blocks[entry.ptr].ratio = 0;
    return;
  }

  //printf("%d\n", vertex_index);
  // Step 2: RANSAC plane finding
  if (threadIdx.x == 0) {
    int i0, i1, i2;
    float3 v0, v1, v2;
    for (int iter = 0; iter < 10; ++iter) {
      // TODO: introduce a assistance array to avoid infinite while loop
      i0 = curand(&state) % vertex_count;
      do {
        i1 = curand(&state) % vertex_count;
      } while (i1 == i0);
      do {
        i2 = curand(&state) % vertex_count;
      } while (i2 == i0 || i2 == i1);

      v0 = blocks[entry.ptr].vertices[i0];
      v1 = blocks[entry.ptr].vertices[i1];
      v2 = blocks[entry.ptr].vertices[i2];

      float3 n = cross(v1 - v0, v2 - v0);
      n = n / length(n);
      if (dot(v1 - camera_pos, n) > 0) n = -n;
      float d = - dot(n, v0);

      int inliner = 0;
      for (int j = 0; j < vertex_count; ++j) {
        float dist = dot(n, (blocks[entry.ptr].vertices[j] - v0));
        if (dist < 0.002) inliner++;
      }

      float ratio = float(inliner) / vertex_count;
      if (ratio > 0.9) {
        blocks[entry.ptr].n = n;
        blocks[entry.ptr].d = d;
        blocks[entry.ptr].ratio = ratio;
        return;
      }
    }

    blocks[entry.ptr].ratio = 0;
  }
}

__global__
void PlaneMergingKernel(
    HashTableGPU        hash_table,
    CompactHashTableGPU compact_hash_table,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    float3              camera_pos) {

  int entry_count = *compact_hash_table.compacted_entry_counter;

  curandState_t state;
  curand_init((unsigned long long)clock(), 0, 0, &state);
  for (int iter = 0; iter < 100; ++iter) {
    int i = (int)(curand_uniform(&state) * entry_count);
    const HashEntry &entry = compact_hash_table.compacted_entries[i];
    float3 n = blocks[entry.ptr].n;
    float d = blocks[entry.ptr].d;
    float ratio = blocks[entry.ptr].ratio;
    if (ratio < 0.5) continue;

    for (int j = 0; j < entry_count; ++j) {
      const HashEntry &entryj = compact_hash_table.compacted_entries[j];
      float3 nj = blocks[entryj.ptr].n;
      float dj = blocks[entryj.ptr].d;
      float ratioj = blocks[entryj.ptr].ratio;

      if (dot(nj, n) > 0.95 && fabs(dj - d) < 0.002) {
        blocks[entryj.ptr].n = n;
        blocks[entryj.ptr].d = d;
      }
      //printf("(%f, %f, %f, %f), %f\n", n.x, n.y, n.z, d, ratio);
    }
  }
}

__global__
void SDFRefiningKernel(
    HashTableGPU        hash_table,
    CompactHashTableGPU compact_hash_table,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    float3              camera_pos) {
// Step 3: Resample SDF field

  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = VoxelToWorld(voxel_pos);

  if (blocks[entry.ptr].voxels[local_idx].weight == 0 || blocks[entry.ptr].ratio == 0) return;

  Cube &this_cube = blocks[entry.ptr].cubes[local_idx];

  float osdf = blocks[entry.ptr].voxels[local_idx].sdf;
  float nsdf = dot(blocks[entry.ptr].n, world_pos) + blocks[entry.ptr].d;

  if (fabs(nsdf) < 0.002 && fabs(nsdf - osdf) < 0.002f) {
    blocks[entry.ptr].voxels[local_idx].sdf = nsdf;

    if (sign(nsdf) == sign(osdf))
      return;

    // Check simplicity
    /// Change value if triangles becomes simpler
    int prev_triangles = 0;
    for (int t = 0; kTriangleTable[this_cube.curr_index][t] != -1; t += 3, ++prev_triangles);

    int cube_index = 0;
    const int kVertexCount = 8;
    const float kVoxelSize = kSDFParams.voxel_size;
    const float kThreshold = 0.2f;
    const float kIsoLevel = 0;
    float d[kVertexCount];
    for (int i = 0; i < kVertexCount; ++i) {
      uint3 offset = make_uint3(kVertexCubeTable[i][0],
                                kVertexCubeTable[i][1],
                                kVertexCubeTable[i][2]);
      Voxel v = GetVoxel(hash_table, blocks, entry, voxel_local_pos + offset);
      if (v.weight == 0) return;
      d[i] = v.sdf;
      if (fabs(d[i]) > kThreshold) return;

      if (d[i] < kIsoLevel) cube_index |= (1 << i);
    }

    int curr_triangles = 0;
    for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3, ++curr_triangles);

//    if (curr_triangles >= prev_triangles)
//      blocks[entry.ptr].voxels[local_idx].sdf = osdf;
  }
}


/// Assume entries are compressed
void Map::PlaneFitting(float3 camera_pos) {
  LOG(INFO) << "Planar fitting";
  int occupied_block_count = compact_hash_table_.entry_count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    PlaneFittingKernel <<< grid_size, block_size >>> (
        hash_table_.gpu_data(),
            compact_hash_table_.gpu_data(),
            blocks_.gpu_data(),
            mesh_.gpu_data(),
            camera_pos); // For normal direction determination
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(1, 1);
    const dim3 block_size(1, 1);

    PlaneMergingKernel <<< grid_size, block_size >>> (
        hash_table_.gpu_data(),
            compact_hash_table_.gpu_data(),
            blocks_.gpu_data(),
            mesh_.gpu_data(),
            camera_pos); // For normal direction determination
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    SDFRefiningKernel <<< grid_size, block_size >>> (
        hash_table_.gpu_data(),
            compact_hash_table_.gpu_data(),
            blocks_.gpu_data(),
            mesh_.gpu_data(),
            camera_pos); // For normal direction determination
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}
