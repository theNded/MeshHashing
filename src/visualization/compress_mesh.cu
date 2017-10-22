//
// Created by wei on 17-10-22.
//

#include "visualization/compress_mesh.h"
#include <glog/logging.h>

////////////////////////////////
/// Compress discrete vertices and triangles
__global__
void CollectVerticesAndTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray       blocks,
    Mesh             mesh,
    CompactMesh      compact_mesh) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Voxel &cube = blocks[entry.ptr].voxels[threadIdx.x];

  for (int i = 0; i < N_TRIANGLE; ++i) {
    int triangle_ptrs = cube.triangle_ptrs[i];
    if (triangle_ptrs != FREE_PTR) {
      int3& triangle = mesh.triangle(triangle_ptrs).vertex_ptrs;
      atomicAdd(&compact_mesh.triangles_ref_count()[triangle_ptrs], 1);
      atomicAdd(&compact_mesh.vertices_ref_count()[triangle.x], 1);
      atomicAdd(&compact_mesh.vertices_ref_count()[triangle.y], 1);
      atomicAdd(&compact_mesh.vertices_ref_count()[triangle.z], 1);
    }
  }
}

__global__
void CompressVerticesKernel(Mesh        mesh,
                            CompactMesh compact_mesh,
                            uint           max_vertex_count,
                            uint*          vertex_ref_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < max_vertex_count && compact_mesh.vertices_ref_count()[idx] > 0) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_mesh.vertex_counter(), local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    compact_mesh.vertex_remapper()[idx] = addr;
    compact_mesh.vertices()[addr] = mesh.vertex(idx).pos;
    compact_mesh.normals()[addr]  = mesh.vertex(idx).normal;
    compact_mesh.colors()[addr]   = mesh.vertex(idx).color;

    atomicAdd(vertex_ref_count, mesh.vertex(idx).ref_count);
  }
}

__global__
void CompressTrianglesKernel(Mesh        mesh,
                             CompactMesh compact_mesh,
                             uint max_triangle_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < max_triangle_count && compact_mesh.triangles_ref_count()[idx] > 0) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_mesh.triangle_counter(), local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    int3 vertex_ptrs = mesh.triangle(idx).vertex_ptrs;
    compact_mesh.triangles()[addr].x = compact_mesh.vertex_remapper()[vertex_ptrs.x];
    compact_mesh.triangles()[addr].y = compact_mesh.vertex_remapper()[vertex_ptrs.y];
    compact_mesh.triangles()[addr].z = compact_mesh.vertex_remapper()[vertex_ptrs.z];
  }
}

void CompressMeshImpl(EntryArray& candidate_entries,
                  BlockArray& blocks,
                  Mesh& mesh,
                  CompactMesh & compact_mesh, int3& stats) {
  compact_mesh.Reset();

  int occupied_block_count = candidate_entries.count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    LOG(INFO) << "Before: " << compact_mesh.vertex_count() << " " << compact_mesh.triangle_count();
    CollectVerticesAndTrianglesKernel <<< grid_size, block_size >>> (
        candidate_entries,
            blocks,
            mesh,
            compact_mesh);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((mesh.params().max_vertex_count
                          + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    uint* vertex_ref_count;
    checkCudaErrors(cudaMalloc(&vertex_ref_count, sizeof(uint)));
    checkCudaErrors(cudaMemset(vertex_ref_count, 0, sizeof(uint)));

    CompressVerticesKernel <<< grid_size, block_size >>> (
        mesh,
            compact_mesh,
            mesh.params().max_vertex_count,
            vertex_ref_count);

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());

    uint vertex_ref_count_cpu;
    checkCudaErrors(cudaMemcpy(&vertex_ref_count_cpu, vertex_ref_count, sizeof(uint),
                               cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(vertex_ref_count));

    LOG(INFO) << vertex_ref_count_cpu;
    stats.z = vertex_ref_count_cpu;
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((mesh.params().max_triangle_count
                          + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    CompressTrianglesKernel <<< grid_size, block_size >>> (
        mesh,
            compact_mesh,
            mesh.params().max_triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  LOG(INFO) << "Vertices: " << compact_mesh.vertex_count()
            << "/" << (mesh.params().max_vertex_count - mesh.vertex_heap_count());
  stats.y = compact_mesh.vertex_count();

  LOG(INFO) << "Triangles: " << compact_mesh.triangle_count()
            << "/" << (mesh.params().max_triangle_count - mesh.triangle_heap_count());
  stats.x = compact_mesh.triangle_count();
}
