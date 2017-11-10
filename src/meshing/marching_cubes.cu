#include <device_launch_parameters.h>
#include "meshing/marching_cubes.h"
#include "geometry/spatial_query.h"
#include "visualization/color_util.h"
//#define REDUCTION

////////////////////
/// class MappingEngine - meshing
////////////////////

////////////////////
/// Device code
////////////////////

/// Marching Cubes
__device__
float3 VertexIntersection(const float3 &p1, const float3 p2,
                          const float &v1, const float &v2,
                          const float &isolevel) {
  if (fabs(v1 - isolevel) < 0.008) return p1;
  if (fabs(v2 - isolevel) < 0.008) return p2;
  float mu = (isolevel - v1) / (v2 - v1);

  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
inline int AllocateVertexWithMutex(
    MeshUnit &mesh_unit,
    uint  &vertex_idx,
    const float3 &vertex_pos,
    Mesh &mesh,
    BlockArray &blocks,
    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
) {
  int ptr = mesh_unit.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&mesh_unit.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0) {
    Voxel voxel_query;
    bool valid = GetSpatialValue(vertex_pos, blocks, hash_table,
                                 geometry_helper, &voxel_query);
    mesh_unit.vertex_ptrs[vertex_idx] = ptr;
    mesh.vertex(ptr).pos = vertex_pos;
    mesh.vertex(ptr).radius = sqrtf(1.0f / voxel_query.inv_sigma2);
    mesh.vertex(ptr).normal = GetSpatialSDFGradient(
        vertex_pos,
        blocks, hash_table,
        geometry_helper
    );

    float rho = voxel_query.a/(voxel_query.a + voxel_query.b);
    //printf("%f %f\n", voxel_query.a, voxel_query.b);
    mesh.vertex(ptr).color = ValToRGB(rho, 0, 1.0f);
    //mesh.vertex(ptr).color = ValToRGB(voxel_query.inv_sigma2/10000.0f, 0, 1.0f);
  }
  return ptr;
}

__global__
void SurfelExtractionKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];

  int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3   voxel_pos = voxel_base_pos + make_int3(offset);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  Voxel& this_voxel = block.voxels[threadIdx.x];
  //////////
  /// 1. Read the scalar values, see mc_tables.h
  const int kVertexCount = 8;
  const float kVoxelSize = geometry_helper.voxel_size;
  const float kThreshold = 0.20f;
  const float kIsoLevel = 0;

  float  d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;
  this_mesh_unit.prev_cube_idx = this_mesh_unit.curr_cube_idx;
  this_mesh_unit.curr_cube_idx = 0;

  // inlier ratio
//  if (this_voxel.inv_sigma2 < 5.0f) return;
  float rho = this_voxel.a / (this_voxel.a + this_voxel.b);
  if (rho < 0.2f || this_voxel.inv_sigma2 < squaref(0.33f / kVoxelSize))
    return;

  /// Check 8 corners of a cube: are they valid?
  Voxel voxel_query;
  for (int i = 0; i < kVertexCount; ++i) {
    if (! GetVoxelValue(entry, voxel_pos + kVtxOffset[i],
                        blocks, hash_table,
                        geometry_helper, &voxel_query)) {
      return;
    }

    d[i] = voxel_query.sdf;
    if (fabs(d[i]) > kThreshold) return;

    rho = voxel_query.a / (voxel_query.a + voxel_query.b);
    if (rho < 0.2f || voxel_query.inv_sigma2 < squaref(0.33f / kVoxelSize))
      return;
    if (d[i] < kIsoLevel) cube_index |= (1 << i);
    p[i] = world_pos + kVoxelSize * make_float3(kVtxOffset[i]);
  }

  this_mesh_unit.curr_cube_idx = cube_index;
  if (cube_index == 0 || cube_index == 255) return;

  const int kEdgeCount = 12;
#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kCubeEdges[cube_index] & (1 << i)) {
      int2 edge_endpoint_vertices = kEdgeEndpointVertices[i];
      uint4 edge_cube_owner_offset = kEdgeOwnerCubeOffset[i];

      // Special noise-bit interpolation here: extrapolation
      float3 vertex_pos = VertexIntersection(
          p[edge_endpoint_vertices.x],
          p[edge_endpoint_vertices.y],
          d[edge_endpoint_vertices.x],
          d[edge_endpoint_vertices.y],
          kIsoLevel);

      MeshUnit &mesh_unit = GetMeshUnitRef(
          entry,
          voxel_pos + make_int3(edge_cube_owner_offset.x,
                                edge_cube_owner_offset.y,
                                edge_cube_owner_offset.z),
          blocks, hash_table,
          geometry_helper);

      AllocateVertexWithMutex(
          mesh_unit,
          edge_cube_owner_offset.w,
          vertex_pos,
          mesh,
          blocks,
          hash_table, geometry_helper,
          enable_sdf_gradient);
    }
  }
}

__device__
inline bool IsInner(uint3 offset) {
  return (offset.x >= 1 && offset.y >= 1 && offset.z >= 1
          && offset.x < BLOCK_SIDE_LENGTH - 1
          && offset.y < BLOCK_SIDE_LENGTH - 1
          && offset.z < BLOCK_SIDE_LENGTH - 1);
}

__global__
void TriangleExtractionKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];
  if (threadIdx.x == 0) {
    block.boundary_surfel_count = 0;
    block.inner_surfel_count = 0;
  }
  __syncthreads();

  int3   voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3  offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3   voxel_pos = voxel_base_pos + make_int3(offset);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  MeshUnit &this_mesh_unit = block.mesh_units[threadIdx.x];
  bool is_inner = IsInner(offset);
  for (int i = 0; i < 3; ++i) {
    if (this_mesh_unit.vertex_ptrs[i] >= 0) {
      if (is_inner) {
        atomicAdd(&block.inner_surfel_count, 1);
      } else {
        atomicAdd(&block.boundary_surfel_count, 1);
      }
    }
  }
  /// Cube type unchanged: NO need to update triangles
//  if (this_cube.curr_cube_idx == this_cube.prev_cube_idx) {
//    blocks[entry.ptr].voxels[local_idx].stats.duration += 1.0f;
//    return;
//  }
//  blocks[entry.ptr].voxels[local_idx].stats.duration = 0;

  if (this_mesh_unit.curr_cube_idx == 0
      || this_mesh_unit.curr_cube_idx == 255) {
    return;
  }

  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  /// This operation is in 2-pass
  /// pass2: Assign
  const int kEdgeCount = 12;
  int vertex_ptrs[kEdgeCount];

#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kCubeEdges[this_mesh_unit.curr_cube_idx] & (1 << i)) {
      uint4 edge_owner_cube_offset = kEdgeOwnerCubeOffset[i];

      MeshUnit &mesh_unit  = GetMeshUnitRef(
          entry,
          voxel_pos + make_int3(edge_owner_cube_offset.x,
                                edge_owner_cube_offset.y,
                                edge_owner_cube_offset.z),
          blocks,
          hash_table,
          geometry_helper);

      vertex_ptrs[i] = mesh_unit.GetVertex(edge_owner_cube_offset.w);
      mesh_unit.ResetMutexes();
    }
  }

  //////////
  /// 3. Assign triangles
  int i = 0;
  for (int t = 0;
       kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t] != -1;
       t += 3, ++i) {
    int triangle_ptr = this_mesh_unit.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) {
      triangle_ptr = mesh.AllocTriangle();
    } else {
      mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
    }
    this_mesh_unit.triangle_ptrs[i] = triangle_ptr;

    mesh.AssignTriangle(
        mesh.triangle(triangle_ptr),
        make_int3(vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 0]],
                  vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 1]],
                  vertex_ptrs[kTriangleVertexEdge[this_mesh_unit.curr_cube_idx][t + 2]]));
    if (!enable_sdf_gradient) {
      mesh.ComputeTriangleNormal(mesh.triangle(triangle_ptr));
    }
  }
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];

  int i = 0;
  for (int t = 0;
       kTriangleVertexEdge[mesh_unit.curr_cube_idx][t] != -1;
       t += 3, ++i);

  for (; i < N_TRIANGLE; ++i) {
    int triangle_ptr = mesh_unit.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
    mesh.triangle(triangle_ptr).Clear();
    mesh.FreeTriangle(triangle_ptr);
    mesh_unit.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleVerticesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  MeshUnit &mesh_unit = blocks[entry.ptr].mesh_units[threadIdx.x];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (mesh_unit.vertex_ptrs[i] != FREE_PTR &&
        mesh.vertex(mesh_unit.vertex_ptrs[i]).ref_count == 0) {
      mesh.vertex(mesh_unit.vertex_ptrs[i]).Clear();
      mesh.FreeVertex(mesh_unit.vertex_ptrs[i]);
      mesh_unit.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

////////////////////
/// Host code
////////////////////
float MarchingCubes(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
) {
  uint occupied_block_count = candidate_entries.count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count == 0)
    return -1;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// Use divide and conquer to avoid read-write conflict
  Timer timer;
  timer.Tick();
  SurfelExtractionKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          hash_table,
          geometry_helper,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass1_seconds = timer.Tock();
  LOG(INFO) << "Pass1 duration: " << pass1_seconds;

  timer.Tick();
  TriangleExtractionKernel << < grid_size, block_size >> > (
      candidate_entries,
          blocks,
          mesh,
          hash_table,
          geometry_helper,
          enable_sdf_gradient);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass2_seconds = timer.Tock();
  LOG(INFO) << "Pass2 duration: " << pass2_seconds;

  RecycleTrianglesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel << < grid_size, block_size >> > (
      candidate_entries, blocks, mesh);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  return (float)(pass1_seconds + pass2_seconds);
}
