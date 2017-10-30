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
    Voxel &voxel,
    uint  &vertex_idx,
    const float3 &vertex_pos,
    Mesh &mesh,

    BlockArray &blocks,

    const HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
) {
  int ptr = voxel.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&voxel.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0) {
    Voxel voxel_query;
    GetSpatialValue(vertex_pos, blocks, hash_table,
                    geometry_helper, &voxel_query);
    voxel.vertex_ptrs[vertex_idx] = ptr;
    mesh.vertex(ptr).pos = vertex_pos;
    mesh.vertex(ptr).color = make_float3(voxel_query.color) / 255.0;
    if (enable_sdf_gradient) {
      mesh.vertex(ptr).normal = GetSpatialSDFGradient(vertex_pos,
                                                      blocks, hash_table,
                                                      geometry_helper);
    }
#ifdef STATS
    float3 val = ValToRGB(stats.duration, 0, 100);
    mesh.vertex(ptr).color = make_float3(val.x, val.y, val.z);
#endif
  }
  return ptr;
}

__device__
void RefineMesh(short &prev_cube, short &curr_cube, float d[8], int is_noise_bit[8]) {
  float kTr = 0.0075;

  /// Step 1: temporal
  short temporal_diff = curr_cube ^prev_cube;
  int dist = 0;
  while (temporal_diff) {
    temporal_diff &= (temporal_diff - 1);
    dist++;
  }
  if (dist > 3) return;

  /// Step 2: Spatially closest
  float min_dist = 1e10;
  int min_idx = -1;
  for (int i = 0; i < 6; ++i) {
    short spatial_diff = curr_cube ^kRegularCubeIndices[i];
    short hamming_dist = 0;
    float euclid_dist;

    for (int j = 0; j < 8; ++j) {
      short mask = (1 << j);
      if (mask & spatial_diff) {
        hamming_dist++;
        euclid_dist += fabs(d[j]);
        if (hamming_dist > 3) break;
      }
    }

    if (hamming_dist <= 3 && euclid_dist < min_dist) {
      min_dist = euclid_dist;
      min_idx = i;
    }
  }
  if (min_idx < 0) return;

  /// Step 3: Valid?
  int noise_bit[3];
  short hamming_dist = 0;
  short binary_xor = curr_cube ^kRegularCubeIndices[min_idx];
  for (int j = 0; j < 8; ++j) {
    short mask = (1 << j);
    if (mask & binary_xor) {
      noise_bit[hamming_dist] = j;
      hamming_dist++;
    }
  }

  for (int j = 0; j < hamming_dist; ++j) {
    if (fabs(d[noise_bit[j]]) > kTr) {
      return;
    }
  }

  for (int i = 0; i < 8; ++i) {
    is_noise_bit[i] = 0;
  }
  for (int j = 0; j < hamming_dist; ++j) {
    //d[noise_bit[j]] = - d[noise_bit[j]];
    is_noise_bit[noise_bit[j]] = 1;
  }
  curr_cube = kRegularCubeIndices[min_idx];
}

__global__
void MarchingCubesPass1Kernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
) {

  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = geometry_helper.DevectorizeIndex(local_idx);
  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];

  //////////
  /// 1. Read the scalar values, see mc_tables.h
  const int kVertexCount = 8;
  const float kVoxelSize = geometry_helper.voxel_size;
  const float kThreshold = 0.20f;
  const float kIsoLevel = 0;

  float d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;
  this_voxel.prev_cube_idx = this_voxel.curr_cube_idx;
  this_voxel.curr_cube_idx = 0;

  /// Check 8 corners of a cube: are they valid?
  Voxel voxel_query;
  for (int i = 0; i < kVertexCount; ++i) {
    if (! GetVoxelValue(entry, voxel_pos + kVtxOffset[i],
                        blocks, hash_table,
                        geometry_helper, &voxel_query))
      return;
//
    if (voxel_query.weight < 3.0f)
      return;

    d[i] = voxel_query.sdf;
    if (fabs(d[i]) > kThreshold) return;

    if (d[i] < kIsoLevel) cube_index |= (1 << i);
    p[i] = world_pos + kVoxelSize * make_float3(kVtxOffset[i]);
  }
  this_voxel.curr_cube_idx = cube_index;
  if (cube_index == 0 || cube_index == 255) return;

  //int is_noise_bit[8];
  //RefineMesh(this_voxel.prev_cube_idx, this_voxel.curr_cube_idx, d, is_noise_bit);
  //cube_index = this_voxel.curr_cube_idx;

  const int kEdgeCount = 12;
#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kEdgeTable[cube_index] & (1 << i)) {
      int2 v_idx = kEdgeVertexTable[i];
      uint4 c_idx = kEdgeCubeTable[i];

      // Special noise-bit interpolation here: extrapolation
      float3 vertex_pos;
      vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                      d[v_idx.x], d[v_idx.y], kIsoLevel);

      Voxel &voxel = GetVoxelRef(entry,
                                 voxel_pos + make_int3(c_idx.x, c_idx.y, c_idx.z),
                                 blocks, hash_table,
                                 geometry_helper);
      AllocateVertexWithMutex(voxel, c_idx.w, vertex_pos, mesh,
                              blocks, hash_table, geometry_helper,
                              enable_sdf_gradient);
    }
  }
}

__global__
void MarchingCubesPass2Kernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    bool enable_sdf_gradient
) {

  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = geometry_helper.DevectorizeIndex(local_idx);
  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = geometry_helper.VoxelToWorld(voxel_pos);

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];

  /// Cube type unchanged: NO need to update triangles
//  if (this_cube.curr_cube_idx == this_cube.prev_cube_idx) {
//    blocks[entry.ptr].voxels[local_idx].stats.duration += 1.0f;
//    return;
//  }
//  blocks[entry.ptr].voxels[local_idx].stats.duration = 0;

  if (this_voxel.curr_cube_idx == 0 || this_voxel.curr_cube_idx == 255) {
    return;
  }

  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  /// This operation is in 2-pass
  /// pass2: Assign
  const int kEdgeCount = 12;
  int vertex_ptr[kEdgeCount];

#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kEdgeTable[this_voxel.curr_cube_idx] & (1 << i)) {
      uint4 c_idx = kEdgeCubeTable[i];
      int3 voxel_p = voxel_pos + make_int3(c_idx.x, c_idx.y, c_idx.z);
      Voxel &voxel  = GetVoxelRef(entry, voxel_p, blocks, hash_table, geometry_helper);
      vertex_ptr[i] = voxel.GetVertex(c_idx.w);
      Vertex& v = mesh.vertex(vertex_ptr[i]);
      voxel.ResetMutexes();
    }
  }

  //////////
  /// 3. Assign triangles
  int i = 0;
  for (int t = 0;
       kTriangleTable[this_voxel.curr_cube_idx][t] != -1;
       t += 3, ++i) {
    int triangle_ptr = this_voxel.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) {
      triangle_ptr = mesh.AllocTriangle();
    } else {
      mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
    }
    this_voxel.triangle_ptrs[i] = triangle_ptr;

    mesh.AssignTriangle(mesh.triangle(triangle_ptr),
                        make_int3(vertex_ptr[kTriangleTable[this_voxel.curr_cube_idx][t + 0]],
                                  vertex_ptr[kTriangleTable[this_voxel.curr_cube_idx][t + 1]],
                                  vertex_ptr[kTriangleTable[this_voxel.curr_cube_idx][t + 2]]));
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

  const uint local_idx = threadIdx.x;  //inside an SDF block
  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];

  int i = 0;
  for (int t = 0; kTriangleTable[voxel.curr_cube_idx][t] != -1; t += 3, ++i);

  for (; i < N_TRIANGLE; ++i) {
    int triangle_ptr = voxel.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh.ReleaseTriangle(mesh.triangle(triangle_ptr));
    mesh.triangle(triangle_ptr).Clear();
    mesh.FreeTriangle(triangle_ptr);
    voxel.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleVerticesKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    Mesh mesh
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (voxel.vertex_ptrs[i] != FREE_PTR &&
        mesh.vertex(voxel.vertex_ptrs[i]).ref_count == 0) {
      mesh.vertex(voxel.vertex_ptrs[i]).Clear();
      mesh.FreeVertex(voxel.vertex_ptrs[i]);
      voxel.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

#ifdef STATS
__global__
void UpdateStatisticsKernel(HashTable        hash_table,
                            EntryArray candidate_entries,
                            BlockArray           blocks) {

  const HashEntry &entry = candidate_entries.entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = DevectorizeIndex(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);

  const int3 offset[] = {
      make_int3(1, 0, 0),
      make_int3(-1, 0, 0),
      make_int3(0, 1, 0),
      make_int3(0, -1, 0),
      make_int3(0, 0, 1),
      make_int3(0, 0, -1)
  };

  float sdf = blocks[entry.ptr].voxels[local_idx].sdf;
  float laplacian = 8 * sdf;

  for (int i = 0; i < 3; ++i) {
    Voxel vp = GetVoxel(hash_table, blocks, VoxelToWorld(voxel_pos + offset[2*i]));
    Voxel vn = GetVoxel(hash_table, blocks, VoxelToWorld(voxel_pos + offset[2*i+1]));
    if (vp.weight == 0 || vn.weight == 0) {
      blocks[entry.ptr].voxels[local_idx].stats.laplacian = 1;
      return;
    }
    laplacian += vp.sdf + vn.sdf;
  }

  blocks[entry.ptr].voxels[local_idx].stats.laplacian = laplacian;
}
#endif

////////////////////
/// Host code
////////////////////
void MarchingCubes(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    bool enable_sdf_gradient
) {
  uint occupied_block_count = candidate_entries.count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count <= 0)
    return;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// First update statistics
#ifdef STATS
  UpdateStatisticsKernel<<<grid_size, block_size>>>(
      hash_table,
          candidate_entries,
          blocks);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
#endif

  /// Use divide and conquer to avoid read-write conflict
  Timer timer;
  timer.Tick();
  MarchingCubesPass1Kernel << < grid_size, block_size >> > (
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
  MarchingCubesPass2Kernel << < grid_size, block_size >> > (
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
}
