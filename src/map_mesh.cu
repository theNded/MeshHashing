#include <glog/logging.h>
#include <unordered_map>
#include "color_util.h"
#include <chrono>
#include <timer.h>
#include <ctime>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

#include "mc_tables.h"
#include "map.h"
#include "gradient.h"

//#define REDUCTION

////////////////////
/// class Map - meshing
////////////////////

////////////////////
/// Device code
////////////////////

/// Marching Cubes
__device__
float3 VertexIntersection(const float3& p1, const float3 p2,
                          const float&  v1, const float& v2,
                          const float& isolevel) {
  if (fabs(v1 - isolevel) < 0.008) return p1;
  if (fabs(v2 - isolevel) < 0.008) return p2;
  float mu = (isolevel - v1) / (v2 - v1);

  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
inline int AllocateVertexWithMutex(const HashTableGPU &hash_table,
                                   BlocksGPU &blocks,
                                   MeshGPU& mesh,
                                   Voxel& voxel,
                                   uint& vertex_idx,
                                   const float3& vertex_pos,
                                   bool use_fine_gradient,
                                   CoordinateConverter& converter) {
  int ptr = voxel.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&voxel.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0) {
    voxel.vertex_ptrs[vertex_idx] = ptr;
    mesh.vertices[ptr].pos = vertex_pos;
    if (use_fine_gradient) {
      mesh.vertices[ptr].normal = GradientAtPoint(hash_table, blocks, vertex_pos, converter);
    }

    float sdf;
    Stat  stats;
    uchar3 color;
    TrilinearInterpolation(hash_table, blocks, vertex_pos, sdf, stats, color, converter);
    float3 val = ValToRGB(stats.duration, 0, 100);
    mesh.vertices[ptr].color = make_float3(val.x, val.y, val.z);
  }

  return ptr;
}

__device__
inline int GetVertex(Voxel& voxel, uint& vertex_idx) {
  voxel.ResetMutexes();// ???

  // It is guaranteed to be non-negative
  return voxel.vertex_ptrs[vertex_idx];
}

__device__
void RefineMesh(short& prev_cube, short& curr_cube, float d[8], int is_noise_bit[8]) {
  float kTr = 0.0075;

  /// Step 1: temporal
  short temporal_diff = curr_cube ^ prev_cube;
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
    short spatial_diff = curr_cube ^ kRegularCubeIndices[i];
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
  short binary_xor = curr_cube ^ kRegularCubeIndices[min_idx];
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
    HashTableGPU        hash_table,
    CandidateEntryPoolGPU candidate_entries,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    bool                use_fine_gradient,
    CoordinateConverter converter) {

  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = converter.BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = converter.IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = converter.VoxelToWorld(voxel_pos);

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];

  //////////
  /// 1. Read the scalar values, see mc_tables.h
  const int   kVertexCount = 8;
  const float kVoxelSize   = converter.voxel_size;
  const float kThreshold   = 0.2f;
  const float kIsoLevel    = 0;

  float  d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;
  this_voxel.prev_index = this_voxel.curr_index;
  this_voxel.curr_index = 0;

  /// Check 8 corners of a cube: are they valid?
  for (int i = 0; i < kVertexCount; ++i) {
    uint3 offset = make_uint3(kVtxOffset[i]);
    float weight;

    d[i] = GetSDF(hash_table, blocks, entry, voxel_local_pos + offset, weight, converter);
    if (weight < EPSILON)
      return;

    if (fabs(d[i]) > kThreshold) return;

    if (d[i] < kIsoLevel) cube_index |= (1 << i);
    p[i] = world_pos + kVoxelSize * make_float3(offset);
  }
  this_voxel.curr_index = cube_index;
  if (cube_index == 0 || cube_index == 255) return;

  //int is_noise_bit[8];
  //RefineMesh(this_voxel.prev_index, this_voxel.curr_index, d, is_noise_bit);
  //cube_index = this_voxel.curr_index;

  const int kEdgeCount = 12;
#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kEdgeTable[cube_index] & (1 << i)) {
      int2  v_idx = kEdgeVertexTable[i];
      uint4 c_idx = kEdgeCubeTable[i];

      // Special noise-bit interpolation here: extrapolation
      float3 vertex_pos;
      vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                      d[v_idx.x], d[v_idx.y], kIsoLevel);

      Voxel &voxel = GetVoxelRef(hash_table, blocks, entry,
                                voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z), converter);
      AllocateVertexWithMutex(hash_table, blocks, mesh,
                              voxel, c_idx.w, vertex_pos,
                              use_fine_gradient, converter);
    }
  }
}

__global__
void MarchingCubesPass2Kernel(
    HashTableGPU        hash_table,
    CandidateEntryPoolGPU candidate_entries,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    bool                use_fine_gradient,
    CoordinateConverter converter) {


  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = converter.BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = converter.IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = converter.VoxelToWorld(voxel_pos);

  Voxel &this_voxel = blocks[entry.ptr].voxels[local_idx];

  /// Cube type unchanged: NO need to update triangles
//  if (this_cube.curr_index == this_cube.prev_index) {
//    blocks[entry.ptr].voxels[local_idx].stats.duration += 1.0f;
//    return;
//  }
//  blocks[entry.ptr].voxels[local_idx].stats.duration = 0;

  if (this_voxel.curr_index == 0 || this_voxel.curr_index == 255) {
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
    if (kEdgeTable[this_voxel.curr_index] & (1 << i)) {
      uint4 c_idx = kEdgeCubeTable[i];
      uint3 voxel_p = voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z);
      Voxel &voxel = GetVoxelRef(hash_table, blocks, entry, voxel_p, converter);
      vertex_ptr[i] = GetVertex(voxel, c_idx.w);
    }
  }

  //////////
  /// 3. Assign triangles
  int i = 0;
  for (int t = 0;
       kTriangleTable[this_voxel.curr_index][t] != -1;
       t += 3, ++i) {
    int triangle_ptr = this_voxel.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) {
      triangle_ptr = mesh.AllocTriangle();
    } else {
      mesh.ReleaseTriangle(mesh.triangles[triangle_ptr]);
    }
    this_voxel.triangle_ptrs[i] = triangle_ptr;

    mesh.AssignTriangle(mesh.triangles[triangle_ptr],
                        make_int3(vertex_ptr[kTriangleTable[this_voxel.curr_index][t + 0]],
                                  vertex_ptr[kTriangleTable[this_voxel.curr_index][t + 1]],
                                  vertex_ptr[kTriangleTable[this_voxel.curr_index][t + 2]]));
    if (! use_fine_gradient) {
      mesh.ComputeTriangleNormal(mesh.triangles[triangle_ptr]);
    }
  }
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(
    CandidateEntryPoolGPU candidate_entries,
    BlocksGPU           blocks,
    MeshGPU             mesh) {
  const HashEntry &entry = candidate_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;  //inside an SDF block
  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];

  int i = 0;
  for (int t = 0; kTriangleTable[voxel.curr_index][t] != -1; t += 3, ++i);

  for (; i < N_TRIANGLE; ++i) {
    int triangle_ptr = voxel.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh.ReleaseTriangle(mesh.triangles[triangle_ptr]);
    mesh.triangles[triangle_ptr].Clear();
    mesh.FreeTriangle(triangle_ptr);
    voxel.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleVerticesKernel(
    CandidateEntryPoolGPU candidate_entries,
    BlocksGPU           blocks,
    MeshGPU             mesh) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (voxel.vertex_ptrs[i] != FREE_PTR &&
        mesh.vertices[voxel.vertex_ptrs[i]].ref_count == 0) {
      mesh.vertices[voxel.vertex_ptrs[i]].Clear();
      mesh.FreeVertex(voxel.vertex_ptrs[i]);
      voxel.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

/// Only update Laplacian at current
#ifdef STATS
__global__
void UpdateStatisticsKernel(HashTableGPU        hash_table,
                            CandidateEntryPoolGPU candidate_entries,
                            BlocksGPU           blocks) {

  const HashEntry &entry = candidate_entries.entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
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
void Map::MarchingCubes() {
  uint occupied_block_count = candidate_entries_.entry_count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count <= 0)
    return;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// First update statistics
#ifdef STATS
  UpdateStatisticsKernel<<<grid_size, block_size>>>(
      hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          blocks_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
#endif

  /// Use divide and conquer to avoid read-write conflict
  Timer timer;
  timer.Tick();
  MarchingCubesPass1Kernel<<<grid_size, block_size>>>(
      hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data(),
          use_fine_gradient_,
          coordinate_converter_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass1_seconds = timer.Tock();
  LOG(INFO) << "Pass1 duration: " << pass1_seconds;
  time_profile_ << pass1_seconds << " ";

  timer.Tick();
  MarchingCubesPass2Kernel<<<grid_size, block_size>>>(
      hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data(),
          use_fine_gradient_,
          coordinate_converter_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  double pass2_seconds = timer.Tock();
  LOG(INFO) << "Pass2 duration: " << pass2_seconds;
  time_profile_ << pass2_seconds << "\n";


  RecycleTrianglesKernel<<<grid_size, block_size>>>(
      candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel<<<grid_size, block_size>>>(
      candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

////////////////////////////////
/// Compress discrete vertices and triangles
__global__
void CollectVerticesAndTrianglesKernel(
    CandidateEntryPoolGPU candidate_entries,
    BlocksGPU           blocks,
    MeshGPU             mesh,
    CompactMeshGPU      compact_mesh) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Voxel &cube = blocks[entry.ptr].voxels[threadIdx.x];

  for (int i = 0; i < N_TRIANGLE; ++i) {
    int triangle_ptrs = cube.triangle_ptrs[i];
    if (triangle_ptrs != FREE_PTR) {
      int3& triangle = mesh.triangles[triangle_ptrs].vertex_ptrs;
      atomicAdd(&compact_mesh.triangles_ref_count[triangle_ptrs], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.x], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.y], 1);
      atomicAdd(&compact_mesh.vertices_ref_count[triangle.z], 1);
    }
  }
}

__global__
void CompressVerticesKernel(MeshGPU        mesh,
                            CompactMeshGPU compact_mesh,
                            uint           max_vertex_count,
                            uint*          vertex_ref_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < max_vertex_count && compact_mesh.vertices_ref_count[idx] > 0) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_mesh.vertex_counter, local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    compact_mesh.vertex_remapper[idx] = addr;
    compact_mesh.vertices[addr] = mesh.vertices[idx].pos;
    compact_mesh.normals[addr]  = mesh.vertices[idx].normal;
    compact_mesh.colors[addr]   = mesh.vertices[idx].color;

    atomicAdd(vertex_ref_count, mesh.vertices[idx].ref_count);
  }
}

__global__
void CompressTrianglesKernel(MeshGPU        mesh,
                             CompactMeshGPU compact_mesh,
                             uint max_triangle_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < max_triangle_count && compact_mesh.triangles_ref_count[idx] > 0) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_mesh.triangle_counter, local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    int3 vertex_ptrs = mesh.triangles[idx].vertex_ptrs;
    compact_mesh.triangles[addr].x = compact_mesh.vertex_remapper[vertex_ptrs.x];
    compact_mesh.triangles[addr].y = compact_mesh.vertex_remapper[vertex_ptrs.y];
    compact_mesh.triangles[addr].z = compact_mesh.vertex_remapper[vertex_ptrs.z];
  }
}

/// Assume this operation is following
/// CollectInFrustumBlocks or
/// CollectAllBlocks
void Map::CompressMesh(int3& stats) {
  compact_mesh_.Reset();

  int occupied_block_count = candidate_entries_.entry_count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    LOG(INFO) << "Before: " << compact_mesh_.vertex_count() << " " << compact_mesh_.triangle_count();
    CollectVerticesAndTrianglesKernel <<< grid_size, block_size >>> (
        candidate_entries_.gpu_data(),
            blocks_.gpu_data(),
            mesh_.gpu_data(),
            compact_mesh_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((mesh_.params().max_vertex_count
                          + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    uint* vertex_ref_count;
    checkCudaErrors(cudaMalloc(&vertex_ref_count, sizeof(uint)));
    checkCudaErrors(cudaMemset(vertex_ref_count, 0, sizeof(uint)));

    CompressVerticesKernel <<< grid_size, block_size >>> (
        mesh_.gpu_data(),
            compact_mesh_.gpu_data(),
            mesh_.params().max_vertex_count,
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
    const dim3 grid_size((mesh_.params().max_triangle_count
                          + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    CompressTrianglesKernel <<< grid_size, block_size >>> (
        mesh_.gpu_data(),
            compact_mesh_.gpu_data(),
            mesh_.params().max_triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  LOG(INFO) << "Vertices: " << compact_mesh_.vertex_count()
            << "/" << (mesh_.params().max_vertex_count - mesh_.vertex_heap_count());
  stats.y = compact_mesh_.vertex_count();

  LOG(INFO) << "Triangles: " << compact_mesh_.triangle_count()
            << "/" << (mesh_.params().max_triangle_count - mesh_.triangle_heap_count());
  stats.x = compact_mesh_.triangle_count();
}

void Map::SaveMesh(std::string path) {
  LOG(INFO) << "Copying data from GPU";

  CollectAllBlocks();
  int3 stats;
  CompressMesh(stats);

  uint compact_vertex_count = compact_mesh_.vertex_count();
  uint compact_triangle_count = compact_mesh_.triangle_count();
  LOG(INFO) << "Vertices: " << compact_vertex_count;
  LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];
  checkCudaErrors(cudaMemcpy(vertices, compact_mesh_.gpu_data().vertices,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normals, compact_mesh_.gpu_data().normals,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, compact_mesh_.gpu_data().triangles,
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  std::stringstream ss;
  LOG(INFO) << "Writing vertices";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss <<  "v " << vertices[i].x << " "
       << vertices[i].y << " "
       << vertices[i].z << "\n";
    out << ss.str();
  }

  if (use_fine_gradient_) {
    LOG(INFO) << "Writing normals";
    for (uint i = 0; i < compact_vertex_count; ++i) {
      ss.str("");
      ss << "vn " << normals[i].x << " "
         << normals[i].y << " "
         << normals[i].z << "\n";
      out << ss.str();
    }
  }

  LOG(INFO) << "Writing faces";
  for (uint i = 0; i < compact_triangle_count; ++i) {
    ss.str("");
    int3 idx = triangles[i] + make_int3(1);
    if (use_fine_gradient_) {
      ss << "f " << idx.x << "//" << idx.x << " "
         << idx.y << "//" << idx.y << " "
         << idx.z << "//" << idx.z << "\n";
    } else {
      ss << "f " << idx.x << " " << idx.y << " " << idx.z << "\n";
    }
    out << ss.str();
  }

  LOG(INFO) << "Finishing vertices";
  delete[] vertices;
  LOG(INFO) << "Finishing normals";
  delete[] normals;
  LOG(INFO) << "Finishing triangles";
  delete[] triangles;
}


void Map::SavePly(std::string path) {
  LOG(INFO) << "Copying data from GPU";

  CollectAllBlocks();
  int3 stats;
  CompressMesh(stats);

  uint compact_vertex_count = compact_mesh_.vertex_count();
  uint compact_triangle_count = compact_mesh_.triangle_count();
  LOG(INFO) << "Vertices: " << compact_vertex_count;
  LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];
  checkCudaErrors(cudaMemcpy(vertices, compact_mesh_.gpu_data().vertices,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normals, compact_mesh_.gpu_data().normals,
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, compact_mesh_.gpu_data().triangles,
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  std::stringstream ss;
  ////// Header
  ss.str("");
  ss << "ply\n"
      "format ascii 1.0\n";
  ss << "element vertex " << compact_vertex_count << "\n";
  ss << "property float x\n"
      "property float y\n"
      "property float z\n"
      "property float nx\n"
      "property float ny\n"
      "property float nz\n";
  ss << "element face " << compact_triangle_count << "\n";
  ss << "property list uchar int vertex_index\n";
  ss << "end_header\n";
  out << ss.str();

  LOG(INFO) << "Writing vertices";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss << vertices[i].x << " "
       << vertices[i].y << " "
       << vertices[i].z << " "
       << normals[i].x << " "
       << normals[i].y << " "
       << normals[i].z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing faces";
  for (uint i = 0; i < compact_triangle_count; ++i) {
    ss.str("");
    int3 idx = triangles[i];
    ss << "3 " << idx.x << " " << idx.y << " " << idx.z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Finishing vertices";
  delete[] vertices;
  LOG(INFO) << "Finishing normals";
  delete[] normals;
  LOG(INFO) << "Finishing triangles";
  delete[] triangles;
}
