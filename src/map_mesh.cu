#include <glog/logging.h>
#include <unordered_map>
#include <color_util.h>
#include <chrono>

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
  if (fabs(v1 - isolevel) < 0.001) return p1;
  if (fabs(v2 - isolevel) < 0.001) return p2;
  float mu = (isolevel - v1) / (v2 - v1);
  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
inline Voxel GetVoxel(const HashTableGPU& hash_table,
                      BlocksGPU&          blocks,
                      const HashEntry&    curr_entry,
                      const uint3         voxel_local_pos) {
  Voxel v; v.Clear();

  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = VoxelLocalPosToIdx(voxel_local_pos);
    v = blocks[curr_entry.ptr].voxels[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return v;
    uint i = VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);

    v = blocks[entry.ptr].voxels[i];
  }

  return v;
}

__device__
inline Cube& GetCube(const HashTableGPU& hash_table,
                     BlocksGPU&          blocks,
                     const HashEntry&    curr_entry,
                     const uint3         voxel_local_pos) {

  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  if (block_offset == make_int3(0)) {
    uint i = VoxelLocalPosToIdx(voxel_local_pos);
    return blocks[curr_entry.ptr].cubes[i];
  } else {
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("GetCube: should never reach here!\n");
    }
    uint i = VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);
    return blocks[entry.ptr].cubes[i];
  }
}

__device__
inline int AllocateVertexLockFree(const HashTableGPU &hash_table,
                                  BlocksGPU& blocks,
                                  MeshGPU& mesh,
                                  Cube& cube,
                                  uint& vertex_idx,
                                  const float3& vertex_pos,
                                  bool use_fine_gradient) {
  int ptr = cube.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    ptr = mesh.AllocVertex();
  }

  cube.vertex_ptrs[vertex_idx] = ptr;
  mesh.vertices[ptr].pos = vertex_pos;
  if (use_fine_gradient) {
    mesh.vertices[ptr].normal = GradientAtPoint(hash_table, blocks, vertex_pos);
  }
  return ptr;
}

__device__
inline int AllocateVertexWithMutex(const HashTableGPU &hash_table,
                                   BlocksGPU &blocks,
                                   MeshGPU& mesh,
                                   Cube& cube,
                                   uint& vertex_idx,
                                   const float3& vertex_pos,
                                   bool use_fine_gradient) {
  int ptr = cube.vertex_ptrs[vertex_idx];
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&cube.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh.AllocVertex();
    } /// Ensure that it is only allocated once
  }

  if (ptr >= 0) {
    cube.vertex_ptrs[vertex_idx] = ptr;
    mesh.vertices[ptr].pos = vertex_pos;
    if (use_fine_gradient) {
      mesh.vertices[ptr].normal = GradientAtPoint(hash_table, blocks, vertex_pos);
    }

    float sdf, entropy;
    uchar3 color;
    TrilinearInterpolation(hash_table, blocks, vertex_pos, sdf, entropy, color);
    float3 val = ValToRGB(entropy, 0, 1);
    mesh.vertices[ptr].color = make_float3(val.x, val.y, val.z);
  }

  return ptr;
}

__device__
inline int GetVertexWithMutex(Cube& cube, uint& vertex_idx) {
  cube.ResetMutexes();
  // It is guaranteed to be non-negative
  return cube.vertex_ptrs[vertex_idx];
}

__device__
inline bool check_mask(uint3 pos, uchar3 mask) {
  return ((pos.x & 1) == mask.x)
         && ((pos.y & 1) == mask.y)
         && ((pos.z & 1) == mask.z);
}

__global__
void MarchingCubesLockFreeKernel(
        HashTableGPU        hash_table,
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh,
        uchar3              mask0,
        uchar3              mask1,
        bool                use_fine_gradient) {

  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  int3 voxel_base_pos   = BlockToVoxel(map_entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = VoxelToWorld(voxel_pos);

  if (!check_mask(voxel_local_pos, mask0)
      && !check_mask(voxel_local_pos, mask1))
    return;

  Cube &this_cube = blocks[map_entry.ptr].cubes[local_idx];

 //////////
  /// 1. Read the scalar values, see mc_tables.h
  Voxel v;
  const int   kVertexCount = 8;
  const float kThreshold = 0.2f;
  const float kIsoLevel = 0;
  const float kVoxelSize = kSDFParams.voxel_size;

  float d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;

#pragma unroll 1
  for (int i = 0; i < kVertexCount; ++i) {
    uint3 offset = make_uint3(kVertexCubeTable[i][0],
                              kVertexCubeTable[i][1],
                              kVertexCubeTable[i][2]);
    v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos + offset);
    if (v.weight() == 0) {
      this_cube.curr_index = 0;
      return;
    }
    p[i] = world_pos + kVoxelSize * make_float3(offset);
    d[i] = v.sdf();

    if (fabs(d[i]) > kThreshold) {
      this_cube.curr_index = 0;
      return;
    }
    if (d[i] < kIsoLevel) cube_index |= (1 << i);
  }
  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255) {
    this_cube.curr_index = 0;
    return;
  }

  //////////
  /// 3. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  const int   kEdgeCount = 12;
  int vertex_ptr[kEdgeCount];
  float3 vertex_pos;

  /// This operation is in 2-pass
  /// pass1: Allocate
#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    int mask = (1 << i);
    if (kEdgeTable[cube_index] & mask) {
      int2 v_idx = make_int2(kEdgeVertexTable[i][0], kEdgeVertexTable[i][1]);
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);

      vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                      d[v_idx.x], d[v_idx.y], kIsoLevel);
      Cube &cube = GetCube(hash_table, blocks, map_entry,
                           voxel_local_pos +
                           make_uint3(c_idx.x, c_idx.y, c_idx.z));
      vertex_ptr[i] = AllocateVertexLockFree(hash_table, blocks, mesh,
                                             cube, c_idx.w, vertex_pos,
                                             use_fine_gradient);
    }
  }

  if (this_cube.curr_index == cube_index) return;
  int i = 0;
  for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3, ++i) {
    int triangle_ptrs = this_cube.triangle_ptrs[i];

    if (triangle_ptrs == FREE_PTR) {
      triangle_ptrs = mesh.AllocTriangle();
    } else {
      mesh.ReleaseTriangle(mesh.triangles[triangle_ptrs]);
    }
    this_cube.triangle_ptrs[i] = triangle_ptrs;
    mesh.AssignTriangle(mesh.triangles[triangle_ptrs],
                        make_int3(vertex_ptr[kTriangleTable[cube_index][t + 0]],
                                  vertex_ptr[kTriangleTable[cube_index][t + 1]],
                                  vertex_ptr[kTriangleTable[cube_index][t + 2]]));
    if (!use_fine_gradient) {
      mesh.ComputeTriangleNormal(mesh.triangles[triangle_ptrs]);
    }
  }
  this_cube.curr_index = cube_index;
}

__global__
void MarchingCubesPass1Kernel(
        HashTableGPU        hash_table,
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh,
        bool                use_fine_gradient) {

  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = VoxelToWorld(voxel_pos);

  Cube &this_cube = blocks[entry.ptr].cubes[local_idx];

  //////////
  /// 1. Read the scalar values, see mc_tables.h
  Voxel v;
  const int   kVertexCount = 8;
  const float kVoxelSize   = kSDFParams.voxel_size;
  const float kThreshold   = 0.2f;
  const float kIsoLevel    = 0;

  float  d[kVertexCount];
  float3 p[kVertexCount];

  short cube_index = 0;
  this_cube.prev_index = this_cube.curr_index;
  this_cube.curr_index = 0;
#pragma unroll 1
  for (int i = 0; i < kVertexCount; ++i) {
    uint3 offset = make_uint3(kVertexCubeTable[i][0],
                              kVertexCubeTable[i][1],
                              kVertexCubeTable[i][2]);
    v = GetVoxel(hash_table, blocks, entry, voxel_local_pos + offset);
    if (v.weight() == 0) return;

    p[i] = world_pos + kVoxelSize * make_float3(offset);
    d[i] = v.sdf();
    if (fabs(d[i]) > kThreshold) return;

    if (d[i] < kIsoLevel) cube_index |= (1 << i);
  }
  this_cube.curr_index = cube_index;

  // Early return does not update vertex positions
  // if (this_cube.curr_index == this_cube.prev_index) return;
  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  /// This operation is in 2-pass
  /// pass1: Allocate
  const int kEdgeCount = 12;

#pragma unroll 1
  for (int i = 0; i < kEdgeCount; ++i) {
    if (kEdgeTable[cube_index] & (1 << i)) {
      int2  v_idx = make_int2(kEdgeVertexTable[i][0], kEdgeVertexTable[i][1]);
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);

      float3 vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                             d[v_idx.x], d[v_idx.y], kIsoLevel);
      Cube &cube = GetCube(hash_table, blocks, entry,
                           voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z));
      AllocateVertexWithMutex(hash_table, blocks, mesh,
                              cube, c_idx.w, vertex_pos,
                              use_fine_gradient);
    }
  }
}

__global__
void MarchingCubesPass2Kernel(
        HashTableGPU        hash_table,
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh,
        bool                use_fine_gradient) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos      = VoxelToWorld(voxel_pos);

  Cube &this_cube       = blocks[entry.ptr].cubes[local_idx];

  /// Cube type unchanged: NO need to update triangles
  if (this_cube.curr_index == this_cube.prev_index) return;
  if (kEdgeTable[this_cube.curr_index] == 0
      || kEdgeTable[this_cube.curr_index] == 255) {
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
    if (kEdgeTable[this_cube.curr_index] & (1 << i)) {
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);
      Cube &cube = GetCube(hash_table, blocks, entry,
                           voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z));
      vertex_ptr[i] = GetVertexWithMutex(cube, c_idx.w);
    }
  }

  //////////
  /// 3. Assign triangles
  for (int t = 0, i = 0;
       kTriangleTable[this_cube.curr_index][t] != -1;
       t += 3, ++i) {
    int triangle_ptr = this_cube.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) {
      triangle_ptr = mesh.AllocTriangle();
    } else {
      mesh.ReleaseTriangle(mesh.triangles[triangle_ptr]);
    }
    this_cube.triangle_ptrs[i] = triangle_ptr;

    mesh.AssignTriangle(mesh.triangles[triangle_ptr],
                        make_int3(vertex_ptr[kTriangleTable[this_cube.curr_index][t + 0]],
                                  vertex_ptr[kTriangleTable[this_cube.curr_index][t + 1]],
                                  vertex_ptr[kTriangleTable[this_cube.curr_index][t + 2]]));
    if (! use_fine_gradient) {
      mesh.ComputeTriangleNormal(mesh.triangles[triangle_ptr]);
    }
  }
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;  //inside an SDF block
  Cube &cube = blocks[entry.ptr].cubes[local_idx];

  int i = 0;
  for (int t = 0; kTriangleTable[cube.curr_index][t] != -1; t += 3, ++i);

  for (; i < Cube::kMaxTrianglesPerCube; ++i) {
    int triangle_ptr = cube.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    /// Clear ref_count of its pointed vertices
    mesh.ReleaseTriangle(mesh.triangles[triangle_ptr]);
    mesh.triangles[triangle_ptr].Clear();
    mesh.FreeTriangle(triangle_ptr);
    cube.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleVerticesKernel(
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  Cube &cube = blocks[entry.ptr].cubes[local_idx];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (cube.vertex_ptrs[i] != FREE_PTR &&
        mesh.vertices[cube.vertex_ptrs[i]].ref_count <= 0) {
      mesh.vertices[cube.vertex_ptrs[i]].Clear();
      mesh.FreeVertex(cube.vertex_ptrs[i]);
      cube.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

__global__
void UpdateStatisticsKernel(HashTableGPU        hash_table,
                            CompactHashTableGPU compact_hash_table,
                            BlocksGPU           blocks) {

  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx   = threadIdx.x;

  int3  voxel_base_pos  = BlockToVoxel(entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  int3 voxel_pos        = voxel_base_pos + make_int3(voxel_local_pos);

  int3 offset[] = {
          make_int3(1, 0, 0),
          make_int3(-1, 0, 0),
          make_int3(0, 1, 0),
          make_int3(0, -1, 0),
          make_int3(0, 0, 1),
          make_int3(0, 0, -1)
  };

  int flag = true;
  float sdf = blocks[entry.ptr].voxels[local_idx].sdf();
  float laplacian = 8 * sdf;

  for (int i = 0; i < 3; ++i) {
    Voxel vp = GetVoxel(hash_table, blocks, VoxelToWorld(voxel_pos + offset[2*i]));
    Voxel vn = GetVoxel(hash_table, blocks, VoxelToWorld(voxel_pos + offset[2*i+1]));
    if (vp.weight() == 0 || vn.weight() == 0) {
      blocks[entry.ptr].voxels[local_idx].stat = 1;
      flag = false;
      break;
    }
   laplacian += vp.sdf() + vn.sdf();
  }

  if (flag) {
    blocks[entry.ptr].voxels[local_idx].stat = laplacian;
  }
}


////////////////////
/// Host code
////////////////////
void Map::MarchingCubes() {
  uint occupied_block_count = compact_hash_table_.entry_count();
  LOG(INFO) << "Marching cubes block count: " << occupied_block_count;
  if (occupied_block_count <= 0)
    return;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  /// First update statistics
//  UpdateStatisticsKernel<<<grid_size, block_size>>>(
//          hash_table_.gpu_data(),
//                  compact_hash_table_.gpu_data(),
//                  blocks_.gpu_data());
//  checkCudaErrors(cudaDeviceSynchronize());
//  checkCudaErrors(cudaGetLastError());

  /// Use divide and conquer to avoid read-write conflict
//#define REDUCTION
#ifndef REDUCTION
  std::ofstream mc_info;
  //mc_info.open("../result/statistics/mc_2pass_gradient.txt", std::fstream::app);
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  MarchingCubesPass1Kernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  mc_info << seconds.count();

  start = std::chrono::system_clock::now();
  MarchingCubesPass2Kernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  end = std::chrono::system_clock::now();
  seconds = end - start;
  mc_info << " " << seconds.count() << "\n";
  mc_info.close();


#else
  std::ofstream mc_info;
  mc_info.open("../result/statistics/mc_reduction_gradient.txt", std::fstream::app);
  std::chrono::time_point<std::chrono::system_clock> start, end;

  start = std::chrono::system_clock::now();
  for (int i = 0; i < 4; ++i) {
    uchar3 mask0 = make_uchar3(kMCReductionMasks[i * 2 + 0][0],
                               kMCReductionMasks[i * 2 + 0][1],
                               kMCReductionMasks[i * 2 + 0][2]);
    uchar3 mask1 = make_uchar3(kMCReductionMasks[i * 2 + 1][0],
                               kMCReductionMasks[i * 2 + 1][1],
                               kMCReductionMasks[i * 2 + 1][2]);
    MarchingCubesLockFreeKernel<<<grid_size, block_size>>>(
            hash_table_.gpu_data(),
                    compact_hash_table_.gpu_data(),
                    blocks_.gpu_data(),
                    mesh_.gpu_data(),
                    mask0, mask1,
                    use_fine_gradient_);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> seconds = end - start;
  mc_info << seconds.count() << "\n";
#endif

  RecycleTrianglesKernel<<<grid_size, block_size>>>(
          compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel<<<grid_size, block_size>>>(
          compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

////////////////////////////////
/// Compress discrete vertices and triangles
__global__
void CollectVerticesAndTrianglesKernel(
        CompactHashTableGPU compact_hash_table,
        BlocksGPU           blocks,
        MeshGPU             mesh,
        CompactMeshGPU      compact_mesh) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  Cube &cube = blocks[entry.ptr].cubes[threadIdx.x];

  for (int i = 0; i < Cube::kMaxTrianglesPerCube; ++i) {
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
void Map::CompressMesh() {
  std::ofstream vertex_info;
  vertex_info.open("../result/statistics/vertex_info.txt", std::fstream::app);

  compact_mesh_.Reset();

  int occupied_block_count = compact_hash_table_.entry_count();
  if (occupied_block_count <= 0) return;

  {
    const uint threads_per_block = BLOCK_SIZE;
    const dim3 grid_size(occupied_block_count, 1);
    const dim3 block_size(threads_per_block, 1);

    CollectVerticesAndTrianglesKernel <<< grid_size, block_size >>> (
            compact_hash_table_.gpu_data(),
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
    vertex_info << vertex_ref_count_cpu;
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
  vertex_info << " " << compact_mesh_.vertex_count() << "\n";

  LOG(INFO) << "Triangles: " << compact_mesh_.triangle_count()
            << "/" << (mesh_.params().max_triangle_count - mesh_.triangle_heap_count());
}

void Map::SaveMesh(std::string path) {
  LOG(INFO) << "Copying data from GPU";

  CollectAllBlocks();
  CompressMesh();

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