#include <glog/logging.h>
#include <unordered_map>

#include "mc_tables.h"
#include "map.h"
#include "gradient.h"

#define REDUCTION

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
                      BlocksGPU&    blocks,
                      const HashEntry& curr_entry,
                      const uint3 voxel_local_pos) {
  Voxel v; v.Clear();

  /// Get voxel local index and block index
  int3 block_offset = make_int3(voxel_local_pos) / BLOCK_SIDE_LENGTH;

  /// Inside the block -- no need to look up in the table
  if (block_offset == make_int3(0)) {
    uint i = VoxelLocalPosToIdx(voxel_local_pos);
    v = blocks[curr_entry.ptr].voxels[i];
  } else { /// Outside the block -- look for it
    HashEntry entry = hash_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return v;
    uint i = VoxelLocalPosToIdx(voxel_local_pos % BLOCK_SIDE_LENGTH);

    v = blocks[entry.ptr].voxels[i];
  }

  return v;
}

__device__
inline Cube& GetCube(const HashTableGPU&  hash_table,
                     BlocksGPU blocks,
                     const HashEntry& curr_entry,
                     const uint3 voxel_local_pos) {

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

/// !!! CAUTION !!! memory leak might goes here with multiple threads!
__device__
inline int AllocateVertexLockFree(HashTableGPU &hash_table,
                          BlocksGPU &blocks,
                          MeshGPU& mesh,
                          Cube& cube,
                          uint& vertex_idx,
                          const float3& vertex_pos,
                          bool use_fine_gradient) {
  int ptr = cube.vertex_ptrs[vertex_idx];
  /// Fallible with multiple threads
  if (ptr == FREE_PTR) {
    ptr = mesh.AllocVertex();
  }

  mesh.vertices[ptr].pos = vertex_pos;
  if (use_fine_gradient) {
    mesh.vertices[ptr].normal
            = GradientAtPoint(hash_table, blocks, vertex_pos);
  }
  cube.vertex_ptrs[vertex_idx] = ptr;

  return ptr;
}

__device__
inline int AllocateVertexWithMutex(HashTableGPU &hash_table,
                                   BlocksGPU &blocks,
                          MeshGPU& mesh,
                          Cube& cube,
                          uint& vertex_idx,
                          const float3& vertex_pos,
                          bool use_fine_gradient) {
  int ptr = cube.vertex_ptrs[vertex_idx];
  /// Fallible with multiple threads
  if (ptr == FREE_PTR) {
    int lock = atomicExch(&cube.vertex_mutexes[vertex_idx], LOCK_ENTRY);
    if (lock != LOCK_ENTRY) {
      ptr = mesh.AllocVertex();
    }
  }

  if (ptr >= 0) {
    mesh.vertices[ptr].pos = vertex_pos;
    if (use_fine_gradient) {
      mesh.vertices[ptr].normal = GradientAtPoint(hash_table, blocks,
                                                  vertex_pos);
    }
    cube.vertex_ptrs[vertex_idx] = ptr;
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

__device__
void MarchingCubesLockFree(HashTableGPU&        hash_table,
                              CompactHashTableGPU& compact_hash_table,
                              BlocksGPU&      blocks,
                              MeshGPU&             mesh_data,
                              /// Cube info
                              Cube&            this_cube,
                              const HashEntry&     map_entry,
                              int3&                voxel_base_pos,
                              uint3&               voxel_local_pos,

                              bool                 use_fine_gradient) {
  const float isolevel = 0;

  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = VoxelToWorld(voxel_pos);

  //////////
  /// 1. Read the scalar values, see mc_tables.h
  Voxel v;
  float voxel_size = kSDFParams.voxel_size;
  float d[8];
  float3 p[8];

#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    uint3 offset = make_uint3(kVertexCubeTable[i][0],
                              kVertexCubeTable[i][1],
                              kVertexCubeTable[i][2]);
    v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos + offset);
    if (v.weight == 0) return;
    p[i] = world_pos + voxel_size * make_float3(offset);
    d[i] = v.sdf;
  }

  //////////
  /// 2. Determine cube type
  const float kThreshold = 0.2f;
  int cube_index = 0;
#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    int mask = (1 << i);
    if (fabs(d[i]) > kThreshold) return;
    if (d[i] < isolevel) cube_index |= mask;
  }

  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 3. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  int vertex_ptr[12];
  float3 vertex_pos;

  /// This operation is in 2-pass
  /// pass1: Allocate
#pragma unroll 1
  for (int i = 0; i < 12; ++i) {
    int mask = (1 << i);
    if (kEdgeTable[cube_index] & mask) {
      int2 v_idx = make_int2(kEdgeVertexTable[i][0], kEdgeVertexTable[i][1]);
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);

      vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                      d[v_idx.x], d[v_idx.y], isolevel);
      Cube &cube = GetCube(hash_table, blocks, map_entry,
                           voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z));
      vertex_ptr[i] = AllocateVertexLockFree(hash_table, blocks, mesh_data,
                                             cube, c_idx.w, vertex_pos,
                                             use_fine_gradient);
    }
  }

  int i = 0;
  for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3, ++i) {
    int triangle_ptrs = this_cube.triangle_ptrs[i];

    /// If the cube type is not changed, do not modify triangles,
    /// as they are what they are
    if (kTriangleTable[cube_index][t]
        != kTriangleTable[this_cube.cube_index][t]) {
      if (triangle_ptrs == FREE_PTR) {
        triangle_ptrs = mesh_data.AllocTriangle();
      } else { // recycle the rubbish (TODO: more sophisticated operations)
        int3 vertex_ptrs = mesh_data.triangles[triangle_ptrs].vertex_ptrs;
        atomicSub(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);
      }
    }

    this_cube.triangle_ptrs[i] = triangle_ptrs;

    Triangle triangle;
    triangle.Clear();
    triangle.vertex_ptrs.x = vertex_ptr[kTriangleTable[cube_index][t + 0]];
    triangle.vertex_ptrs.y = vertex_ptr[kTriangleTable[cube_index][t + 1]];
    triangle.vertex_ptrs.z = vertex_ptr[kTriangleTable[cube_index][t + 2]];

    if (! use_fine_gradient) {
      float3 p0 = mesh_data.vertices[triangle.vertex_ptrs.x].pos;
      float3 p1 = mesh_data.vertices[triangle.vertex_ptrs.y].pos;
      float3 p2 = mesh_data.vertices[triangle.vertex_ptrs.z].pos;
      float3 n = normalize(cross(p2 - p0, p1 - p0));
      mesh_data.vertices[triangle.vertex_ptrs.x].normal = n;
      mesh_data.vertices[triangle.vertex_ptrs.y].normal = n;
      mesh_data.vertices[triangle.vertex_ptrs.z].normal = n;
    }

    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.y].ref_count, 1);
    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.x].ref_count, 1);
    atomicAdd(&mesh_data.vertices[triangle.vertex_ptrs.z].ref_count, 1);

    mesh_data.triangles[triangle_ptrs] = triangle;
  }
  this_cube.cube_index = cube_index;
}

__device__
void MarchingCubesPass1(HashTableGPU&        hash_table,
                          CompactHashTableGPU& compact_hash_table,
                          BlocksGPU&      blocks,
                          MeshGPU&             mesh_data,
                          /// Cube info
                          Cube&            this_cube,
                          const HashEntry&     map_entry,
                          int3&                voxel_base_pos,
                          uint3&               voxel_local_pos,

                          bool                 use_fine_gradient) {
  const float isolevel = 0;

  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = VoxelToWorld(voxel_pos);

  //////////
  /// 1. Read the scalar values, see mc_tables.h
  Voxel v;
  const float voxel_size = kSDFParams.voxel_size;
  const float kThreshold = 0.2f;

  float d[8];
  float3 p[8];
  short cube_index = 0;
  this_cube.prev_index = this_cube.cube_index;
  this_cube.cube_index = 0;
#pragma unroll 1
  for (int i = 0; i < 8; ++i) {
    uint3 offset = make_uint3(kVertexCubeTable[i][0],
                              kVertexCubeTable[i][1],
                              kVertexCubeTable[i][2]);
    v = GetVoxel(hash_table, blocks, map_entry, voxel_local_pos + offset);
    if (v.weight == 0) return;

    p[i] = world_pos + voxel_size * make_float3(offset);
    d[i] = v.sdf;
    if (fabs(d[i]) > kThreshold) return;

    int mask = (1 << i);
    if (d[i] < isolevel) cube_index |= mask;
  }
  this_cube.cube_index = cube_index;
  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  /// This operation is in 2-pass
  /// pass1: Allocate
#pragma unroll 1
  for (int i = 0; i < 12; ++i) {
    int mask = (1 << i);
    if (kEdgeTable[cube_index] & mask) {
      int2 v_idx = make_int2(kEdgeVertexTable[i][0], kEdgeVertexTable[i][1]);
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);

      float3 vertex_pos = VertexIntersection(p[v_idx.x], p[v_idx.y],
                                             d[v_idx.x], d[v_idx.y], isolevel);
      Cube &cube = GetCube(hash_table, blocks, map_entry,
                           voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z));
      AllocateVertexWithMutex(hash_table, blocks, mesh_data,
                              cube, c_idx.w, vertex_pos,
                              use_fine_gradient);
    }
  }
}


__device__
void MarchingCubesPass2(HashTableGPU&        hash_table,
                          CompactHashTableGPU& compact_hash_table,
                          BlocksGPU&      blocks,
                          MeshGPU&             mesh_data,
                          /// Cube info
                          Cube&            this_cube,
                          const HashEntry&     map_entry,
                          int3&                voxel_base_pos,
                          uint3&               voxel_local_pos,

                          bool                 use_fine_gradient) {
  const float isolevel = 0;

  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = VoxelToWorld(voxel_pos);

  if (kEdgeTable[this_cube.cube_index] == 0
      || kEdgeTable[this_cube.cube_index] == 255) {
    return;
  }

  //////////
  /// 2. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  int vertex_ptr[12];

  /// This operation is in 2-pass
  /// pass1: Allocate
#pragma unroll 1
  for (int i = 0; i < 12; ++i) {
    int mask = (1 << i);
    if (kEdgeTable[this_cube.cube_index] & mask) {
      uint4 c_idx = make_uint4(kEdgeCubeTable[i][0], kEdgeCubeTable[i][1],
                               kEdgeCubeTable[i][2], kEdgeCubeTable[i][3]);
      Cube &cube = GetCube(hash_table, blocks, map_entry,
                           voxel_local_pos + make_uint3(c_idx.x, c_idx.y, c_idx.z));
      vertex_ptr[i] = GetVertexWithMutex(cube, c_idx.w);
    }
  }

  int i = 0;
  if (this_cube.cube_index == this_cube.prev_index) return;
  for (int t = 0; kTriangleTable[this_cube.cube_index][t] != -1; t += 3, ++i) {
    int triangle_ptrs = this_cube.triangle_ptrs[i];

    /// If the cube type is not changed, do not modify triangles,
    /// as they are what they are
//    if (kTriangleTable[cube_index][t]
//        != kTriangleTable[this_cube.cube_index][t]) {
      if (triangle_ptrs == FREE_PTR) {
        triangle_ptrs = mesh_data.AllocTriangle();
      } else { // recycle the rubbish (TODO: more sophisticated operations)
        int3 vertex_ptrs = mesh_data.triangles[triangle_ptrs].vertex_ptrs;
        atomicSub(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
        atomicSub(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);
      }


    this_cube.triangle_ptrs[i] = triangle_ptrs;

    int3 vertex_ptrs;
    vertex_ptrs.x = vertex_ptr[kTriangleTable[this_cube.cube_index][t + 0]];
    vertex_ptrs.y = vertex_ptr[kTriangleTable[this_cube.cube_index][t + 1]];
    vertex_ptrs.z = vertex_ptr[kTriangleTable[this_cube.cube_index][t + 2]];

    if (! use_fine_gradient) {
      float3 p0 = mesh_data.vertices[vertex_ptrs.x].pos;
      float3 p1 = mesh_data.vertices[vertex_ptrs.y].pos;
      float3 p2 = mesh_data.vertices[vertex_ptrs.z].pos;
      float3 n = normalize(cross(p2 - p0, p1 - p0));
      mesh_data.vertices[vertex_ptrs.x].normal = n;
      mesh_data.vertices[vertex_ptrs.y].normal = n;
      mesh_data.vertices[vertex_ptrs.z].normal = n;
    }

    atomicAdd(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
    atomicAdd(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
    atomicAdd(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);

    mesh_data.triangles[triangle_ptrs].vertex_ptrs = vertex_ptrs;
  }
}


__global__
void MarchingCubesLockFreeKernel(HashTableGPU        hash_table,
                         CompactHashTableGPU compact_hash_table,
                         BlocksGPU      blocks,
                         MeshGPU             mesh_data,
                         uchar3 mask1,
                         uchar3 mask2,
                         bool                use_fine_gradient) {

  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;
  int3  voxel_base_pos  = BlockToVoxel(map_entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  if (check_mask(voxel_local_pos, mask1)
      || check_mask(voxel_local_pos, mask2)) {
    Cube &this_cube = blocks[map_entry.ptr].cubes[local_idx];
    this_cube.cube_index = 0;
    MarchingCubesLockFree(hash_table,
                          compact_hash_table,
                          blocks,
                          mesh_data,
                          this_cube,
                          map_entry,
                          voxel_base_pos,
                          voxel_local_pos,
                          use_fine_gradient);
  }
}


__global__
void MarchingCubesPass1Kernel(HashTableGPU        hash_table,
                         CompactHashTableGPU compact_hash_table,
                         BlocksGPU      blocks,
                         MeshGPU             mesh_data,
                         bool                use_fine_gradient) {

  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;
  int3  voxel_base_pos  = BlockToVoxel(map_entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  Cube &this_cube = blocks[map_entry.ptr].cubes[local_idx];
  MarchingCubesPass1(hash_table,
                       compact_hash_table,
                       blocks,
                       mesh_data,
                       this_cube,
                       map_entry,
                       voxel_base_pos,
                       voxel_local_pos,
                       use_fine_gradient);
}

__global__
void MarchingCubesPass2Kernel(HashTableGPU        hash_table,
                         CompactHashTableGPU compact_hash_table,
                         BlocksGPU      blocks,
                         MeshGPU             mesh_data,
                         bool                use_fine_gradient) {

  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;
  int3  voxel_base_pos  = BlockToVoxel(map_entry.pos);
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  Cube &this_cube = blocks[map_entry.ptr].cubes[local_idx];
  MarchingCubesPass2(hash_table,
                           compact_hash_table,
                           blocks,
                           mesh_data,
                           this_cube,
                           map_entry,
                           voxel_base_pos,
                           voxel_local_pos,
                           use_fine_gradient);
}

/// Garbage collection (ref count)
__global__
void RecycleTrianglesKernel(CompactHashTableGPU compact_hash_table,
                            BlocksGPU      blocks,
                            MeshGPU             mesh_data) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;  //inside an SDF block
  Cube &cube = blocks[entry.ptr].cubes[local_idx];

  int i = 0;
  for (int t = 0; kTriangleTable[cube.cube_index][t] != -1; t += 3, ++i);

  for (; i < Cube::kMaxTrianglesPerCube; ++i) {
    int triangle_ptrs = cube.triangle_ptrs[i];
    if (triangle_ptrs == FREE_PTR) continue;

    int3 vertex_ptrs = mesh_data.triangles[triangle_ptrs].vertex_ptrs;
    atomicSub(&mesh_data.vertices[vertex_ptrs.x].ref_count, 1);
    atomicSub(&mesh_data.vertices[vertex_ptrs.y].ref_count, 1);
    atomicSub(&mesh_data.vertices[vertex_ptrs.z].ref_count, 1);

    cube.triangle_ptrs[i] = FREE_PTR;
    mesh_data.triangles[triangle_ptrs].Clear();
    mesh_data.FreeTriangle(triangle_ptrs);
  }
}

__global__
void RecycleVerticesKernel(CompactHashTableGPU compact_hash_table,
                           BlocksGPU      blocks,
                           MeshGPU             mesh_data) {
  const HashEntry &entry = compact_hash_table.compacted_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  Cube &cube = blocks[entry.ptr].cubes[local_idx];

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (cube.vertex_ptrs[i] != FREE_PTR &&
        mesh_data.vertices[cube.vertex_ptrs[i]].ref_count <= 0) {
      mesh_data.vertices[cube.vertex_ptrs[i]].Clear();
      mesh_data.FreeVertex(cube.vertex_ptrs[i]);
      cube.vertex_ptrs[i] = FREE_PTR;
    }
  }
}

/// Compress discrete vertices and triangles
__global__
void CollectVerticesAndTrianglesKernel(CompactHashTableGPU compact_hash_table,
                                       BlocksGPU      blocks,
                                       MeshGPU             mesh,
                                       CompactMeshGPU      compact_mesh) {
  const HashEntry &map_entry = compact_hash_table.compacted_entries[blockIdx.x];
  Cube &cube = blocks[map_entry.ptr].cubes[threadIdx.x];

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
void AssignVertexRemapperKernel(MeshGPU        mesh,
                                CompactMeshGPU compact_mesh,
                                uint max_vertex_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_vertex_count && compact_mesh.vertices_ref_count[idx] > 0) {
    int addr = atomicAdd(compact_mesh.vertex_counter, 1);
    compact_mesh.vertex_index_remapper[idx] = addr;
    compact_mesh.vertices[addr] = mesh.vertices[idx].pos;
    compact_mesh.normals[addr]  = mesh.vertices[idx].normal;
  }
}

__global__
void AssignTrianglesKernel(MeshGPU        mesh,
                           CompactMeshGPU compact_mesh,
                           uint max_triangle_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < max_triangle_count && compact_mesh.triangles_ref_count[idx] > 0) {
    int addr = atomicAdd(compact_mesh.triangle_counter, 1);
    compact_mesh.triangles[addr].x
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.x];
    compact_mesh.triangles[addr].y
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.y];
    compact_mesh.triangles[addr].z
            = compact_mesh.vertex_index_remapper[
            mesh.triangles[idx].vertex_ptrs.z];
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

  /// Use divide and conquer to avoid read-write conflict
#ifndef REDUCTION
  MarchingCubesPass1Kernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data(),
          use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesPass2Kernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
#else
  MarchingCubesLockFreeKernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  make_uchar3(0, 0, 0), make_uchar3(1, 1, 1),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesLockFreeKernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  make_uchar3(0, 0, 1), make_uchar3(1, 1, 0),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesLockFreeKernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  make_uchar3(0, 1, 0), make_uchar3(1, 0, 1),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  MarchingCubesLockFreeKernel<<<grid_size, block_size>>>(
          hash_table_.gpu_data(),
                  compact_hash_table_.gpu_data(),
                  blocks_.gpu_data(),
                  mesh_.gpu_data(),
                  make_uchar3(1, 0, 0), make_uchar3(0, 1, 1),
                  use_fine_gradient_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
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

/// Assume this operation is following
/// CollectInFrustumBlocks or
/// CollectAllBlocks
void Map::CompressMesh() {
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

    AssignVertexRemapperKernel <<< grid_size, block_size >>> (
            mesh_.gpu_data(),
            compact_mesh_.gpu_data(),
            mesh_.params().max_vertex_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const uint threads_per_block = 256;
    const dim3 grid_size((mesh_.params().max_triangle_count
                          + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    AssignTrianglesKernel <<< grid_size, block_size >>> (
            mesh_.gpu_data(),
            compact_mesh_.gpu_data(),
            mesh_.params().max_triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  LOG(INFO) << "Vertices: " << compact_mesh_.vertex_count()
            << "/" << (mesh_.params().max_vertex_count - mesh_.vertex_heap_count());
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