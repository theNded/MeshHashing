#include <glog/logging.h>
#include <unordered_map>
#include "mesh.h"

#include "mc_tables.h"

// TODO(wei): 1. a better allocation function (done)
//            2. remove isolated vertices     (done with bugs)
//            3. +/- detection for MarchingCubes
__global__
void ResetHeapKernel(MeshData mesh_data) {
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    mesh_data.vertex_heap_counter[0] = kMaxVertexCount - 1;
    mesh_data.triangle_heap_counter[0] = kMaxVertexCount - 1;
  }

  if (idx < kMaxVertexCount) {
    mesh_data.vertex_heap[idx] = kMaxVertexCount - idx - 1;
    mesh_data.triangle_heap[idx] = kMaxVertexCount - idx - 1;
    mesh_data.vertices[idx].Clear();
    mesh_data.triangles[idx].Clear();
  }
}

template <typename T>
__global__
void CollectAllBlocksKernel(HashTableGPU<T> hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count) {
    if (hash_table.hash_entries[idx].ptr != FREE_ENTRY) {
      addr_local = atomicAdd(&local_counter, 1);
    }
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(hash_table.compacted_hash_entry_counter, local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    hash_table.compacted_hash_entries[addr] = hash_table.hash_entries[idx];
  }
}

/// At current we suppose 1 - 1 correspondence for voxels
/// No deletion of vertices is considered
__device__
float3 VertexIntersection(const float3& p1, const float3 p2,
                          const float& v1,  const float& v2,
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
inline Voxel GetVoxel(HashTableGPU<VoxelBlock>& map_table,
                      const HashEntry& curr_entry,
                      uint3 voxel_local_pos,
                      const uint3 local_offset) {
  Voxel v; v.Clear();

  voxel_local_pos = voxel_local_pos + local_offset;
  int3 block_offset = make_int3(voxel_local_pos.x / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.y / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.z / BLOCK_SIDE_LENGTH);
  if (block_offset.x == 0
      && block_offset.y == 0
      && block_offset.z == 0) {
    v = map_table.values[curr_entry.ptr](VoxelLocalPosToIdx(voxel_local_pos));
  } else {
    HashEntry entry = map_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return v;
    int3 voxel_local_pos_ = make_int3(voxel_local_pos.x % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.y % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.z % BLOCK_SIDE_LENGTH);
    int i = VoxelPosToIdx(voxel_local_pos_);
    v = map_table.values[entry.ptr](i);
  }

  return v;
}

__device__
inline MeshCube& GetMeshCube(HashTableGPU<MeshCubeBlock>& mesh_table,
                                     MeshData &mesh_data,
                                     const HashEntry& curr_entry,
                                     uint3 voxel_local_pos,
                                     const uint3 local_offset) {

  voxel_local_pos = voxel_local_pos + local_offset;
  int3 block_offset = make_int3(voxel_local_pos.x / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.y / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.z / BLOCK_SIDE_LENGTH);

  if (block_offset.x == 0
      && block_offset.y == 0
      && block_offset.z == 0) {
    return mesh_table.values[curr_entry.ptr](VoxelLocalPosToIdx(voxel_local_pos));
  } else {
    HashEntry entry = mesh_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) {
      printf("Should never reach here! %d %d %d\n",
             voxel_local_pos.x,
             voxel_local_pos.y,
             voxel_local_pos.z);
    }
    /// this should never happen
    /// if (entry.ptr == FREE_ENTRY) return indices;
    int3 voxel_local_pos_ = make_int3(voxel_local_pos.x % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.y % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.z % BLOCK_SIDE_LENGTH);
    int i = VoxelPosToIdx(voxel_local_pos_);
    return mesh_table.values[entry.ptr](i);
  }
}

__device__
inline int AllocateVertex(MeshData& mesh_data,
                           int& vertex_ptr,
                           const float3& vertex_pos) {
  int ptr = vertex_ptr;
  if (ptr == -1) ptr = mesh_data.AllocVertexHeap();
  mesh_data.vertices[ptr].pos = vertex_pos;
  atomicAdd(&mesh_data.vertices[ptr].ref_count, 1);
  vertex_ptr = ptr;
  return ptr;
}

// TODO(wei): add locks
__global__
void MarchingCubesKernel(HashTableGPU<VoxelBlock> map_table,
                         HashTableGPU<MeshCubeBlock> mesh_table,
                         MeshData mesh_data) {
  const float isolevel = 0;

  const HashEntry &mesh_entry = mesh_table.compacted_hash_entries[blockIdx.x];
  const HashEntry &map_entry  = map_table.GetEntry(mesh_entry.pos);
  if (map_entry.ptr == FREE_ENTRY) {
    return;
  }

  int3  voxel_base_pos = BlockToVoxel(map_entry.pos);
  const uint local_idx = threadIdx.x;
  uint3 voxel_local_pos = IdxToVoxelLocalPos(local_idx);
  // TODO(wei): deal with border condition alone to save processing time?

  int3 voxel_pos = voxel_base_pos + make_int3(voxel_local_pos);
  float3 world_pos = VoxelToWorld(voxel_pos);

  //////////
  /// 1. Read the scalar values
  /// Refer to paulbourke.net/geometry/polygonise
  /// Our coordinate system:
  ///       ^
  ///      /
  ///    z
  ///   /
  /// o -- x -->
  /// |
  /// y
  /// |
  /// v
  // 0 -> 011
  // 1 -> 111
  // 2 -> 110
  // 3 -> 010
  // 4 -> 001
  // 5 -> 101
  // 6 -> 100
  // 7 -> 000
  Voxel  v;
  float  d[8];
  float3 p[8];

  float voxel_size = kSDFParams.voxel_size;
  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(0, 1, 1));
  if (v.weight == 0) return;
  p[0] = world_pos + voxel_size * make_float3(0, 1, 1);
  d[0] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(1, 1, 1));
  if (v.weight == 0) return;
  p[1] = world_pos + voxel_size * make_float3(1, 1, 1);
  d[1] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(1, 1, 0));
  if (v.weight == 0) return;
  p[2] = world_pos + voxel_size * make_float3(1, 1, 0);
  d[2] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(0, 1, 0));
  if (v.weight == 0) return;
  p[3] = world_pos + voxel_size * make_float3(0, 1, 0);
  d[3] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(0, 0, 1));
  if (v.weight == 0) return;
  p[4] = world_pos + voxel_size * make_float3(0, 0, 1);
  d[4] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(1, 0, 1));
  if (v.weight == 0) return;
  p[5] = world_pos + voxel_size * make_float3(1, 0, 1);
  d[5] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(1, 0, 0));
  if (v.weight == 0) return;
  p[6] = world_pos + voxel_size * make_float3(1, 0, 0);
  d[6] = v.sdf;

  v = GetVoxel(map_table, map_entry, voxel_local_pos, make_uint3(0, 0, 0));
  if (v.weight == 0) return;
  p[7] = world_pos + voxel_size * make_float3(0, 0, 0);
  d[7] = v.sdf;

  //////////
  /// 2. Determine cube type
  int cube_index = 0;
  if (d[0] < isolevel) cube_index |= 1;
  if (d[1] < isolevel) cube_index |= 2;
  if (d[2] < isolevel) cube_index |= 4;
  if (d[3] < isolevel) cube_index |= 8;
  if (d[4] < isolevel) cube_index |= 16;
  if (d[5] < isolevel) cube_index |= 32;
  if (d[6] < isolevel) cube_index |= 64;
  if (d[7] < isolevel) cube_index |= 128;

  const float kThreshold = 0.2f;
  if (fabs(d[0]) > kThreshold) return;
  if (fabs(d[1]) > kThreshold) return;
  if (fabs(d[2]) > kThreshold) return;
  if (fabs(d[3]) > kThreshold) return;
  if (fabs(d[4]) > kThreshold) return;
  if (fabs(d[5]) > kThreshold) return;
  if (fabs(d[6]) > kThreshold) return;
  if (fabs(d[7]) > kThreshold) return;
  for(uint k = 0; k < 8; k++) {
    for(uint l = 0; l < 8; l++) {
      if(d[k]*d[l] < 0.0f) {
        if(fabs(d[k]) + fabs(d[l]) > kThreshold) return;
      }
      else {
        if(fabs(d[k] - d[l]) > kThreshold) return;
      }
    }
  }

  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 3. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, the voxels holding edges must exist
  // 0 -> 011.x, (0, 1)
  // 1 -> 110.z, (1, 2)
  // 2 -> 010.x, (2, 3)
  // 3 -> 010.z, (3, 0)
  // 4 -> 001.x, (4, 5)
  // 5 -> 100.z, (5, 6)
  // 6 -> 000.x, (6, 7)
  // 7 -> 000.z, (7, 4)
  // 8 -> 001.y, (4, 0)
  // 9 -> 101.y, (5, 1)
  //10 -> 100.y, (6, 2)
  //11 -> 000.y, (7, 3)
  int vertex_ptr[12];
  float3 vertex_pos;

  /// plane y = 1
  if (kEdgeTable[cube_index] & 1) {
    vertex_pos = VertexIntersection(p[0], p[1], d[0], d[1], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 1));
    vertex_ptr[0] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 2) {
    vertex_pos = VertexIntersection(p[1], p[2], d[1], d[2], isolevel);

    MeshCube &cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 1, 0));
    vertex_ptr[1] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 4) {
    vertex_pos = VertexIntersection(p[2], p[3], d[2], d[3], isolevel);

    MeshCube &cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 0));
    vertex_ptr[2] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 8) {
    vertex_pos = VertexIntersection(p[3], p[0], d[3], d[0], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 0));
    vertex_ptr[3] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }

  /// plane y = 0
  if (kEdgeTable[cube_index] & 16) {
    vertex_pos = VertexIntersection(p[4], p[5], d[4], d[5], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 1));
    vertex_ptr[4] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 32) {
    vertex_pos = VertexIntersection(p[5], p[6], d[5], d[6], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 0));
    vertex_ptr[5] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 64) {
    vertex_pos = VertexIntersection(p[6], p[7], d[6], d[7], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[6] = AllocateVertex(mesh_data, cube.vertex_ptrs.x, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 128) {
    vertex_pos = VertexIntersection(p[7], p[4], d[7], d[4], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[7] = AllocateVertex(mesh_data, cube.vertex_ptrs.z, vertex_pos);
  }

  /// vertical
  if (kEdgeTable[cube_index] & 256) {
    vertex_pos = VertexIntersection(p[4], p[0], d[4], d[0], isolevel);

    MeshCube& cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 1));
    vertex_ptr[8] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 512) {
    vertex_pos = VertexIntersection(p[5], p[1], d[5], d[1], isolevel);

    MeshCube &cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 1));
    vertex_ptr[9] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 1024) {
    vertex_pos = VertexIntersection(p[6], p[2], d[6], d[2], isolevel);

    MeshCube &cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 0));
    vertex_ptr[10] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }
  if (kEdgeTable[cube_index] & 2048) {
    vertex_pos = VertexIntersection(p[7], p[3], d[7], d[3], isolevel);

    MeshCube &cube = GetMeshCube(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0));
    vertex_ptr[11] = AllocateVertex(mesh_data, cube.vertex_ptrs.y, vertex_pos);
  }

  MeshCube &cube = mesh_table.values[mesh_entry.ptr](local_idx);
  cube.cube_index = cube_index;
  int i = 0;
  for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3, ++i) {
    int triangle_ptr = cube.triangle_ptr[i];
    if (triangle_ptr == -1)
      triangle_ptr = mesh_data.AllocTriangleHeap();

    cube.triangle_ptr[i] = triangle_ptr;

    Triangle triangle; triangle.Clear();
    triangle.vertex_ptrs.x = vertex_ptr[kTriangleTable[cube_index][t + 0]];
    triangle.vertex_ptrs.y = vertex_ptr[kTriangleTable[cube_index][t + 1]];
    triangle.vertex_ptrs.z = vertex_ptr[kTriangleTable[cube_index][t + 2]];

    mesh_data.triangles[triangle_ptr] = triangle;
  }
}

__global__
void RecycleTrianglesKernel(HashTableGPU<MeshCubeBlock> mesh_table, MeshData mesh_data) {
  const HashEntry &mesh_entry = mesh_table.compacted_hash_entries[blockIdx.x];

  const uint local_idx = threadIdx.x;  //inside of an SDF block
  MeshCube &cube = mesh_table.values[mesh_entry.ptr](local_idx);

  int t = 0;
  for (; kTriangleTable[cube.cube_index][t] != -1; t += 3);

  for (; t < MeshCube::kTrianglePerCube; ++t) {
    int triangle_ptr = cube.triangle_ptr[t];
    if (triangle_ptr == -1) continue;

    atomicSub(&mesh_data.vertices[mesh_data.triangles[triangle_ptr].vertex_ptrs.x].ref_count, 1);
    atomicSub(&mesh_data.vertices[mesh_data.triangles[triangle_ptr].vertex_ptrs.y].ref_count, 1);
    atomicSub(&mesh_data.vertices[mesh_data.triangles[triangle_ptr].vertex_ptrs.z].ref_count, 1);

    cube.triangle_ptr[t] = -1;

    mesh_data.triangles[triangle_ptr].Clear();
    mesh_data.FreeTriangleHeap(triangle_ptr);
  }
}

__global__
void RecycleVerticesKernel(HashTableGPU<MeshCubeBlock> mesh_table, MeshData mesh_data) {
  const HashEntry &mesh_entry = mesh_table.compacted_hash_entries[blockIdx.x];
  const uint local_idx = threadIdx.x;
  MeshCube &cube = mesh_table.values[mesh_entry.ptr](local_idx);

  if (cube.vertex_ptrs.x != -1 && mesh_data.vertices[cube.vertex_ptrs.x].ref_count == 0) {
    mesh_data.vertices[cube.vertex_ptrs.x].Clear();
    mesh_data.FreeVertexHeap(cube.vertex_ptrs.x);
    cube.vertex_ptrs.x = -1;
  }
  if (cube.vertex_ptrs.y != -1 && mesh_data.vertices[cube.vertex_ptrs.y].ref_count == 0) {
    mesh_data.vertices[cube.vertex_ptrs.y].Clear();
    mesh_data.FreeVertexHeap(cube.vertex_ptrs.y);
    cube.vertex_ptrs.y = -1;
  }
  if (cube.vertex_ptrs.z != -1 && mesh_data.vertices[cube.vertex_ptrs.z].ref_count == 0) {
    mesh_data.vertices[cube.vertex_ptrs.z].Clear();
    mesh_data.FreeVertexHeap(cube.vertex_ptrs.z);
    cube.vertex_ptrs.z = -1;
  }
}

Mesh::Mesh(const HashParams &params) {
  checkCudaErrors(cudaMalloc(&mesh_data_.vertex_heap,
                             sizeof(uint) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&mesh_data_.vertex_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&mesh_data_.vertices,
                             sizeof(Vertex) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&mesh_data_.triangle_heap,
                             sizeof(uint) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&mesh_data_.triangle_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&mesh_data_.triangles,
                             sizeof(Triangle) * kMaxVertexCount));

  hash_table_.Resize(params);

  Reset();
}

Mesh::~Mesh() {
  checkCudaErrors(cudaFree(mesh_data_.vertex_heap));
  checkCudaErrors(cudaFree(mesh_data_.vertex_heap_counter));
  checkCudaErrors(cudaFree(mesh_data_.vertices));
  checkCudaErrors(cudaFree(mesh_data_.triangle_heap));
  checkCudaErrors(cudaFree(mesh_data_.triangle_heap_counter));
  checkCudaErrors(cudaFree(mesh_data_.triangles));
}

void Mesh::Reset() {
  const int threads_per_block = 64;
  const dim3 grid_size((kMaxVertexCount + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetHeapKernel<<<grid_size, block_size>>>(mesh_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  hash_table_.Reset();
}

/// Assume hash_table_ is compactified
void Mesh::CollectAllBlocks() {
  const uint threads_per_block = 256;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  LOG(INFO) << "entry count: " << entry_count;

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  checkCudaErrors(cudaMemset(gpu_data().compacted_hash_entry_counter,
                             0, sizeof(int)));
  CollectAllBlocksKernel<<<grid_size, block_size >>>(gpu_data());

  int res;
  checkCudaErrors(cudaMemcpy(&res, gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  LOG(INFO) << "Mesh block count in all: " << res;
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Mesh::MarchingCubes(Map *map) {
  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count,
                             gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  if (occupied_block_count <= 0)
    return;

  const uint threads_per_block = BLOCK_SIZE;
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);
  MarchingCubesKernel<<<grid_size, block_size>>>(map->gpu_data(), gpu_data(),
          mesh_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleTrianglesKernel<<<grid_size, block_size>>>(gpu_data(), mesh_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleVerticesKernel<<<grid_size, block_size>>>(gpu_data(), mesh_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Mesh::SaveMesh(std::string path) {
  /// get data from GPU
  LOG(INFO) << "Copying data from GPU";
  Vertex* vertices = new Vertex[kMaxVertexCount];
  Triangle *triangles = new Triangle[kMaxVertexCount];
  checkCudaErrors(cudaMemcpy(vertices, mesh_data_.vertices,
                             sizeof(Vertex) * kMaxVertexCount,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, mesh_data_.triangles,
                             sizeof(Triangle) * kMaxVertexCount,
                             cudaMemcpyDeviceToHost));

  LOG(INFO) << "Writing data";
  std::ofstream out(path);
  std::stringstream ss;

  std::unordered_map<int, int> indexer;
  indexer.reserve(kMaxVertexCount);

  int valid_idx = 0;
  for (int i = 0; i < kMaxVertexCount; ++i) {
    if (vertices[i].ref_count == 0) continue;

    ss.str("");
    ss <<  "v " << vertices[i].pos.x << " "
       << vertices[i].pos.y << " "
       << vertices[i].pos.z << "\n";
    indexer.emplace(i, valid_idx);

    //LOG(INFO) << ss.str();
    ++valid_idx;
    out << ss.str();
  }
  LOG(INFO) << "vertex count: " << valid_idx;

  int triangle_count = 0;
  for (int i = 0; i < kMaxVertexCount; ++i) {
    if (triangles[i].vertex_ptrs.x == -1
        || triangles[i].vertex_ptrs.y == -1
        || triangles[i].vertex_ptrs.z == -1)
      continue;

    ss.str("");
    int3 idx;
    try {
      idx.x = indexer.at(triangles[i].vertex_ptrs.x);
      idx.y = indexer.at(triangles[i].vertex_ptrs.y);
      idx.z = indexer.at(triangles[i].vertex_ptrs.z);
    } catch (std::exception& e) {
      LOG(INFO) << triangles[i].vertex_ptrs.x;
      LOG(INFO) << triangles[i].vertex_ptrs.y;
      LOG(INFO) << triangles[i].vertex_ptrs.z;
      continue;
    }

    if (length(vertices[triangles[i].vertex_ptrs.x].pos - vertices[triangles[i].vertex_ptrs.y].pos) > 0.1
            || length(vertices[triangles[i].vertex_ptrs.y].pos - vertices[triangles[i].vertex_ptrs.z].pos) > 0.1
            || length(vertices[triangles[i].vertex_ptrs.z].pos - vertices[triangles[i].vertex_ptrs.x].pos) > 0.1)
      continue;

    ss << "f " << idx.x + 1 << " " << idx.y + 1 << " " << idx.z + 1 << "\n";
    //LOG(INFO) << ss.str();

    ++triangle_count;
    out << ss.str();
  }
  LOG(INFO) << "triangle count: " << triangle_count;

  delete[] vertices;
  delete[] triangles;
}