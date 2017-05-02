#include <glog/logging.h>
#include "mesh.h"

#include "mc_tables.h"

__global__
void ResetHeapKernel(MeshData mesh_data) {
  const uint max_vertice_count = 2500000;
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    mesh_data.vertex_heap_counter[0] = max_vertice_count - 1;	//points to the last element of
    mesh_data.triangle_heap_counter[0] = max_vertice_count - 1;
  }

  if (idx < max_vertice_count) {
    mesh_data.vertex_heap[idx] = max_vertice_count - idx - 1;
    mesh_data.triangle_heap[idx] = max_vertice_count - idx - 1;
    mesh_data.vertices[idx].Clear();
    mesh_data.triangles[idx].Clear();
  }
}

/// At current we suppose 1 - 1 correspondence for voxels
/// No deletion of vertices is considered
__device__
float3 VertexIntersection(const float3& p1, const float3 p2,
                          const float& v1,  const float& v2, const float& isolevel) {
  if (fabs(v1 - isolevel) < 0.00001) return p1;
  if (fabs(v2 - isolevel) < 0.00001) return p2;
  float mu = (isolevel - v1) / (v2 - v1);
  float3 p = make_float3(p1.x + mu * (p2.x - p1.x),
                         p1.y + mu * (p2.y - p1.y),
                         p1.z + mu * (p2.z - p1.z));
  return p;
}

__device__
inline Voxel GetVoxel(HashTableGPU<VoxelBlock>& scalar_table,
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
    v = scalar_table.values[curr_entry.ptr](VoxelLocalPosToIdx(voxel_local_pos));
  } else {
    HashEntry entry = scalar_table.GetEntry(curr_entry.pos + block_offset);
    if (entry.ptr == FREE_ENTRY) return v;
    int3 voxel_local_pos_ = make_int3(voxel_local_pos.x % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.y % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.z % BLOCK_SIDE_LENGTH);
    int i = VoxelPosToIdx(voxel_local_pos_);
    v = scalar_table.values[entry.ptr](i);
  }

  return v;
}

__device__
inline VertexIndices GetVertexIndice(HashTableGPU<VertexIndicesBlock>& mesh_table,
                                     MeshData &mesh_data,
                                     const HashEntry& curr_entry,
                                     uint3 voxel_local_pos,
                                     uint3 local_offset) {
  VertexIndices indices;

  voxel_local_pos = voxel_local_pos + local_offset;
  int3 block_offset = make_int3(voxel_local_pos.x / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.y / BLOCK_SIDE_LENGTH,
                                voxel_local_pos.z / BLOCK_SIDE_LENGTH);

  if (block_offset.x == 0
      && block_offset.y == 0
      && block_offset.z == 0) {
    indices = mesh_table.values[curr_entry.ptr](VoxelLocalPosToIdx(voxel_local_pos));
  } else {
    HashEntry entry = mesh_table.GetEntry(curr_entry.pos + block_offset);
    /// this should never happen
    /// if (entry.ptr == FREE_ENTRY) return indices;
    int3 voxel_local_pos_ = make_int3(voxel_local_pos.x % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.y % BLOCK_SIDE_LENGTH,
                                      voxel_local_pos.z % BLOCK_SIDE_LENGTH);
    int i = VoxelPosToIdx(voxel_local_pos_);
    indices = mesh_table.values[entry.ptr](i);
  }

  return indices;
}

// TODO(wei): add locks
__global__
void MarchingCubesKernel(HashTableGPU<VoxelBlock> scalar_table,
                         HashTableGPU<VertexIndicesBlock> mesh_table,
                         MeshData mesh_data) {
  const float isolevel = 0;
  const HashEntry &scalar_entry
          = scalar_table.compacted_hash_entries[blockIdx.x];
  const HashEntry &mesh_entry
          = mesh_table.compacted_hash_entries[blockIdx.x];

  int3  voxel_base_pos = BlockToVoxel(scalar_entry.pos);
  const uint local_idx = threadIdx.x;  //inside of an SDF block
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
  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(0, 1, 1));
  if (v.weight == 0) return;
  p[0] = world_pos + voxel_size * make_float3(0, 1, 1);
  d[0] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(1, 1, 1));
  if (v.weight == 0) return;
  p[1] = world_pos + voxel_size * make_float3(1, 1, 1);
  d[1] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(1, 1, 0));
  if (v.weight == 0) return;
  p[2] = world_pos + voxel_size * make_float3(1, 1, 0);
  d[2] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(0, 1, 0));
  if (v.weight == 0) return;
  p[3] = world_pos + voxel_size * make_float3(0, 1, 0);
  d[3] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(0, 0, 1));
  if (v.weight == 0) return;
  p[4] = world_pos + voxel_size * make_float3(0, 0, 1);
  d[4] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(1, 0, 1));
  if (v.weight == 0) return;
  p[5] = world_pos + voxel_size * make_float3(1, 0, 1);
  d[5] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(1, 0, 0));
  if (v.weight == 0) return;
  p[6] = world_pos + voxel_size * make_float3(1, 0, 0);
  d[6] = v.sdf;

  v = GetVoxel(scalar_table, scalar_entry, voxel_local_pos, make_uint3(0, 0, 0));
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

  if (kEdgeTable[cube_index] == 0 || kEdgeTable[cube_index] == 255)
    return;

  //////////
  /// 3. Determine vertices (ptr allocated via (shared) edges
  /// If the program reach here, than voxels holding edges must exist
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
  VertexIndices indices;
  int ptr;
  /// plane y = 1
  if (kEdgeTable[cube_index] & 1) {
    vertex_pos = VertexIntersection(p[0], p[1], d[0], d[1], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 1)).indices.x;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[0] = ptr;
  }
  if (kEdgeTable[cube_index] & 2) {
    vertex_pos = VertexIntersection(p[1], p[2], d[1], d[2], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 1, 0)).indices.z;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[1] = ptr;
  }
  if (kEdgeTable[cube_index] & 4) {
    vertex_pos = VertexIntersection(p[2], p[3], d[2], d[3], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 0)).indices.x;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[2] = ptr;
  }
  if (kEdgeTable[cube_index] & 8) {
    vertex_pos = VertexIntersection(p[3], p[0], d[3], d[0], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 1, 0)).indices.z;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[3] = ptr;
  }

  /// plane y = 0
  if (kEdgeTable[cube_index] & 16) {
    vertex_pos = VertexIntersection(p[4], p[5], d[4], d[5], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 1)).indices.x;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[4] = ptr;
  }
  if (kEdgeTable[cube_index] & 32) {
    vertex_pos = VertexIntersection(p[5], p[6], d[5], d[6], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 0)).indices.z;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[5] = ptr;
  }
  if (kEdgeTable[cube_index] & 64) {
    vertex_pos = VertexIntersection(p[6], p[7], d[6], d[7], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0)).indices.x;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[6] = ptr;
  }
  if (kEdgeTable[cube_index] & 128) {
    vertex_pos = VertexIntersection(p[7], p[4], d[7], d[4], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0)).indices.z;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[7] = ptr;
  }

  /// vertical
  if (kEdgeTable[cube_index] & 256) {
    vertex_pos = VertexIntersection(p[4], p[0], d[4], d[0], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 1)).indices.y;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[8] = ptr;
  }
  if (kEdgeTable[cube_index] & 512) {
    vertex_pos = VertexIntersection(p[5], p[1], d[5], d[1], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 1)).indices.y;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[9] = ptr;
  }
  if (kEdgeTable[cube_index] & 1024) {
    vertex_pos = VertexIntersection(p[6], p[2], d[6], d[2], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(1, 0, 0)).indices.y;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[10] = ptr;
  }
  if (kEdgeTable[cube_index] & 2048) {
    vertex_pos = VertexIntersection(p[7], p[3], d[7], d[3], isolevel);

    ptr = GetVertexIndice(mesh_table, mesh_data, mesh_entry, voxel_local_pos, make_uint3(0, 0, 0)).indices.y;
    ptr = (ptr == -1) ? mesh_data.AllocVertexHeap() : ptr;
    mesh_data.vertices[ptr].pos = vertex_pos;
    vertex_ptr[11] = ptr;
  }

  for (int t = 0; kTriangleTable[cube_index][t] != -1; t += 3) {
    Triangle triangle;
    triangle.indices.x = vertex_ptr[kTriangleTable[cube_index][t + 0]];
    triangle.indices.y = vertex_ptr[kTriangleTable[cube_index][t + 1]];
    triangle.indices.z = vertex_ptr[kTriangleTable[cube_index][t + 2]];
    int triangle_ptr = mesh_data.AllocTriangleHeap();
    mesh_data.triangles[triangle_ptr] = triangle;
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
  for (int i = 0; i < kMaxVertexCount; ++i) {
    ss.str("");
    ss <<  "v " << vertices[i].pos.x << " "
       << vertices[i].pos.y << " "
       << vertices[i].pos.z << "\n";
    //LOG(INFO) << ss.str();
    if (vertices[i].pos.x == 0.0f
            && vertices[i].pos.y == 0.0f
            && vertices[i].pos.z == 0.0f) break;
    out << ss.str();
  }

  for (int i = 0; i < kMaxVertexCount; ++i) {
    ss.str("");
    ss << "f " <<  triangles[i].indices.x + 1 << " "
                << triangles[i].indices.y + 1 << " "
                << triangles[i].indices.z + 1 << "\n";
    //LOG(INFO) << ss.str();
    if (triangles[i].indices.x == -1
            || triangles[i].indices.y == -1
            || triangles[i].indices.z == -1)
      break;
    out << ss.str();
  }

  delete[] vertices;
  delete[] triangles;
}