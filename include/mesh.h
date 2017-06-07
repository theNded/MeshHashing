//
// Created by wei on 17-5-21.
//

#ifndef VH_MESH_H
#define VH_MESH_H

#include "common.h"
#include <helper_cuda.h>
#include <helper_math.h>

#include "params.h"

struct __ALIGN__(4) Vertex {
  float3 pos;
  float3 normal;
  float3 color;
  int    ref_count;

  __device__
  void Clear() {
    pos = make_float3(0.0);
    normal = make_float3(0.0);
    color = make_float3(0);
    ref_count = 0;
  }
};

struct __ALIGN__(4) Triangle {
  int3 vertex_ptrs;

  __device__
  void Clear() {
    vertex_ptrs = make_int3(-1, -1, -1);
  }
};

struct MeshGPU {
  // Dynamic memory management for vertices
  // We need compact operation,
  // as MC during updating might release some triangles
  uint*   vertex_heap;
  uint*   vertex_heap_counter;
  Vertex* vertices;

  uint*     triangle_heap;
  uint*     triangle_heap_counter;
  Triangle* triangles;

#ifdef __CUDACC__
  __device__
  uint AllocVertex() {
    uint addr = atomicSub(&vertex_heap_counter[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("v: %d -> %d\n", addr, vertex_heap[addr]);
    }
    return vertex_heap[addr];
  }
  __device__
  void FreeVertex(uint ptr) {
    uint addr = atomicAdd(&vertex_heap_counter[0], 1);
    vertex_heap[addr + 1] = ptr;
  }

  __device__
  uint AllocTriangle() {
    uint addr = atomicSub(&triangle_heap_counter[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("t: %d -> %d\n", addr, vertex_heap[addr]);
    }
    return triangle_heap[addr];
  }
  __device__
  void FreeTriangle(uint ptr) {
    uint addr = atomicAdd(&triangle_heap_counter[0], 1);
    triangle_heap[addr + 1] = ptr;
  }

  /// Release is NOT always a FREE operation
  __device__
  void ReleaseTriangle(Triangle& triangle) {
    int3 vertex_ptrs = triangle.vertex_ptrs;
    atomicSub(&vertices[vertex_ptrs.x].ref_count, 1);
    atomicSub(&vertices[vertex_ptrs.y].ref_count, 1);
    atomicSub(&vertices[vertex_ptrs.z].ref_count, 1);
  }

  __device__
  void AssignTriangle(Triangle& triangle, int3 vertex_ptrs) {
    triangle.vertex_ptrs = vertex_ptrs;
    atomicAdd(&vertices[vertex_ptrs.y].ref_count, 1);
    atomicAdd(&vertices[vertex_ptrs.x].ref_count, 1);
    atomicAdd(&vertices[vertex_ptrs.z].ref_count, 1);
  }

  __device__
  void ComputeTriangleNormal(Triangle& triangle) {
    int3 vertex_ptrs = triangle.vertex_ptrs;
    float3 p0 = vertices[vertex_ptrs.x].pos;
    float3 p1 = vertices[vertex_ptrs.y].pos;
    float3 p2 = vertices[vertex_ptrs.z].pos;
    float3 n = normalize(cross(p2 - p0, p1 - p0));
    vertices[vertex_ptrs.x].normal = n;
    vertices[vertex_ptrs.y].normal = n;
    vertices[vertex_ptrs.z].normal = n;
  }
#endif // __CUDACC__
};

class Mesh {
private:
  MeshGPU gpu_data_;
  MeshParams mesh_params_;

  void Alloc(const MeshParams &mesh_params);
  void Free();

public:
  Mesh();
  ~Mesh();

  void Resize(const MeshParams &mesh_params);
  void Reset();

  MeshGPU& gpu_data() {
    return gpu_data_;
  }
  const MeshParams& params() {
    return mesh_params_;
  }

  uint vertex_heap_count();
  uint triangle_heap_count();

};

struct CompactMeshGPU {
  // Remap from the separated vertices to the compacted vertices
  int*      vertex_remapper;

  float3*   vertices;
  float3*   normals;
  float3*   colors;
  int*      vertices_ref_count;
  uint*     vertex_counter;

  int3*     triangles;
  int*      triangles_ref_count;
  uint*     triangle_counter;
};

class CompactMesh {
private:
  CompactMeshGPU gpu_data_;
  MeshParams     mesh_params_;

  void Alloc(const MeshParams& mesh_params);
  void Free();

public:
  CompactMesh();
  ~CompactMesh();

  uint vertex_count();
  uint triangle_count();

  float3* vertices() {
    return gpu_data_.vertices;
  }
  float3* normals() {
    return gpu_data_.normals;
  }
  float3* colors() {
    return gpu_data_.colors;
  }
  int3* triangles() {
    return gpu_data_.triangles;
  }

  void Resize(const MeshParams &mesh_params);
  void Reset();

  CompactMeshGPU& gpu_data() {
    return gpu_data_;
  }
};

//////////////////////
/// Bonuding Box, used for debugging
/////////////////////
struct BBoxGPU {
  float3* vertices;
  uint*   vertex_counter;
};

class BBox {
private:
  BBoxGPU gpu_data_;
  int max_vertex_count_;

  void Alloc(int max_vertex_count);
  void Free();

public:
  BBox();
  ~BBox();

  uint vertex_count();

  float3* vertices() {
    return gpu_data_.vertices;
  }

  void Resize(int amx_vertex_count);
  void Reset();

  BBoxGPU& gpu_data() {
    return gpu_data_;
  }

};
#endif //VOXEL_HASHING_MESH_H
