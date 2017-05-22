//
// Created by wei on 17-5-21.
//

#ifndef VH_MESH_H
#define VH_MESH_H

#include "common.h"
#include <helper_cuda.h>
#include <helper_math.h>

struct __ALIGN__(4) Vertex {
  float3 pos;
  int    ref_count;

  __device__
  void Clear() {
    pos = make_float3(0.0);
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

static const int kMaxVertexCount = 10000000;

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
    return triangle_heap[addr];
  }
  __device__
  void FreeTriangle(uint ptr) {
    uint addr = atomicAdd(&triangle_heap_counter[0], 1);
    triangle_heap[addr + 1] = ptr;
  }
#endif // __CUDACC__
};

class Mesh {
private:
  MeshGPU gpu_data_;

  void Alloc(uint vertex_count, uint triangle_count);
  void Free();

public:
  Mesh();
  ~Mesh();

  void Resize(uint vertex_count, uint triangle_count);
  void Reset();

  MeshGPU& gpu_data() {
    return gpu_data_;
  }
};

struct CompactMeshGPU {
  // Remap from the separated vertices to the compacted vertices
  int*      vertex_index_remapper;

  float3*   vertices;
  int*      vertices_ref_count;
  uint*     vertex_counter;

  int3*     triangles;
  int*      triangles_ref_count;
  uint*     triangle_counter;
};

class CompactMesh {
private:
  CompactMeshGPU gpu_data_;

  void Alloc(uint vertex_count, uint triangle_count);
  void Free();

public:
  CompactMesh();
  ~CompactMesh();

  void Resize(uint vertex_count, uint triangle_count);
  void Reset();

  CompactMeshGPU& gpu_data() {
    return gpu_data_;
  }
};


#endif //VOXEL_HASHING_MESH_H
