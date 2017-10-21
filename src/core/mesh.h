//
// Created by wei on 17-5-21.
//

#ifndef CORE_MESH_H
#define CORE_MESH_H

#include "core/common.h"
#include "core/params.h"
#include "core/vertex.h"
#include "core/triangle.h"

#include <helper_cuda.h>
#include <helper_math.h>

class Mesh {
public:
  __host__ Mesh();
  // __host__ ~Mesh();

  __host__ void Alloc(const MeshParams &mesh_params);
  __host__ void Free();
  __host__ void Resize(const MeshParams &mesh_params);
  __host__ void Reset();

  const MeshParams& params() {
    return mesh_params_;
  }

  __device__ __host__ Vertex& vertex(uint i) {
    return vertices[i];
  }
  __device__ __host__ Triangle& triangle(uint i) {
    return triangles[i];
  }

  __host__ uint vertex_heap_count();
  __host__ uint triangle_heap_count();

private:
  uint*     vertex_heap_;
  uint*     vertex_heap_counter_;
  Vertex*   vertices;

  uint*     triangle_heap_;
  uint*     triangle_heap_counter_;
  Triangle* triangles;

#ifdef __CUDACC__
public:
  __device__
  uint AllocVertex() {
    uint addr = atomicSub(&vertex_heap_counter_[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("v: %d -> %d\n", addr, vertex_heap_[addr]);
    }
    return vertex_heap_[addr];
  }
  __device__
  void FreeVertex(uint ptr) {
    uint addr = atomicAdd(&vertex_heap_counter_[0], 1);
    vertex_heap_[addr + 1] = ptr;
  }

  __device__
  uint AllocTriangle() {
    uint addr = atomicSub(&triangle_heap_counter_[0], 1);
    if (addr < MEMORY_LIMIT) {
      printf("t: %d -> %d\n", addr, vertex_heap_[addr]);
    }
    return triangle_heap_[addr];
  }
  __device__
  void FreeTriangle(uint ptr) {
    uint addr = atomicAdd(&triangle_heap_counter_[0], 1);
    triangle_heap_[addr + 1] = ptr;
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

  MeshParams mesh_params_;

};

#endif //VOXEL_HASHING_MESH_H
