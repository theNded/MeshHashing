//
// Created by wei on 17-10-21.
//

#ifndef VISUALIZATION_COMPACT_MESH_H
#define VISUALIZATION_COMPACT_MESH_H

#include "core/common.h"
#include "core/params.h"
#include "core/vertex.h"
#include "core/triangle.h"

class CompactMesh {
public:
  CompactMesh();
  //~CompactMesh();

  uint vertex_count();
  uint triangle_count();

  __device__ __host__
  int* vertex_remapper() {
    return vertex_remapper_;
  }
  __device__ __host__
  float3* vertices() {
    return vertices_;
  }
  __device__ __host__
  float3* normals() {
    return normals_;
  }
  __device__ __host__
  float3* colors() {
    return colors_;
  }
  __device__ __host__
  int3* triangles() {
    return triangles_;
  }
  __device__ __host__
  int* triangles_ref_count() {
    return triangles_ref_count_;
  }
  __device__ __host__
  int* vertices_ref_count() {
    return vertices_ref_count_;
  }
  __device__ __host__
  uint* triangle_counter() {
    return triangle_counter_;
  }
  __device__ __host__
  uint* vertex_counter() {
    return vertex_counter_;
  }
  void Alloc(const MeshParams& mesh_params);
  void Free();

  void Resize(const MeshParams &mesh_params);
  void Reset();

private:
  int*      vertex_remapper_;

  // They are decoupled so as to be separately assigned to the rendering pipeline
  float3*   vertices_;
  float3*   normals_;
  float3*   colors_;
  int*      vertices_ref_count_;
  uint*     vertex_counter_;

  int3*     triangles_;
  int*      triangles_ref_count_;
  uint*     triangle_counter_;
  MeshParams     mesh_params_;
};


#endif //MESH_HASHING_COMPACT_MESH_H
