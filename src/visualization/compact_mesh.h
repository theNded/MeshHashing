//
// Created by wei on 17-10-21.
//

#ifndef VISUALIZATION_COMPACT_MESH_H
#define VISUALIZATION_COMPACT_MESH_H

#include "core/common.h"
#include "core/params.h"
#include "core/vertex.h"
#include "core/triangle.h"


struct CompactMeshGPU {
  // Remap from the separated vertices to the compacted vertices
  int*      vertex_remapper;

  // They are decoupled so as to be separately assigned to the rendering pipeline
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
  CompactMeshGPU gpu_memory_;
  MeshParams     mesh_params_;

  void Alloc(const MeshParams& mesh_params);
  void Free();

public:
  CompactMesh();
  ~CompactMesh();

  uint vertex_count();
  uint triangle_count();

  float3* vertices() {
    return gpu_memory_.vertices;
  }
  float3* normals() {
    return gpu_memory_.normals;
  }
  float3* colors() {
    return gpu_memory_.colors;
  }
  int3* triangles() {
    return gpu_memory_.triangles;
  }

  void Resize(const MeshParams &mesh_params);
  void Reset();

  CompactMeshGPU& gpu_memory() {
    return gpu_memory_;
  }
};


#endif //MESH_HASHING_COMPACT_MESH_H
