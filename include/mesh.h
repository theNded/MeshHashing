//
// Created by wei on 17-4-18.
//

#ifndef VH_MESHER_H
#define VH_MESHER_H

#include "common.h"

#include "hash_table.h"
#include "map.h"
#include "params.h"

struct SharedMash {
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

struct CompactMesh {
  // Remap from the separated vertices to the compacted vertices
  int*      vertex_index_remapper;

  Vertex*   vertices;
  int*     vertices_ref_count;
  uint*     vertex_counter;

  Triangle* triangles;
  int*     triangles_ref_count;
  uint*     triangle_counter;
};

static const int kMaxVertexCount = 10000000;

class Mesh {
private:
  SharedMash  mesh_data_;
  CompactMesh compact_mesh_;

public:
  Mesh();
  ~Mesh();

  void ResetSharedMesh();
  void ResetCompactMesh();

  /// For offline MC
  void CollectAllBlocks();
  void MarchingCubes(Map* map);
  void SaveMesh(Map* map, std::string path);

  void CompressMesh(Map* map);
};


#endif //VH_MESHER_H
