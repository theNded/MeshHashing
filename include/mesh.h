//
// Created by wei on 17-4-18.
//

#ifndef VH_MESHER_H
#define VH_MESHER_H

#include "common.h"

#include "hash_table.h"
#include "map.h"
#include "params.h"

struct MeshData {
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
  uint AllocVertexHeap() {
    uint addr = atomicSub(&vertex_heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    return vertex_heap[addr];
  }
  __device__
  void FreeVertexHeap(uint ptr) {
    uint addr = atomicAdd(&vertex_heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    vertex_heap[addr + 1] = ptr;
  }

  __device__
  uint AllocTriangleHeap() {
    uint addr = atomicSub(&triangle_heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    return triangle_heap[addr];
  }
  __device__
  void FreeTriangleHeap(uint ptr) {
    uint addr = atomicAdd(&triangle_heap_counter[0], 1);
    //TODO MATTHIAS check some error handling?
    triangle_heap[addr + 1] = ptr;
  }
#endif // __CUDACC__
};

static const int kMaxVertexCount = 2500000;

class Mesh {
private:
  HashTable<VertexIndicesBlock> hash_table_;
  MeshData mesh_data_;

public:
  Mesh(const HashParams &params);
  ~Mesh();

  void Reset();
  void MarchingCubes(Map* map);
  void SaveMesh(std::string path);

  HashTable<VertexIndicesBlock> &hash_table() {
    return hash_table_;
  }
  HashTableGPU<VertexIndicesBlock> &gpu_data() {
    return hash_table_.gpu_data();
  }
};


#endif //VH_MESHER_H
