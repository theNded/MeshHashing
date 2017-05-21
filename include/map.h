//
// Created by wei on 17-4-5.
//
// Map: managing HashTable<VoxelBlock> and might be other structs later

#ifndef VH_MAP_H
#define VH_MAP_H

#include "hash_table.h"
#include "block.h"
#include "sensor.h"

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


class Map {
private:
  HashTable   hash_table_;
  VoxelBlocks blocks_;

  uint integrated_frame_count_;
  SharedMash  mesh_data_;
  CompactMesh compact_mesh_;

  CompactHashTable compact_hash_table_;

  /// Garbage collection
  void StarveOccupiedVoxels();
  void CollectInvalidBlockInfo();
  void RecycleInvalidBlock();

  /// Fusion part
  void UpdateBlocks(Sensor* sensor);
  void AllocBlocks(Sensor* sensor);

public:
  Map(const HashParams& hash_params);
  ~Map();

  void Integrate(Sensor *sensor, unsigned int* is_streamed_mask);

  void Reset();
  void Recycle();

  /// Core: Compress hash entries for parallel computation
  void CollectTargetBlocks(Sensor *sensor);
  void CollectAllBlocks();

  void ResetBlocks(int value_capacity);

  /// Mesh
  void ResetSharedMesh();
  void ResetCompactMesh();

  /// For offline MC
  void MarchingCubes();
  void SaveMesh(std::string path);

  void CompressMesh();

  /// Only classes with Kernel function should call it
  /// The other part of the hash_table should be hidden
  HashTable &hash_table() {
    return hash_table_;
  }
  HashTableGPU &gpu_data() {
    return hash_table_.gpu_data();
  }
  uint& frame_count() {
    return integrated_frame_count_;
  }
  VoxelBlocks& blocks() {
    return blocks_;
  }
  void Debug() {
    hash_table_.Debug();
  }
};


#endif //VH_MAP_H
