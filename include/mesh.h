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
  uint*   heap;
  uint*   heap_counter;
  Vertex* vertices;
};

class Mesh {
private:
  HashTable<TriangleBlock> hash_table_;
  MeshData mesh_data_;

  HashParams   hash_params_;
  SensorParams sensor_params_;

public:
  Mesh(const HashParams &params);
  ~Mesh();

  void Reset();
  /// A naive version (per frame version)
  void CollectTargetBlocks(float4x4 c_T_w);

  void MarchingCubes(Map* map);

  HashTable<TriangleBlock> &hash_table() {
    return hash_table_;
  }
  HashTableGPU<TriangleBlock> &gpu_data() {
    return hash_table_.gpu_data();
  }
  SensorParams& sensor_params() {
    return sensor_params_;
  }
};


#endif //VH_MESHER_H
