//
// Created by wei on 17-4-5.
//
// Map: managing HashTable<Block> and might be other structs later

#ifndef VH_MAP_H
#define VH_MAP_H

#include "hash_table.h"
#include "block.h"
#include "mesh.h"

#include "sensor.h"

class Map {
private:
  HashTable   hash_table_;
  Blocks      blocks_;
  Mesh        mesh_;

  CompactHashTable compact_hash_table_;
  CompactMesh      compact_mesh_;

  uint integrated_frame_count_;
  bool use_fine_gradient_;

  BBox       bbox_;

////////////////////
/// Core
////////////////////
private:
  /// Garbage collection
  void StarveOccupiedBlocks();
  void CollectGarbageBlocks();
  void RecycleGarbageBlocks();

  /// Compress entries
  void CollectInFrustumBlocks(Sensor& sensor);

public:
  void CollectAllBlocks();

  /// Life cycle
  Map(const HashParams& hash_params, const MeshParams& mesh_params);
  ~Map();

  /// Reset and recycle
  void Reset();
  void Recycle(int frame_count);

////////////////////
/// Fusion
////////////////////
private:
  void UpdateBlocks(Sensor& sensor);
  void AllocBlocks(Sensor& sensor);

public:
  void Integrate(Sensor &sensor);

////////////////////
/// Meshing
////////////////////
public:
  void MarchingCubes();
  void SaveMesh(std::string path);

  void CompressMesh();

  /// Only classes with Kernel function should call it
  /// The other part of the hash_table should be hidden
  const uint& frame_count() {
    return integrated_frame_count_;
  }
  bool& use_fine_gradient() {
    return use_fine_gradient_;
  }

  /// Access for RayCaster
  HashTable& hash_table() {
    return hash_table_;
  }
  CompactHashTable& compact_hash_table() {
    return compact_hash_table_;
  }
  Blocks& blocks() {
    return blocks_;
  }
  Mesh& mesh() {
    return mesh_;
  }
  CompactMesh& compact_mesh() {
    return compact_mesh_;
  }

  BBox& bbox() {
    return bbox_;
  }

public:
  void GetBoundingBoxes();
};


#endif //VH_MAP_H
