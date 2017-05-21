//
// Created by wei on 17-4-5.
//
// Map: managing HashTable<VoxelBlock> and might be other structs later

#ifndef VH_MAP_H
#define VH_MAP_H

#include "hash_table.h"
#include "block.h"
#include "mesh.h"

#include "sensor.h"

class Map {
private:
  HashTable   hash_table_;
  VoxelBlocks blocks_;
  Mesh        mesh_;

  CompactHashTable compact_hash_table_;
  CompactMesh      compact_mesh_;

  uint integrated_frame_count_;

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
  void CollectAllBlocks();

public:
  /// Life cycle
  Map(const HashParams& hash_params);
  ~Map();

  /// Reset and recycle
  void Reset();
  void Recycle();

////////////////////
/// Fusion
////////////////////
private:
  void UpdateBlocks(Sensor& sensor);
  void AllocBlocks(Sensor& sensor);

public:
  void Integrate(Sensor &sensor, unsigned int* is_streamed_mask);

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

  /// Access for RayCaster
  HashTable& hash_table() {
    return hash_table_;
  }
  VoxelBlocks& blocks() {
    return blocks_;
  }
};


#endif //VH_MAP_H
