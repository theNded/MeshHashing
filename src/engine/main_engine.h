//
// Created by wei on 17-4-5.
//
// MainEngine: managing HashTable<Block> and might be other structs later

#ifndef VH_MAP_H
#define VH_MAP_H

#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/mesh.h"

#include "visualization/compact_mesh.h"
#include "visualization/bounding_box.h"
#include "sensor/rgbd_sensor.h"

class MainEngine {
private:
  // Core
  HashTable        hash_table_;
  BlockArray       blocks_;
  EntryArray       candidate_entries_;

  // Meshing
  Mesh             mesh_;

  // Visualization
  CompactMesh      compact_mesh_;

  // Geometry
  CoordinateConverter coordinate_converter_;


  uint             integrated_frame_count_;
  bool             use_fine_gradient_;

  BBox             bbox_;

////////////////////
/// Core
////////////////////

public:

  /// Life cycle
  MainEngine(
      const HashParams& hash_params,
      const MeshParams& mesh_params,
      const SDFParams&  sdf_params);
  ~MainEngine();

  /// Reset and recycle
  void Reset();

////////////////////
/// Fusion
////////////////////
public:
  void Mapping(Sensor &sensor);
  void Meshing();
  void Recycle();



  ////////////////////
/// Access functions
////////////////////
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
  EntryArray& candidate_entries() {
    return candidate_entries_;
  }
  BlockArray& blocks() {
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
  CoordinateConverter& converter() {
    return coordinate_converter_;
  }

public:
  void GetBoundingBoxes();
};


#endif //VH_MAP_H
