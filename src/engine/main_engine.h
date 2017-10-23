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

#include "engine/visualizing_engine.h"
#include "visualization/compact_mesh.h"
#include "visualization/bounding_box.h"
#include "sensor/rgbd_sensor.h"

class MainEngine {
private:
  VisualizingEngine vis_engine_;

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

  HashParams hash_params_;
  MeshParams mesh_params_;
  VolumeParams  volume_params_;
public:
  // configure main data
  MainEngine(
      const HashParams& hash_params,
      const MeshParams& mesh_params,
      const VolumeParams&  sdf_params);
  ~MainEngine();

  // configure engines
  void ConfigVisualizingEngineMesh(Light& light,
                                   bool free_viewpoint,
                                   bool render_global_mesh);
  void ConfigVisualizingEngineRaycaster(const RayCasterParams& params);

  /// Reset and recycle
  void Reset();

  void Mapping(Sensor &sensor);
  void Meshing();
  void Recycle();
  void Visualizing(float4x4 view);

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
