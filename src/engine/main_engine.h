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
#include "engine/logging_engine.h"
#include "visualization/compact_mesh.h"
#include "visualization/bounding_box.h"
#include "sensor/rgbd_sensor.h"

class MainEngine {
private:
  VisualizingEngine vis_engine_;
  LoggingEngine     log_engine_;

  // Core
  HashTable        hash_table_;
  BlockArray       blocks_;
  EntryArray       candidate_entries_;

  // Meshing
  Mesh             mesh_;

  // Visualization

  // Geometry
  CoordinateConverter coordinate_converter_;


  uint             integrated_frame_count_;
  bool             use_fine_gradient_;

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
                                   bool render_global_mesh,
                                   bool bounding_box,
                                   bool enable_trajectory);
  void ConfigVisualizingEngineRaycaster(const RayCasterParams& params);
  void ConfigLoggingEngine(std::string path, bool enable_video, bool enable_ply);

  /// Reset and recycle
  void Reset();

  void Mapping(Sensor &sensor);
  void Meshing();
  void Recycle();
  void Visualize(float4x4 view);
  void Log();
  void FinalLog();

  const uint& frame_count() {
    return integrated_frame_count_;
  }
  bool& use_fine_gradient() {
    return use_fine_gradient_;
  }

  CoordinateConverter& converter() {
    return coordinate_converter_;
  }
};


#endif //VH_MAP_H
