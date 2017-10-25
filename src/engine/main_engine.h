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
public:
  // configure main data
  MainEngine(
      const HashParams& hash_params,
      const MeshParams& mesh_params,
      const VolumeParams&  sdf_params);
  ~MainEngine();
  void Reset();

  // configure engines
  void ConfigVisualizingEngineMesh(Light& light,
                                   bool enable_navigation,
                                   bool enable_global_mesh,
                                   bool enable_bounding_box,
                                   bool enable_trajectory,
                                   bool enable_polygon_mode);
  void ConfigVisualizingEngineRaycaster(const RayCasterParams& params);
  void ConfigLoggingEngine(std::string path, bool enable_video, bool enable_ply);

  void Mapping(Sensor &sensor);
  void Meshing();
  void Recycle();
  void Visualize(float4x4 view);
  void Log();
  void FinalLog();

  const int& frame_count() {
    return integrated_frame_count_;
  }
  bool& use_fine_gradient() {
    return use_fine_gradient_;
  }

  GeometryHelper& geometry_helper() {
    return geometry_helper_;
  }

private:
  // Engines
  VisualizingEngine vis_engine_;
  LoggingEngine     log_engine_;

  // Core
  HashTable        hash_table_;
  BlockArray       blocks_;
  EntryArray       candidate_entries_;

  // Meshing
  Mesh             mesh_;

  // Geometry
  GeometryHelper geometry_helper_;

  int             integrated_frame_count_;
  bool            use_fine_gradient_;

  HashParams hash_params_;
  MeshParams mesh_params_;
  VolumeParams  volume_params_;
};


#endif //VH_MAP_H
