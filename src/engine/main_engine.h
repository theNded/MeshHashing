//
// Created by wei on 17-4-5.
//
// MainEngine: managing HashTable<Block> and might be other structs later

#ifndef ENGINE_MAIN_ENGINE_H
#define ENGINE_MAIN_ENGINE_H

#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/mesh.h"

#include "engine/visualizing_engine.h"
#include "engine/logging_engine.h"
#include "visualization/compact_mesh.h"
#include "visualization/bounding_box.h"
#include "sensor/rgbd_sensor.h"
#include "mapping_engine.h"

class MainEngine {
public:
  // configure main data
  MainEngine(
      const HashParams& hash_params,
      const VolumeParams& sdf_params,
      const MeshParams& mesh_params,
      const SensorParams& sensor_params,
      const RayCasterParams& ray_caster_params
  );
  ~MainEngine();
  void Reset();

  // configure engines
  void ConfigMappingEngine(
      bool enable_input_refine
  );
  void ConfigVisualizingEngine(
      Light& light,
      bool enable_navigation,
      bool enable_global_mesh,
      bool enable_bounding_box,
      bool enable_trajectory,
      bool enable_polygon_mode,
      bool enable_ray_caster,
      bool enable_color
  );

  void ConfigLoggingEngine(
      std::string path,
      bool enable_video,
      bool enable_ply
  );


  void Mapping(Sensor &sensor);
  void Meshing();
  void Recycle();
  int Visualize(float4x4 view);
  void Log();
  void RecordBlocks();
  void FinalLog();

  const int& frame_count() {
    return integrated_frame_count_;
  }
  bool& enable_sdf_gradient() {
    return enable_sdf_gradient_;
  }

private:
  // Engines
  MappingEngine     map_engine_;
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

  int             integrated_frame_count_ = 0;
  bool            enable_sdf_gradient_;

  HashParams hash_params_;
  VolumeParams  volume_params_;
  MeshParams mesh_params_;
  SensorParams sensor_params_;
  RayCasterParams ray_caster_params_;
};


#endif //VH_MAP_H
