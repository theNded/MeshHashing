//
// Created by wei on 17-10-22.
//

#include <mapping/update_bayesian.h>
#include <optimize/primal_dual.h>
#include "engine/main_engine.h"

#include "core/collect_block_array.h"
#include "mapping/allocate.h"
#include "mapping/update_simple.h"
#include "mapping/recycle.h"
#include "meshing/marching_cubes.h"
#include "visualization/compress_mesh.h"
#include "visualization/extract_bounding_box.h"

////////////////////
/// Host code
////////////////////
void MainEngine::Mapping(Sensor &sensor) {
  double alloc_time = AllocBlockArray(
      hash_table_,
      sensor,
      geometry_helper_
  );

  double collect_time = CollectBlocksInFrustum(
      hash_table_,
      sensor,
      geometry_helper_,
      candidate_entries_
  );

  double update_time = 0;
  if (!map_engine_.enable_bayesian_update()) {
    LOG(INFO) << "Simple update";
    update_time = UpdateBlocksSimple(candidate_entries_,
                                     blocks_,
                                     sensor,
                                     hash_table_,
                                     geometry_helper_);
    log_engine_.WriteMappingTimeStamp(
        alloc_time,
        collect_time,
        update_time,
        integrated_frame_count_);
  } else {
    float predict_seconds = PredictOutlierRatio(
        candidate_entries_,
        blocks_,
        mesh_,
        sensor,
        hash_table_,
        geometry_helper_
    );
    float update_seconds = UpdateBlocksBayesian(
        candidate_entries_,
        blocks_,
        sensor,
        hash_table_,
        geometry_helper_
    );

    log_engine_.WriteMappingTimeStamp(
        alloc_time,
        collect_time,
        predict_seconds,
        update_time,
        integrated_frame_count_);
  }

  integrated_frame_count_ ++;
}

void MainEngine::Meshing() {
  float time = MarchingCubes(candidate_entries_,
                             blocks_,
                             mesh_,
                             hash_table_,
                             geometry_helper_,
                             enable_sdf_gradient_);
  log_engine_.WriteMeshingTimeStamp(time, integrated_frame_count_);
}

void MainEngine::Recycle() {
  // TODO(wei): change it via global parameters

  int kRecycleGap = 15;
  if (integrated_frame_count_ % kRecycleGap == kRecycleGap - 1) {
    StarveOccupiedBlockArray(candidate_entries_, blocks_);

    CollectGarbageBlockArray(candidate_entries_,
                             blocks_,
                             geometry_helper_);
    hash_table_.ResetMutexes();
    RecycleGarbageBlockArray(candidate_entries_,
                             blocks_,
                             mesh_,
                             hash_table_);
  }
}

// view: world -> camera
int MainEngine::Visualize(float4x4 view) {
  if (vis_engine_.enable_interaction()) {
    vis_engine_.update_view_matrix();
  } else {
    glm::mat4 glm_view;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        glm_view[i][j] = view.entries2[i][j];
    glm_view = glm::transpose(glm_view);
    vis_engine_.set_view_matrix(glm_view);
  }

  if (vis_engine_.enable_global_mesh()) {
    CollectAllBlocks(hash_table_, candidate_entries_);
  } // else CollectBlocksInFrustum

  int3 timing;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(),
               timing);

  if (vis_engine_.enable_bounding_box()) {
    vis_engine_.bounding_box().Reset();

    ExtractBoundingBox(candidate_entries_,
                       vis_engine_.bounding_box(),
                       geometry_helper_);
  }
  if (vis_engine_.enable_trajectory()) {
    vis_engine_.trajectory().AddPose(view.getInverse());
  }


  if (vis_engine_.enable_ray_casting()) {
    vis_engine_.RenderRayCaster(view,
                                blocks_,
                                hash_table_,
                                geometry_helper_);
  }

  return  vis_engine_.Render();
}

void MainEngine::Log() {
  if (log_engine_.enable_video()) {
    cv::Mat capture = vis_engine_.Capture();
    log_engine_.WriteVideo(capture);
  }
}

void MainEngine::FinalLog() {
  CollectAllBlocks(hash_table_, candidate_entries_);
  Meshing();
  int3 timing;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(), timing);
  if (log_engine_.enable_ply()) {
    log_engine_.WritePly(vis_engine_.compact_mesh());
  }
  log_engine_.WriteMeshStats(vis_engine_.compact_mesh().vertex_count(),
                             vis_engine_.compact_mesh().triangle_count());
}

/// Life cycle
MainEngine::MainEngine(
    const HashParams& hash_params,
    const VolumeParams &volume_params,
    const MeshParams &mesh_params,
    const SensorParams &sensor_params,
    const RayCasterParams &ray_caster_params
) {

  hash_params_ = hash_params;
  volume_params_ = volume_params;
  mesh_params_ = mesh_params;
  sensor_params_ = sensor_params;
  ray_caster_params_ = ray_caster_params;

  hash_table_.Resize(hash_params);
  candidate_entries_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);

  geometry_helper_.Init(volume_params);
}

MainEngine::~MainEngine() {
  hash_table_.Free();
  blocks_.Free();
  mesh_.Free();

  candidate_entries_.Free();
}

/// Reset
void MainEngine::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  candidate_entries_.Reset();
}

void MainEngine::ConfigMappingEngine(
    bool enable_bayesian_update
) {
  map_engine_.Init(sensor_params_.width,
                   sensor_params_.height,
                   enable_bayesian_update);
}

void MainEngine::ConfigVisualizingEngine(
    Light &light,
    bool enable_navigation,
    bool enable_global_mesh,
    bool enable_bounding_box,
    bool enable_trajectory,
    bool enable_polygon_mode,
    bool enable_ray_caster,
    bool enable_color
) {
  vis_engine_.Init("VisEngine", 640, 480);
  vis_engine_.set_interaction_mode(enable_navigation);
  vis_engine_.set_light(light);
  vis_engine_.BindMainProgram(mesh_params_.max_vertex_count,
                              mesh_params_.max_triangle_count,
                              enable_global_mesh,
                              enable_polygon_mode,
                              enable_color);
  vis_engine_.compact_mesh().Resize(mesh_params_);

  if (enable_bounding_box || enable_trajectory) {
    vis_engine_.BuildHelperProgram();
  }

  if (enable_bounding_box) {
    vis_engine_.InitBoundingBoxData(hash_params_.value_capacity*24);
  }
  if (enable_trajectory) {
    vis_engine_.InitTrajectoryData(10000);
  }

  if (enable_ray_caster) {
    vis_engine_.BuildRayCaster(ray_caster_params_);
  }
}

void MainEngine::ConfigLoggingEngine(
    std::string path,
    bool enable_video,
    bool enable_ply
) {
  log_engine_.Init(path);
  if (enable_video) {
    log_engine_.ConfigVideoWriter(640, 480);
  }
  if (enable_ply) {
    log_engine_.ConfigPlyWriter();
  }
}

void MainEngine::RecordBlocks(std::string prefix) {
  //CollectAllBlocks(hash_table_, candidate_entries_);
  BlockMap block_map = log_engine_.RecordBlockToMemory(
      blocks_.GetGPUPtr(), hash_params_.value_capacity,
      candidate_entries_.GetGPUPtr(), candidate_entries_.count());

  std::stringstream ss("");
  ss << integrated_frame_count_ - 1;
  log_engine_.WriteRawBlocks(block_map, prefix + ss.str());
}