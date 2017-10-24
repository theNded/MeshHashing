//
// Created by wei on 17-10-22.
//

#include "engine/main_engine.h"
#include "mapping/allocate.h"
#include "mapping/update.h"
#include "mapping/recycle.h"
#include "core/collect_block_array.h"
#include "visualization/compress_mesh.h"
#include "visualization/extract_bounding_box.h"
#include "meshing/marching_cubes.h"


////////////////////
/// Host code
////////////////////
void MainEngine::Mapping(Sensor &sensor) {
  AllocBlockArray(hash_table_, sensor, coordinate_converter_);

  CollectInFrustumBlockArray(hash_table_, candidate_entries_, sensor, coordinate_converter_);

  UpdateBlockArray(candidate_entries_,
                   hash_table_,
                   blocks_,
                   mesh_,
                   sensor,
                   coordinate_converter_);
  integrated_frame_count_ ++;
}

void MainEngine::Meshing() {
  MarchingCubes(candidate_entries_,
                hash_table_,
                blocks_,
                mesh_,
                use_fine_gradient_,
                coordinate_converter_);
}

void MainEngine::Recycle() {
  // TODO(wei): change it via global parameters

  int kRecycleGap = 15;
  if (integrated_frame_count_ % kRecycleGap == kRecycleGap - 1) {
    StarveOccupiedBlockArray(candidate_entries_, blocks_);

    CollectGarbageBlockArray(candidate_entries_,
                             blocks_,
                             coordinate_converter_);
    hash_table_.ResetMutexes();
    RecycleGarbageBlockArray(hash_table_, candidate_entries_, blocks_, mesh_);
  }
}

void MainEngine::Visualizing(float4x4 view) {
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
    CollectAllBlockArray(candidate_entries_, hash_table_);
  }

  int3 timing;
  CompressMesh(candidate_entries_,
               blocks_,
               mesh_,
               vis_engine_.compact_mesh(),
               timing);

  if (vis_engine_.enable_bounding_box()) {
    vis_engine_.bounding_box().Reset();
    std::cout << 1 << std::endl;

    ExtractBoundingBox(candidate_entries_,
                       vis_engine_.bounding_box(),
                       coordinate_converter_);
  }

  vis_engine_.Render();

  if (vis_engine_.enable_ray_casting()) {
    vis_engine_.RenderRayCaster(view, hash_table_, blocks_, coordinate_converter_);
  }
}
/// Life cycle
MainEngine::MainEngine(const HashParams& hash_params,
                       const MeshParams &mesh_params,
                       const VolumeParams &sdf_params) {
  hash_params_ = hash_params;
  mesh_params_ = mesh_params;
  volume_params_ = sdf_params;

  hash_table_.Resize(hash_params);
  candidate_entries_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);
  //bbox_.Resize(hash_params.value_capacity * 24);

  coordinate_converter_.voxel_size = sdf_params.voxel_size;
  coordinate_converter_.truncation_distance_scale =
      sdf_params.truncation_distance_scale;
  coordinate_converter_.truncation_distance =
      sdf_params.truncation_distance;
  coordinate_converter_.sdf_upper_bound = sdf_params.sdf_upper_bound;
  coordinate_converter_.weight_sample = sdf_params.weight_sample;
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

void MainEngine::ConfigVisualizingEngineMesh(Light &light, bool free_viewpoints, bool render_global_mesh, bool enable_bounding_box) {
  vis_engine_.Init("VisEngine", 640, 480);
  vis_engine_.set_interaction_mode(free_viewpoints);
  vis_engine_.set_light(light);
  vis_engine_.BuildMultiLightGeometryProgram(mesh_params_.max_vertex_count,
                                             mesh_params_.max_triangle_count,
                                             render_global_mesh);
  vis_engine_.compact_mesh().Resize(mesh_params_);

  if (enable_bounding_box) {
    vis_engine_.BuildBoundingBoxProgram(hash_params_.value_capacity*24);
    vis_engine_.bounding_box().Resize(hash_params_.value_capacity*24);
  }
}

void MainEngine::ConfigVisualizingEngineRaycaster(const RayCasterParams &params) {
  vis_engine_.BuildRayCaster(params);
}