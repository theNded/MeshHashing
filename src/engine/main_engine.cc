//
// Created by wei on 17-10-22.
//

#include "engine/main_engine.h"
#include "mapping/allocate.h"
#include "mapping/update.h"
#include "mapping/recycle.h"
#include "core/collect.h"
#include "visualization/compress_mesh.h"
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

/// Life cycle
MainEngine::MainEngine(const HashParams& hash_params,
                             const MeshParams &mesh_params,
                             const SDFParams &sdf_params) {
  hash_table_.Resize(hash_params);
  candidate_entries_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);
  compact_mesh_.Resize(mesh_params);
  bbox_.Resize(hash_params.value_capacity * 24);

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
  compact_mesh_.Free();
}

/// Reset
void MainEngine::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  candidate_entries_.Reset();
  compact_mesh_.Reset();
  bbox_.Reset();
}
