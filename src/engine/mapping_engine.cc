//
// Created by wei on 17-10-22.
//

#include "engine/mapping_engine.h"
#include "mapping/fusion.h"
#include "mapping/recycle.h"
#include "core/collect.h"
#include "visualization/compress_mesh.h"

////////////////////
/// Host code
////////////////////
void MappingEngine::Integrate(Sensor& sensor) {
  AllocBlockArray(hash_table_, sensor, coordinate_converter_);

  CollectInFrustumBlockArray(hash_table_, candidate_entries_, sensor, coordinate_converter_);
  UpdateBlockArray(candidate_entries_,
                   hash_table_,
                   blocks_, mesh_,
                   sensor,
                   coordinate_converter_);

  Recycle(integrated_frame_count_);
  integrated_frame_count_ ++;
}

void MappingEngine::Recycle(int frame_count) {
  // TODO(wei): change it via global parameters

  int kRecycleGap = 15;
  if (frame_count % kRecycleGap == kRecycleGap - 1) {
    StarveOccupiedBlockArray(candidate_entries_, blocks_);

    CollectGarbageBlockArray(candidate_entries_,
                             blocks_,
                             coordinate_converter_);
    hash_table_.ResetMutexes();
    RecycleGarbageBlockArray(hash_table_, candidate_entries_, blocks_, mesh_);
  }
}

/// Life cycle
MappingEngine::MappingEngine(const HashParams &hash_params,
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

MappingEngine::~MappingEngine() {
  time_profile_.close();
  memo_profile_.close();
}

/// Reset
void MappingEngine::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  candidate_entries_.Reset();
  compact_mesh_.Reset();
  bbox_.Reset();
}

void MappingEngine::CompressMesh(int3 &stats) {
  CompressMeshImpl(candidate_entries_, blocks_, mesh_, compact_mesh_, stats);
}