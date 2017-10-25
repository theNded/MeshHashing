//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_MARCHING_CUBES_H
#define MESH_HASHING_MARCHING_CUBES_H

#include <glog/logging.h>
#include <unordered_map>
#include <chrono>

#include <ctime>
#include "mc_tables.h"
#include "util/timer.h"
#include "engine/main_engine.h"
#include "core/collect_block_array.h"

void MarchingCubes(EntryArray& candidate_entries,
                   HashTable& hash_table,
                   BlockArray& blocks,
                   Mesh& mesh,
                   bool use_fine_gradient,
                   GeometryHelper& geoemtry_helper);
#endif //MESH_HASHING_MARCHING_CUBES_H
