//
// Created by wei on 17-10-26.
//

#include <core/entry_array.h>
#include <core/block_array.h>
#include <core/hash_table.h>
#include <mapping/update_simple.h>
#include <glog/logging.h>
#include "mapping_engine.h"
#include "optimize/primal_dual.h"

MappingEngine::MappingEngine(
    int sensor_width,
    int sensor_height,
    bool enable_bayesian_update
) {
  Init(sensor_width, sensor_height, enable_bayesian_update);
}

void MappingEngine::Init(
    int sensor_width, int sensor_height,
    bool enable_bayesian_update
) {
  enable_bayesian_update_ = enable_bayesian_update;
  if (enable_bayesian_update) {
    linear_equations_.Alloc(sensor_width, sensor_height);
  }
}

MappingEngine::~MappingEngine() {
  linear_equations_.Free();
}