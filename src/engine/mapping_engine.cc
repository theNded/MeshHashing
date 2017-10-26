//
// Created by wei on 17-10-26.
//

#include "mapping_engine.h"

MappingEngine::MappingEngine(
    int sensor_width,
    int sensor_height,
    bool enable_input_refine
) {
  Init(sensor_width, sensor_height, enable_input_refine);
}

void MappingEngine::Init(
    int sensor_width, int sensor_height,
    bool enable_input_refine
) {
  enable_input_refine_ = enable_input_refine;
  if (enable_input_refine) {
    linear_equations_.Alloc(sensor_width, sensor_height);
  }
}

MappingEngine::~MappingEngine() {
  linear_equations_.Free();
}