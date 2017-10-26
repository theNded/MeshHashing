//
// Created by wei on 17-10-26.
//

#ifndef MESH_HASHING_MAPPING_ENGINE_H
#define MESH_HASHING_MAPPING_ENGINE_H

#include "optimize/linear_equations.h"
#include "helper_math.h"

class MappingEngine {
public:
  MappingEngine() = default;
  void Init(
      int sensor_width,
      int sensor_height,
      bool enable_input_refine
  );
  MappingEngine(
      int sensor_width,
      int sensor_height,
      bool enable_input_refine
  );
  ~MappingEngine();

  bool enable_input_refine() {
    return enable_input_refine_;
  }
  SensorLinearEquations& linear_equations() {
    return linear_equations_;
  }
private:
  bool enable_input_refine_ = false;
  SensorLinearEquations linear_equations_;
};


#endif //MESH_HASHING_MAPPING_ENGINE_H
