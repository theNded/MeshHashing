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
      bool enable_bayesian_update
  );
  MappingEngine(
      int sensor_width,
      int sensor_height,
      bool enable_bayesian_update
  );
  ~MappingEngine();


  bool enable_bayesian_update() {
    return enable_bayesian_update_;
  }
  SensorLinearEquations& linear_equations() {
    return linear_equations_;
  }
private:
  bool enable_bayesian_update_ = false;
  SensorLinearEquations linear_equations_;
};


#endif //MESH_HASHING_MAPPING_ENGINE_H
