//
// Created by wei on 17-5-21.
//

#ifndef CORE_BLOCK_H
#define CORE_BLOCK_H

#include "core/common.h"
#include "core/voxel.h"

#include <helper_math.h>

// Typically Block is a 8x8x8 voxel array
struct __ALIGN__(8) Block {
  Voxel voxels[BLOCK_SIZE];
  MeshUnit mesh_units[BLOCK_SIZE];
  PrimalDualVariables primal_dual_variables[BLOCK_SIZE];

  __host__ __device__
  void Clear() {
#ifdef __CUDA_ARCH__ // __CUDA_ARCH__ is only defined for __device__
#pragma unroll 8
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      voxels[i].Clear();
      mesh_units[i].Clear();
      primal_dual_variables[i].Clear();
    }
  }
};

#endif // CORE_BLOCK_H
