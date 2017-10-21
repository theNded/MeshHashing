//
// Created by wei on 17-5-21.
//

#ifndef CORE_BLOCK_H
#define CORE_BLOCK_H

#include <iostream>

#include "core/common.h"
#include "core/voxel.h"

#include "helper_math.h"

// Typically Block is a 8x8x8 voxel array
struct __ALIGN__(8) Block {
  Voxel voxels[BLOCK_SIZE];

  __host__ __device__
  void Clear() {
#ifdef __CUDA_ARCH__ // __CUDA_ARCH__ is only defined for __device__
#pragma unroll 8
#endif
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      voxels[i].Clear();
    }
  }
};

typedef Block* BlockGPUMemory;
typedef Block* BlockCPUMemory;

// TODO: enable CPU version
class Blocks {
public:
  Blocks();
  Blocks(uint block_count);
  ~Blocks();

  void Reset();
  void Resize(uint block_count);

  BlockGPUMemory &gpu_memory() {
    return gpu_memory_;
  }

private:
  BlockGPUMemory gpu_memory_;
  uint           block_count_;

  void Alloc(uint block_count);
  void Free();
};

#endif // CORE_BLOCK_H
