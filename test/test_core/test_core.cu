#include <stdio.h>
#include "test_core.h"

/// Test simple wrapper
struct Wrapper {
  Voxel* voxel_;
};

__global__ void SetVoxel(Wrapper w) {
  printf("__global__: SetVoxel\n");

  w.voxel_->sdf = 3.14;
  w.voxel_->color = make_uchar3(12, 0, 3);
  w.voxel_->weight = 2;
}

__host__ Voxel TestCore::Run() {
  Wrapper  wrapper;
  Voxel    cpu_v;
  cudaMalloc(&wrapper.voxel_, sizeof(Voxel));
  SetVoxel<<<1, 1>>>(wrapper);
  cudaMemcpy(&cpu_v, wrapper.voxel_, sizeof(Voxel), cudaMemcpyDeviceToHost);
  cudaFree(&wrapper.voxel_);
  return cpu_v;
}