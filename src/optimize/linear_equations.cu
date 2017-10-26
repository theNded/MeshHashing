//
// Created by wei on 17-10-25.
//

#include "linear_equations.h"

void SensorLinearEquations::Alloc(int width, int height) {
  width_ = width;
  height_ = height;

  if (!is_allocated_on_gpu_) {
    cudaMalloc(&A, sizeof(float3x3) * width*height);
    cudaMalloc(&b, sizeof(float3) * width*height);
    is_allocated_on_gpu_ = true;
  }
}

void SensorLinearEquations::Free() {
  if (is_allocated_on_gpu_) {
    cudaFree(A);
    cudaFree(b);
    is_allocated_on_gpu_ = false;
  }
}

void SensorLinearEquations::Reset() {
  if (is_allocated_on_gpu_) {
    cudaMemset(A, 0, sizeof(float3x3)*width_*height_);
    cudaMemset(b, 0, sizeof(float3)*width_*height_);
  }
}