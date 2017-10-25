//
// Created by wei on 17-10-21.
//

#include "bounding_box.h"
#include "core/common.h"
#include "core/entry_array.h"
#include "geometry/geometry_helper.h"
#include "helper_cuda.h"

//BoundingBox::~BoundingBox() {
//  Free();
//}

void BoundingBox::Alloc(int max_vertex_count) {
  if (!is_allocated_on_gpu_) {
    checkCudaErrors(cudaMalloc(&vertex_counter_, sizeof(uint)));
    checkCudaErrors(cudaMalloc(&vertices_, sizeof(float3) * max_vertex_count));
    is_allocated_on_gpu_ = true;
  }
}

void BoundingBox::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(vertex_counter_));
    checkCudaErrors(cudaFree(vertices_));
    is_allocated_on_gpu_ = false;
  }
}

void BoundingBox::Resize(int max_vertex_count) {
  max_vertex_count_ = max_vertex_count;
  if (is_allocated_on_gpu_) {
    Free();
  }
  Alloc(max_vertex_count);
  Reset();
}

void BoundingBox::Reset() {
  checkCudaErrors(cudaMemset(vertex_counter_, 0, sizeof(uint)));
}

uint BoundingBox::vertex_count() {
  uint vertex_count;
  checkCudaErrors(cudaMemcpy(&vertex_count,
                             vertex_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return vertex_count;
}