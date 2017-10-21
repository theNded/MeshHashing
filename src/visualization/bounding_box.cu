//
// Created by wei on 17-10-21.
//

#include "bounding_box.h"
#include "helper_cuda.h"

////////////////////
/// class BBox
////////////////////
BBox::BBox() {}
BBox::~BBox() {
  Free();
}

void BBox::Alloc(int max_vertex_count) {
  checkCudaErrors(cudaMalloc(&gpu_memory_.vertex_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_memory_.vertices,
                             sizeof(float3) * max_vertex_count));
}

void BBox::Free() {
  checkCudaErrors(cudaFree(gpu_memory_.vertex_counter));
  checkCudaErrors(cudaFree(gpu_memory_.vertices));
}

void BBox::Resize(int max_vertex_count) {
  max_vertex_count_ = max_vertex_count;
  Alloc(max_vertex_count);
  Reset();
}

void BBox::Reset() {
  checkCudaErrors(cudaMemset(gpu_memory_.vertex_counter,
                             0, sizeof(uint)));
}

uint BBox::vertex_count() {
  uint vertex_count;
  checkCudaErrors(cudaMemcpy(&vertex_count,
                             gpu_memory_.vertex_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return vertex_count;
}