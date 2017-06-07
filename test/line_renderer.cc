//
// Created by wei on 17-6-7.
//

#include "renderer.h"

int main() {
  float3 cpu_vertices[6] = {
          {0, 0, 0}, {0, 0, 1},
          {0, 0, 0}, {0, 1, 0},
          {0, 0, 0}, {1, 0, 0}
  };
  float3 *gpu_vertices;
  checkCudaErrors(cudaMalloc(&gpu_vertices, sizeof(float3) * 6));
  checkCudaErrors(cudaMemcpy(gpu_vertices, cpu_vertices, sizeof(float3) * 6,
                             cudaMemcpyHostToDevice));

  BBoxRenderer renderer("BBox", 640, 480, 3);
  renderer.free_walk() = true;

  while (true) {
    float4x4 mat; mat.setIdentity();
    renderer.Render(gpu_vertices, 6, mat);
  }
  return 0;
}