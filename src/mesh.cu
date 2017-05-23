#include "mesh.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>

////////////////////
/// class Mesh
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void ResetHeapKernel(MeshGPU mesh) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < kMaxVertexCount) {
    mesh.vertex_heap[idx] = kMaxVertexCount - idx - 1;
    mesh.triangle_heap[idx] = kMaxVertexCount - idx - 1;
    mesh.vertices[idx].Clear();
    mesh.triangles[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
Mesh::Mesh() {}

Mesh::~Mesh() {
  Free();
}

void Mesh::Alloc(uint vertex_count, uint triangle_count) {
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_heap,
                             sizeof(uint) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices,
                             sizeof(Vertex) * kMaxVertexCount));

  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_heap,
                             sizeof(uint) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles,
                             sizeof(Triangle) * kMaxVertexCount));
}

void Mesh::Free() {
  checkCudaErrors(cudaFree(gpu_data_.vertex_heap));
  checkCudaErrors(cudaFree(gpu_data_.vertex_heap_counter));
  checkCudaErrors(cudaFree(gpu_data_.vertices));

  checkCudaErrors(cudaFree(gpu_data_.triangle_heap));
  checkCudaErrors(cudaFree(gpu_data_.triangle_heap_counter));
  checkCudaErrors(cudaFree(gpu_data_.triangles));
}

void Mesh::Resize(uint vertex_count, uint triangle_count) {
  Alloc(vertex_count, triangle_count);
  Reset();
}

void Mesh::Reset() {
  uint val = kMaxVertexCount - 1;
  checkCudaErrors(cudaMemcpy(gpu_data_.vertex_heap_counter, &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(gpu_data_.triangle_heap_counter, &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  const int threads_per_block = 64;
  const dim3 grid_size((kMaxVertexCount + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetHeapKernel<<<grid_size, block_size>>>(gpu_data_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

////////////////////
/// class CompactMesh
////////////////////

/// Life cycle
CompactMesh::CompactMesh() {}

CompactMesh::~CompactMesh() {
  Free();
}

void CompactMesh::Alloc(uint vertex_count, uint triangle_count) {
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_index_remapper,
                             sizeof(int) * kMaxVertexCount));

  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices_ref_count,
                             sizeof(int) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices,
                             sizeof(float3) * kMaxVertexCount));

  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles_ref_count,
                             sizeof(int) * kMaxVertexCount));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles,
                             sizeof(int3) * kMaxVertexCount));
}

void CompactMesh::Free() {
  checkCudaErrors(cudaFree(gpu_data_.vertex_index_remapper));

  checkCudaErrors(cudaFree(gpu_data_.vertex_counter));
  checkCudaErrors(cudaFree(gpu_data_.vertices_ref_count));
  checkCudaErrors(cudaFree(gpu_data_.vertices));

  checkCudaErrors(cudaFree(gpu_data_.triangle_counter));
  checkCudaErrors(cudaFree(gpu_data_.triangles_ref_count));
  checkCudaErrors(cudaFree(gpu_data_.triangles));
}

void CompactMesh::Resize(uint vertex_count, uint triangle_count) {
  Alloc(vertex_count, triangle_count);
  Reset();
}

/// Reset
void CompactMesh::Reset() {
  checkCudaErrors(cudaMemset(gpu_data_.vertex_index_remapper, 0xff,
                             sizeof(int) * kMaxVertexCount));
  checkCudaErrors(cudaMemset(gpu_data_.vertices_ref_count, 0,
                             sizeof(int) * kMaxVertexCount));
  checkCudaErrors(cudaMemset(gpu_data_.vertex_counter,
                             0, sizeof(uint)));
  checkCudaErrors(cudaMemset(gpu_data_.triangles_ref_count, 0,
                             sizeof(int) * kMaxVertexCount));
  checkCudaErrors(cudaMemset(gpu_data_.triangle_counter,
                             0, sizeof(uint)));
}