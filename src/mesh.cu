#include "mesh.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include <params.h>

////////////////////
/// class Mesh
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void ResetHeapKernel(MeshGPU mesh,
                     int max_vertex_count,
                     int max_triangle_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_vertex_count) {
    mesh.vertex_heap[idx] = max_vertex_count - idx - 1;
    mesh.vertices[idx].Clear();
  }
  if (idx < max_triangle_count) {
    mesh.triangle_heap[idx] = max_triangle_count - idx - 1;
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

void Mesh::Alloc(const MeshParams &mesh_params) {
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_heap,
                             sizeof(uint) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices,
                             sizeof(Vertex) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_heap,
                             sizeof(uint) * mesh_params.max_triangle_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles,
                             sizeof(Triangle) * mesh_params.max_triangle_count));
}

void Mesh::Free() {
  checkCudaErrors(cudaFree(gpu_data_.vertex_heap));
  checkCudaErrors(cudaFree(gpu_data_.vertex_heap_counter));
  checkCudaErrors(cudaFree(gpu_data_.vertices));

  checkCudaErrors(cudaFree(gpu_data_.triangle_heap));
  checkCudaErrors(cudaFree(gpu_data_.triangle_heap_counter));
  checkCudaErrors(cudaFree(gpu_data_.triangles));
}

void Mesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  Alloc(mesh_params);
  Reset();
}

void Mesh::Reset() {
  checkCudaErrors(cudaMemcpy(gpu_data_.vertex_heap_counter,
                             &mesh_params_.max_vertex_count,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(gpu_data_.triangle_heap_counter,
                             &mesh_params_.max_triangle_count,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  const int threads_per_block = 64;
  const dim3 grid_size((kMaxVertexCount + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetHeapKernel<<<grid_size, block_size>>>(gpu_data_,
          mesh_params_.max_vertex_count,
          mesh_params_.max_triangle_count);
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

void CompactMesh::Alloc(const MeshParams &mesh_params) {
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_index_remapper,
                             sizeof(int) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices_ref_count,
                             sizeof(int) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertices,
                             sizeof(float3) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.normals,
                             sizeof(float3) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&gpu_data_.triangle_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles_ref_count,
                             sizeof(int) * mesh_params.max_triangle_count));
  checkCudaErrors(cudaMalloc(&gpu_data_.triangles,
                             sizeof(int3) * mesh_params.max_triangle_count));
}

void CompactMesh::Free() {
  checkCudaErrors(cudaFree(gpu_data_.vertex_index_remapper));

  checkCudaErrors(cudaFree(gpu_data_.vertex_counter));
  checkCudaErrors(cudaFree(gpu_data_.vertices_ref_count));
  checkCudaErrors(cudaFree(gpu_data_.vertices));
  checkCudaErrors(cudaFree(gpu_data_.normals));

  checkCudaErrors(cudaFree(gpu_data_.triangle_counter));
  checkCudaErrors(cudaFree(gpu_data_.triangles_ref_count));
  checkCudaErrors(cudaFree(gpu_data_.triangles));
}

void CompactMesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  Alloc(mesh_params);
  Reset();
}

/// Reset
void CompactMesh::Reset() {
  checkCudaErrors(cudaMemset(gpu_data_.vertex_index_remapper, 0xff,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(gpu_data_.vertices_ref_count, 0,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(gpu_data_.vertex_counter,
                             0, sizeof(uint)));
  checkCudaErrors(cudaMemset(gpu_data_.triangles_ref_count, 0,
                             sizeof(int) * mesh_params_.max_triangle_count));
  checkCudaErrors(cudaMemset(gpu_data_.triangle_counter,
                             0, sizeof(uint)));
}

uint CompactMesh::vertex_count() {
  uint compact_vertex_count;
  checkCudaErrors(cudaMemcpy(&compact_vertex_count,
                             gpu_data_.vertex_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_vertex_count;
}

uint CompactMesh::triangle_count() {
  uint compact_triangle_count;
  checkCudaErrors(cudaMemcpy(&compact_triangle_count,
                             gpu_data_.triangle_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_triangle_count;
}
