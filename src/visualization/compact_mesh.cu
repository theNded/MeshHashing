//
// Created by wei on 17-10-21.
//

#include "compact_mesh.h"
#include "helper_cuda.h"
////////////////////
/// class CompactMesh
////////////////////

/// Life cycle
CompactMesh::CompactMesh() {}

CompactMesh::~CompactMesh() {
  Free();
}

void CompactMesh::Alloc(const MeshParams &mesh_params) {
  checkCudaErrors(cudaMalloc(&gpu_memory_.vertex_remapper,
                             sizeof(int) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&gpu_memory_.vertex_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_memory_.vertices_ref_count,
                             sizeof(int) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_memory_.vertices,
                             sizeof(float3) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_memory_.normals,
                             sizeof(float3) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&gpu_memory_.colors,
                             sizeof(float3) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&gpu_memory_.triangle_counter,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&gpu_memory_.triangles_ref_count,
                             sizeof(int) * mesh_params.max_triangle_count));
  checkCudaErrors(cudaMalloc(&gpu_memory_.triangles,
                             sizeof(int3) * mesh_params.max_triangle_count));
}

void CompactMesh::Free() {
  checkCudaErrors(cudaFree(gpu_memory_.vertex_remapper));

  checkCudaErrors(cudaFree(gpu_memory_.vertex_counter));
  checkCudaErrors(cudaFree(gpu_memory_.vertices_ref_count));
  checkCudaErrors(cudaFree(gpu_memory_.vertices));
  checkCudaErrors(cudaFree(gpu_memory_.normals));
  checkCudaErrors(cudaFree(gpu_memory_.colors));

  checkCudaErrors(cudaFree(gpu_memory_.triangle_counter));
  checkCudaErrors(cudaFree(gpu_memory_.triangles_ref_count));
  checkCudaErrors(cudaFree(gpu_memory_.triangles));
}

void CompactMesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  Alloc(mesh_params);
  Reset();
}

/// Reset
void CompactMesh::Reset() {
  checkCudaErrors(cudaMemset(gpu_memory_.vertex_remapper, 0xff,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(gpu_memory_.vertices_ref_count, 0,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(gpu_memory_.vertex_counter,
                             0, sizeof(uint)));
  checkCudaErrors(cudaMemset(gpu_memory_.triangles_ref_count, 0,
                             sizeof(int) * mesh_params_.max_triangle_count));
  checkCudaErrors(cudaMemset(gpu_memory_.triangle_counter,
                             0, sizeof(uint)));
}

uint CompactMesh::vertex_count() {
  uint compact_vertex_count;
  checkCudaErrors(cudaMemcpy(&compact_vertex_count,
                             gpu_memory_.vertex_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_vertex_count;
}

uint CompactMesh::triangle_count() {
  uint compact_triangle_count;
  checkCudaErrors(cudaMemcpy(&compact_triangle_count,
                             gpu_memory_.triangle_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_triangle_count;
}
