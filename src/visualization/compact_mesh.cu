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

//CompactMesh::~CompactMesh() {
//  Free();
//}

void CompactMesh::Alloc(const MeshParams &mesh_params) {
  checkCudaErrors(cudaMalloc(&vertex_remapper_,
                             sizeof(int) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&vertex_counter_,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&vertices_ref_count_,
                             sizeof(int) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&vertices_,
                             sizeof(float3) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&normals_,
                             sizeof(float3) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&colors_,
                             sizeof(float3) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&triangle_counter_,
                             sizeof(uint)));
  checkCudaErrors(cudaMalloc(&triangles_ref_count_,
                             sizeof(int) * mesh_params.max_triangle_count));
  checkCudaErrors(cudaMalloc(&triangles_,
                             sizeof(int3) * mesh_params.max_triangle_count));
}

void CompactMesh::Free() {
  checkCudaErrors(cudaFree(vertex_remapper_));

  checkCudaErrors(cudaFree(vertex_counter_));
  checkCudaErrors(cudaFree(vertices_ref_count_));
  checkCudaErrors(cudaFree(vertices_));
  checkCudaErrors(cudaFree(normals_));
  checkCudaErrors(cudaFree(colors_));

  checkCudaErrors(cudaFree(triangle_counter_));
  checkCudaErrors(cudaFree(triangles_ref_count_));
  checkCudaErrors(cudaFree(triangles_));
}

void CompactMesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  Alloc(mesh_params);
  Reset();
}

/// Reset
void CompactMesh::Reset() {
  checkCudaErrors(cudaMemset(vertex_remapper_, 0xff,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(vertices_ref_count_, 0,
                             sizeof(int) * mesh_params_.max_vertex_count));
  checkCudaErrors(cudaMemset(vertex_counter_,
                             0, sizeof(uint)));
  checkCudaErrors(cudaMemset(triangles_ref_count_, 0,
                             sizeof(int) * mesh_params_.max_triangle_count));
  checkCudaErrors(cudaMemset(triangle_counter_,
                             0, sizeof(uint)));
}

uint CompactMesh::vertex_count() {
  uint compact_vertex_count;
  checkCudaErrors(cudaMemcpy(&compact_vertex_count,
                             vertex_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_vertex_count;
}

uint CompactMesh::triangle_count() {
  uint compact_triangle_count;
  checkCudaErrors(cudaMemcpy(&compact_triangle_count,
                             triangle_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return compact_triangle_count;
}
