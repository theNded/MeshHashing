#include "mesh.h"

#include <helper_cuda.h>
#include <device_launch_parameters.h>
#include "params.h"
#include <glog/logging.h>

////////////////////
/// class Mesh
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void MeshResetVerticesKernel(uint* vertex_heap, Vertex* vertices, int max_vertex_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_vertex_count) {
    vertex_heap[idx] = max_vertex_count - idx - 1;
    vertices[idx].Clear();
  }
}

__global__
void MeshResetTrianglesKernel(uint* triangle_heap, Triangle* triangles, int max_triangle_count) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < max_triangle_count) {
    triangle_heap[idx] = max_triangle_count - idx - 1;
    triangles[idx].Clear();
  }
}

////////////////////
/// Host code
////////////////////
__host__ Mesh::Mesh() {}

// Mesh::~Mesh() {
  //Free();
//}

__host__
void Mesh::Alloc(const MeshParams &mesh_params) {
  checkCudaErrors(cudaMalloc(&vertex_heap_,
                             sizeof(uint) * mesh_params.max_vertex_count));
  checkCudaErrors(cudaMalloc(&vertex_heap_counter_, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&vertices,
                             sizeof(Vertex) * mesh_params.max_vertex_count));

  checkCudaErrors(cudaMalloc(&triangle_heap_,
                             sizeof(uint) * mesh_params.max_triangle_count));
  checkCudaErrors(cudaMalloc(&triangle_heap_counter_, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&triangles,
                             sizeof(Triangle) * mesh_params.max_triangle_count));
}

void Mesh::Free() {
  checkCudaErrors(cudaFree(vertex_heap_));
  checkCudaErrors(cudaFree(vertex_heap_counter_));
  checkCudaErrors(cudaFree(vertices));

  checkCudaErrors(cudaFree(triangle_heap_));
  checkCudaErrors(cudaFree(triangle_heap_counter_));
  checkCudaErrors(cudaFree(triangles));
}

void Mesh::Resize(const MeshParams &mesh_params) {
  mesh_params_ = mesh_params;
  Alloc(mesh_params);
  Reset();
}

void Mesh::Reset() {
  uint val;

  val = mesh_params_.max_vertex_count - 1;
  checkCudaErrors(cudaMemcpy(vertex_heap_counter_,
                             &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  val = mesh_params_.max_triangle_count - 1;
  checkCudaErrors(cudaMemcpy(triangle_heap_counter_,
                             &val,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));

  {
    const int threads_per_block = 64;
    const dim3 grid_size((mesh_params_.max_vertex_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    MeshResetVerticesKernel<<< grid_size, block_size >>> (vertex_heap_, vertices,
        mesh_params_.max_vertex_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    const int threads_per_block = 64;
    const dim3 grid_size((mesh_params_.max_triangle_count + threads_per_block - 1)
                         / threads_per_block, 1);
    const dim3 block_size(threads_per_block, 1);

    MeshResetTrianglesKernel<<<grid_size, block_size>>> (triangle_heap_, triangles,
        mesh_params_.max_triangle_count);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

uint Mesh::vertex_heap_count() {
  uint vertex_heap_count;
  checkCudaErrors(cudaMemcpy(&vertex_heap_count,
                             vertex_heap_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return vertex_heap_count;
}

uint Mesh::triangle_heap_count() {
  uint triangle_heap_count;
  checkCudaErrors(cudaMemcpy(&triangle_heap_count,
                             triangle_heap_counter_,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  return triangle_heap_count;
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