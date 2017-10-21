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