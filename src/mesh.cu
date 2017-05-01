#include "mesh.h"

__global__
void ResetHeapKernel(MeshData mesh_data) {
  const uint max_vertice_count = 2500000;
  uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    mesh_data.heap_counter[0] = max_vertice_count - 1;	//points to the last element of the array
  }

  if (idx < max_vertice_count) {
    mesh_data.heap[idx] = max_vertice_count - idx - 1;
    mesh_data.vertices[idx].Clear();
  }
}

Mesh::Mesh(const HashParams &params) {
  const int max_vertex_count = 2500000;
  checkCudaErrors(cudaMalloc(&mesh_data_.heap,
                             sizeof(uint) * max_vertex_count));
  checkCudaErrors(cudaMalloc(&mesh_data_.heap_counter, sizeof(uint)));
  checkCudaErrors(cudaMalloc(&mesh_data_.vertices,
                             sizeof(Vertex) * max_vertex_count));
  hash_table_.Resize(params);
  Reset();
}

Mesh::~Mesh() {
  checkCudaErrors(cudaFree(mesh_data_.heap));
  checkCudaErrors(cudaFree(mesh_data_.heap_counter));
  checkCudaErrors(cudaFree(mesh_data_.vertices));
}

void Mesh::Reset() {
  const int max_vertex_count = 2500000;
  const int threads_per_block = 64;
  const dim3 grid_size((max_vertex_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  ResetHeapKernel<<<grid_size, block_size>>>(mesh_data_);
}

void Mesh::CollectTargetBlocks(float4x4 c_T_w) {

}

void Mesh::MarchingCubes(Map *map) {
  /// Assume it is compactified
}