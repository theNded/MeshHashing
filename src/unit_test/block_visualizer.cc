//
// Created by wei on 17-6-10.
//

#include "renderer.h"
#include "block.h"
#include "color_util.h"

int main() {
  Renderer renderer("Block", 640, 480);
  renderer.free_walk() = true;

  std::ifstream in("../result/statistics/(1,1,5).txt");
  int block_side_length, block_voxel_count;
  in >> block_side_length >> block_voxel_count;
  LOG(INFO) << "Sizes: " << block_side_length << " " << block_voxel_count;

  LOG(INFO) << block_side_length << " " << block_voxel_count;
  PointObject voxel_points(block_voxel_count);
  float3 *voxel_centers = new float3[block_voxel_count];
  float3 *voxel_colors  = new float3[block_voxel_count];

  for (int i = 0; i < block_voxel_count; ++i) {
    float3 voxel_pos; float sdf; uint weight;
    in >> voxel_pos.x >> voxel_pos.y >> voxel_pos.z;
    in >> sdf >> weight;
    voxel_centers[i] = voxel_pos;
    voxel_colors[i] = (weight == 0) ? make_float3(1) : ValToRGB(sdf, -0.1f, 0.1f);
  }

  float3* cuda_voxel_centers, *cuda_voxel_colors;
  checkCudaErrors(cudaMalloc(&cuda_voxel_centers, sizeof(float3) * block_voxel_count));
  checkCudaErrors(cudaMalloc(&cuda_voxel_colors, sizeof(float3) * block_voxel_count));

  checkCudaErrors(cudaMemcpy(cuda_voxel_centers, voxel_centers,
                             sizeof(float3) * block_voxel_count,
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_voxel_colors, voxel_colors,
                             sizeof(float3) * block_voxel_count,
                             cudaMemcpyHostToDevice));

  voxel_points.SetData(cuda_voxel_centers, block_voxel_count,
                       cuda_voxel_colors, block_voxel_count);
  renderer.AddObject(&voxel_points);


  MeshObject mesh_object(block_voxel_count * 3, block_voxel_count * 5, kColor);
  int vertex_count;
  in >> vertex_count;
  LOG(INFO) << "Vertex count: " << vertex_count;
  std::vector<float3> vertex_positions;
  std::vector<float3> vertex_colors;
  for (int i = 0; i < vertex_count; ++i) {
    float3 pos, color;
    in >> pos.x >> pos.y >> pos.z >> color.x >> color.y >> color.z;
    vertex_positions.push_back(pos);
    vertex_colors.push_back(color);
  }

  int triangle_count;
  in >> triangle_count;
  LOG(INFO) << "Triangle count: " << triangle_count;
  std::vector<int3> triangles;
  for (int i = 0; i < triangle_count; ++i) {
    int3 triangle;
    in >> triangle.x >> triangle.y >> triangle.z;
    triangles.push_back(triangle);
  }

  float3* cuda_vertex_positions;
  float3 *cuda_vertex_colors;
  int3 *cuda_triangle_indices;
  checkCudaErrors(cudaMalloc(&cuda_vertex_positions, sizeof(float3) * vertex_positions.size()));
  checkCudaErrors(cudaMalloc(&cuda_vertex_colors, sizeof(float3) * vertex_colors.size()));
  checkCudaErrors(cudaMalloc(&cuda_triangle_indices, sizeof(int3) * triangles.size()));

  checkCudaErrors(cudaMemcpy(cuda_vertex_positions, vertex_positions.data(),
                             sizeof(float3) * vertex_positions.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_vertex_colors, vertex_colors.data(),
                             sizeof(float3) * vertex_colors.size(),
                             cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_triangle_indices, triangles.data(),
                             sizeof(int3) * triangles.size(),
                             cudaMemcpyHostToDevice));

  mesh_object.ploygon_mode() = 1;
  mesh_object.SetData(cuda_vertex_positions, vertex_positions.size(),
                      NULL, 0,
                      cuda_vertex_colors, vertex_colors.size(),
                      cuda_triangle_indices, triangles.size());
  renderer.AddObject(&mesh_object);

  while (true) {
    float4x4 dummy;
    renderer.Render(dummy);
  }

  return 0;
}
