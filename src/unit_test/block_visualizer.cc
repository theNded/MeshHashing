//
// Created by wei on 17-6-10.
//

#include "renderer.h"
#include "block.h"
#include "color_util.h"

int main() {
  Renderer renderer("Points", 640, 480);
  renderer.free_walk() = true;

  PointObject points(512);
  float3 vertices[8][8][8];
  float3 colors[8][8][8];

  std::ifstream in("../result/statistics/2-0-5.txt");
  int voxel_cnt;
  in >> voxel_cnt;

  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      for (int k = 0; k < 8; ++k) {
        float3 voxel_pos;float sdf;float2 ssdf;uint2 sweight;
        in >> voxel_pos.x >> voxel_pos.y >> voxel_pos.z;
        in >> sdf >> ssdf.x >> ssdf.y >> sweight.x >> sweight.y;
        vertices[i][j][k] = voxel_pos;
        if (sweight.x + sweight.y == 0)
          colors[i][j][k] = make_float3(1);
        else
          colors[i][j][k] = ValToRGB(sdf, -0.1f, 0.1f);
      }
    }
  }

  float3* cuda_points, *cuda_colors;
  checkCudaErrors(cudaMalloc(&cuda_points, sizeof(float3) * 512));
  checkCudaErrors(cudaMalloc(&cuda_colors, sizeof(float3) * 512));

  checkCudaErrors(cudaMemcpy(cuda_points, vertices,
                  sizeof(float3) * 512, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_colors, colors,
                  sizeof(float3) * 512, cudaMemcpyHostToDevice));

  points.SetData(cuda_points, 512,
                 cuda_colors, 512);
  renderer.AddObject(&points);


  MeshObject mesh_object(600, 600, kColor);
  int vertex_count;
  in >> vertex_count;
  std::vector<float3> vposs, vcolors;
  for (int i = 0; i < vertex_count; ++i) {
    float3 pos, color;
    in >> pos.x >> pos.y >> pos.z >> color.x >> color.y >> color.z;
    LOG(INFO) << pos.x << " " << pos.y << " " << pos.z << " "
              << color.x << " " << color.y << " " << color.z;
    vposs.push_back(pos);
    vcolors.push_back(color);
  }

  int triangle_count;
  in >> triangle_count;
  std::vector<int3> triangles;
  for (int i = 0; i < triangle_count; ++i) {
    int3 triangle;
    in >> triangle.x >> triangle.y >> triangle.z;
    triangles.push_back(triangle);
  }

  float3* cuda_vertices, *cuda_vcolors;
  int3 *cuda_indices;
  checkCudaErrors(cudaMalloc(&cuda_vertices, sizeof(float3) * 600));
  checkCudaErrors(cudaMalloc(&cuda_vcolors, sizeof(float3) * 600));
  checkCudaErrors(cudaMalloc(&cuda_indices, sizeof(int3) * 600));

  LOG(INFO) << vposs.size();
  checkCudaErrors(cudaMemcpy(cuda_vertices, vposs.data(),
                             sizeof(float3) * vposs.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_vcolors, vcolors.data(),
                             sizeof(float3) * vcolors.size(), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuda_indices, triangles.data(),
                             sizeof(int3) * triangles.size(), cudaMemcpyHostToDevice));

  mesh_object.ploygon_mode() = 1;
  mesh_object.SetData(cuda_vertices, vposs.size(),
                      NULL, 0,
                      cuda_vcolors, vposs.size(),
                      cuda_indices, triangles.size());
  renderer.AddObject(&mesh_object);
  while (true) {
    float4x4 dummy;
    renderer.Render(dummy);
  }

  return 0;
}
