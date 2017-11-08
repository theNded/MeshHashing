//
// Created by wei on 17-10-22.
//

#include <glog/logging.h>
#include "io/mesh_writer.h"
#include "helper_cuda.h"
#include "visualization/compact_mesh.h"


//CollectAllBlocks(candidate_entries_, hash_table_);
//int3 stats;
//CompressMesh(stats);
void SaveObj(CompactMesh& compact_mesh, std::string path) {
  LOG(INFO) << "Copying data from GPU";


  uint compact_vertex_count = compact_mesh.vertex_count();
  uint compact_triangle_count = compact_mesh.triangle_count();
  LOG(INFO) << "Vertices: " << compact_vertex_count;
  LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];
  checkCudaErrors(cudaMemcpy(vertices, compact_mesh.vertices(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normals, compact_mesh.normals(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, compact_mesh.triangles(),
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  std::stringstream ss;
  LOG(INFO) << "Writing vertices";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss <<  "v " << vertices[i].x << " "
       << vertices[i].y << " "
       << vertices[i].z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing normals";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss << "vn " << normals[i].x << " "
       << normals[i].y << " "
       << normals[i].z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing faces";
  for (uint i = 0; i < compact_triangle_count; ++i) {
    ss.str("");
    int3 idx = triangles[i] + make_int3(1);
    ss << "f " << idx.x << "//" << idx.x << " "
         << idx.y << "//" << idx.y << " "
         << idx.z << "//" << idx.z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Finishing vertices";
  delete[] vertices;
  LOG(INFO) << "Finishing normals";
  delete[] normals;
  LOG(INFO) << "Finishing triangles";
  delete[] triangles;
}


void SavePly(CompactMesh& compact_mesh, std::string path) {
  LOG(INFO) << "Copying data from GPU";

  uint compact_vertex_count = compact_mesh.vertex_count();
  uint compact_triangle_count = compact_mesh.triangle_count();
  LOG(INFO) << "Vertices: " << compact_vertex_count;
  LOG(INFO) << "Triangles: " << compact_triangle_count;

  float3* vertices = new float3[compact_vertex_count];
  float3* normals  = new float3[compact_vertex_count];
  float3* colors = new float3[compact_vertex_count];
  int3* triangles  = new int3  [compact_triangle_count];
  checkCudaErrors(cudaMemcpy(vertices, compact_mesh.vertices(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normals, compact_mesh.normals(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(colors, compact_mesh.colors(),
                             sizeof(float3) * compact_vertex_count,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(triangles, compact_mesh.triangles(),
                             sizeof(int3) * compact_triangle_count,
                             cudaMemcpyDeviceToHost));

  std::ofstream out(path);
  std::stringstream ss;
  ////// Header
  ss.str("");
  ss << "ply\n"
      "format ascii 1.0\n";
  ss << "element vertex " << compact_vertex_count << "\n";
  ss << "property float x\n"
      "property float y\n"
      "property float z\n"
      "property float nx\n"
      "property float ny\n"
      "property float nz\n"
      "property uchar red\n"
      "property uchar green\n"
      "property uchar blue\n";
  ss << "element face " << compact_triangle_count << "\n";
  ss << "property list uchar int vertex_index\n";
  ss << "end_header\n";
  out << ss.str();

  LOG(INFO) << "Writing vertices";
  for (uint i = 0; i < compact_vertex_count; ++i) {
    ss.str("");
    ss << vertices[i].x << " "
       << vertices[i].y << " "
       << vertices[i].z << " "
       << normals[i].x << " "
       << normals[i].y << " "
       << normals[i].z << " "
       << int(255.0f * colors[i].x) << " "
       << int(255.0f * colors[i].y) << " "
       << int(255.0f * colors[i].z) << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Writing faces";
  for (uint i = 0; i < compact_triangle_count; ++i) {
    ss.str("");
    int3 idx = triangles[i];
    ss << "3 " << idx.x << " " << idx.y << " " << idx.z << "\n";
    out << ss.str();
  }

  LOG(INFO) << "Finishing vertices";
  delete[] vertices;
  LOG(INFO) << "Finishing normals";
  delete[] normals;
  LOG(INFO) << "Finishing triangles";
  delete[] triangles;
}
