//
// Created by wei on 17-6-4.
//

#include <glog/logging.h>
#include "debugger.h"

Debugger::Debugger(int entry_count, int block_count,
                   int vertex_count, int triangle_count,
                   float voxel_size) {
  entry_count_ = entry_count;
  entries_ = new HashEntry[entry_count];
  heap_ = new uint[block_count];
  heap_counter_ = new uint[1];

  blocks_ = new Block[block_count];
  block_count_ = block_count;

  vertex_count_ = vertex_count;
  vertices_ = new Vertex[vertex_count];
  vertex_heap_ = new uint[vertex_count];
  vertex_heap_counter_ = new uint[1];

  triangle_count_ = triangle_count;
  triangles_ = new Triangle[triangle_count];
  triangle_heap_ = new uint[triangle_count];
  triangle_heap_counter_ = new uint[1];

  voxel_size_ = voxel_size;
}

Debugger::~Debugger() {
  delete[] entries_;
  delete[] heap_;
  delete[] heap_counter_;
  delete[] blocks_;
}

void Debugger::CoreDump(CompactHashTableGPU &hash_table) {
  checkCudaErrors(cudaMemcpy(heap_counter_, hash_table.compacted_entry_counter,
                             sizeof(int),
                             cudaMemcpyDeviceToHost));
  LOG(INFO) << *heap_counter_;
  checkCudaErrors(cudaMemcpy(entries_, hash_table.compacted_entries,
                             sizeof(HashEntry) * (*heap_counter_),
                             cudaMemcpyDeviceToHost));
}

/// Blocks are not compactified, so we have to dump all of them
void Debugger::CoreDump(BlocksGPU &blocks) {
  LOG(INFO) << block_count_;
  checkCudaErrors(cudaMemcpy(blocks_, blocks,
                             sizeof(Block) * block_count_,
                             cudaMemcpyDeviceToHost));
}

void Debugger::CoreDump(MeshGPU &mesh) {
  checkCudaErrors(cudaMemcpy(triangle_heap_counter_, mesh.triangle_heap_counter,
                             sizeof(uint),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(vertex_heap_counter_, mesh.vertex_heap_counter,
                             sizeof(uint),
                             cudaMemcpyDeviceToHost));

  LOG(INFO) << "Triangles: " << *triangle_heap_counter_;
  LOG(INFO) << "Vertices: " << *vertex_heap_counter_;

  checkCudaErrors(cudaMemcpy(triangles_, mesh.triangles,
                             sizeof(Triangle) * triangle_count_,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(vertices_, mesh.vertices,
                             sizeof(Vertex) * vertex_count_,
                             cudaMemcpyDeviceToHost));
}

void Debugger::DebugAll() {
  /// entry -> block -> (sdf, mesh)

  std::stringstream ss;
  for (int e = 0; e < *heap_counter_; ++e) {
    LOG(INFO) << "Entry " << e;

    HashEntry &entry = entries_[e];
    CHECK(entry.ptr != FREE_ENTRY) << "Invalid compacted entry!";
    Block &block = blocks_[entry.ptr];

    ss.str("");
    ss << "../result/statistics/(" << entry.pos.x << "," << entry.pos.y << "," << entry.pos.z << ").txt";
    std::ofstream out(ss.str());
    LOG(INFO) << "Writing to " << ss.str();

    out << BLOCK_SIDE_LENGTH << " " << BLOCK_SIZE << "\n";
    int3 voxel_pos = entry.pos * BLOCK_SIDE_LENGTH;

    std::vector<float3> vertices;
    std::vector<float3> colors;
    std::vector<int3>   triangles;
    for (int i = 0; i < BLOCK_SIZE; ++i) {
      /// SDF info
      uint3 local_pos = IdxToVoxelLocalPos(i);
      float3 world_pos = voxel_size_ * make_float3(voxel_pos + make_int3(local_pos));

      out << world_pos.x << " " << world_pos.y << " " << world_pos.z << " "
          << block.voxels[i].sdf << " " << (int) block.voxels[i].weight
          << "\n";

      /// Mesh info
      for (int j = 0; j < 5; ++j) {
        int triangle_ptr = block.voxels[i].triangle_ptrs[j];
        if (triangle_ptr >= 0) {
          int3 vertex_ptrs = triangles_[triangle_ptr].vertex_ptrs;
          vertices.push_back(vertices_[vertex_ptrs.x].pos);
          vertices.push_back(vertices_[vertex_ptrs.y].pos);
          vertices.push_back(vertices_[vertex_ptrs.z].pos);

          colors.push_back(vertices_[vertex_ptrs.x].color);
          colors.push_back(vertices_[vertex_ptrs.y].color);
          colors.push_back(vertices_[vertex_ptrs.z].color);

          int n = int(triangles.size());
          triangles.push_back(make_int3(3 * n, 3 * n + 1, 3 * n + 2));
        }
      }
    }

    out << vertices.size() << "\n";
    for (int i = 0; i < vertices.size(); ++i) {
      float3 pos   = vertices[i];
      float3 color = colors[i];
      out << pos.x << " " << pos.y << " " << pos.z << " "
          << color.x << " " << color.y << " " << color.z << "\n";
    }

    out << triangles.size() << "\n";
    for (auto &triangle : triangles) {
      out << triangle.x << " " << triangle.y << " " << triangle.z << "\n";
    }
  }
}
