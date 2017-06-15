//
// Created by wei on 17-6-4.
//

#include <glog/logging.h>
#include "debugger.h"

Debugger::Debugger(int entry_count, int block_count,
                   int vertex_count, int triangle_count) {
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

void Debugger::CoreDump(BlocksGPU &blocks) {
  LOG(INFO) << block_count_;
  checkCudaErrors(cudaMemcpy(blocks_, blocks,
                             sizeof(Block) * block_count_,
                             cudaMemcpyDeviceToHost));
}

void Debugger::CoreDump(MeshGPU &mesh) {
  checkCudaErrors(cudaMemcpy(triangle_heap_counter_,
                             mesh.triangle_heap_counter,
                             sizeof(uint),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(vertex_heap_counter_,
                             mesh.vertex_heap_counter,
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

void Debugger::DebugHashToBlock() {
  for (int i = 0; i < *heap_counter_; ++i) {
    HashEntry& entry = entries_[i];
    if (entry.ptr != FREE_ENTRY) {
      Block& block = blocks_[entry.ptr];
      block_map_[entry.pos] = block;
    }
  }
}

std::vector<BlockAnalyzer> Debugger::DebugBlockToMesh() {
  std::vector<BlockAnalyzer> vec;

  for (auto& item : block_map_) {
    BlockAnalyzer block_stats;
    block_stats.ssdf.reserve(BLOCK_SIZE);
    block_stats.sweight.reserve(BLOCK_SIZE);

    for (int i = 0; i < BLOCK_SIZE; ++i) {
      block_stats.ssdf.push_back(item.second.voxels[i].ssdf);
      block_stats.sweight.push_back(item.second.voxels[i].sweight);

      for (int j = 0; j < 5; ++j) {
        int triangle_ptr = item.second.cubes[i].triangle_ptrs[j];
        if (triangle_ptr >= 0) {
          int3 vertex_ptrs = triangles_[triangle_ptr].vertex_ptrs;
          block_stats.vertices.push_back(vertices_[vertex_ptrs.x].pos);
          block_stats.vertices.push_back(vertices_[vertex_ptrs.y].pos);
          block_stats.vertices.push_back(vertices_[vertex_ptrs.z].pos);

          int n = block_stats.triangles.size();
          block_stats.triangles.push_back(make_int3(n, n+1, n+2));
        }
      }
    }
    vec.push_back(block_stats);
  }
  return vec;
}

void Debugger::DebugAll() {
  for (int i = 0; i < *heap_counter_; ++i) {
    HashEntry& entry = entries_[i];
    std::stringstream ss;

    if (entry.ptr != FREE_ENTRY) {
      Block& block = blocks_[entry.ptr];
      ss.str("");
      int3 pos = entry.pos;
      ss << "../result/statistics/" << pos.x << "-" << pos.y << "-" << pos.z << ".txt";
      std::ofstream out(ss.str());

      out << BLOCK_SIZE << "\n";
      int3 voxel_pos = pos * BLOCK_SIDE_LENGTH;
      BlockAnalyzer block_stats;
      block_stats.ssdf.reserve(BLOCK_SIZE);
      block_stats.sweight.reserve(BLOCK_SIZE);

      for (int i = 0; i < BLOCK_SIZE; ++i) {
        uint3 local_pos = IdxToVoxelLocalPos(i);
        float3 world_pos = 0.04f * make_float3(voxel_pos + make_int3(local_pos));

        out << world_pos.x << " " << world_pos.y << " " << world_pos.z << " "
            << block.voxels[i].sdf() << " "
            << block.voxels[i].ssdf.x << " " << block.voxels[i].ssdf.y << " "
            << (int)block.voxels[i].sweight.x << " " << (int)block.voxels[i].sweight.y << " "
            << "\n";

        LOG(INFO) << "----------";
        block_stats.ssdf.push_back(block.voxels[i].ssdf);
        block_stats.sweight.push_back(block.voxels[i].sweight);

        for (int j = 0; j < 5; ++j) {
          int triangle_ptr = block.cubes[i].triangle_ptrs[j];
          if (triangle_ptr >= 0) {
            int3 vertex_ptrs = triangles_[triangle_ptr].vertex_ptrs;
            block_stats.vertices.push_back(vertices_[vertex_ptrs.x].pos);
            block_stats.vertices.push_back(vertices_[vertex_ptrs.y].pos);
            block_stats.vertices.push_back(vertices_[vertex_ptrs.z].pos);

            block_stats.colors.push_back(vertices_[vertex_ptrs.x].color);
            block_stats.colors.push_back(vertices_[vertex_ptrs.y].color);
            block_stats.colors.push_back(vertices_[vertex_ptrs.z].color);

            int n = block_stats.triangles.size();
            block_stats.triangles.push_back(make_int3(3 * n, 3* n+1, 3*n+2));
          }
        }
      }

      out << block_stats.vertices.size() << "\n";
      for (int i = 0; i < block_stats.vertices.size(); ++i) {
        float3 pos = block_stats.vertices[i];
        float3 color = block_stats.colors[i];
        out << pos.x << " " << pos.y << " " << pos.z << " "
            << color.x << " " << color.y << " " << color.z << "\n";
      }

      out << block_stats.triangles.size() << "\n";
      for (auto& triangle : block_stats.triangles) {
        out << triangle.x << " " << triangle.y << " " << triangle.z << "\n";
      }
    }
  }
}

void Debugger::PrintDebugInfo() {
  for (auto& item : block_map_) {
    int3 pos = item.first;
    LOG(INFO) << "(" << pos.x << " " << pos.y << " " << pos.z << "): ";
  }
}
