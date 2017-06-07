//
// Created by wei on 17-6-4.
//

#include <glog/logging.h>
#include <vector>
#include <algorithm>
#include "debugger.h"

Debugger::Debugger(int entry_count, int block_count) {
  entry_count_ = entry_count;
  block_count_ = block_count;

  entries_ = new HashEntry[entry_count];
  heap_ = new uint[block_count];
  heap_counter_ = new uint[1];

  blocks_ = new Block[block_count];
}

Debugger::~Debugger() {
  delete[] entries_;
  delete[] heap_;
  delete[] heap_counter_;
  delete[] blocks_;
}

void Debugger::CoreDump(HashTableGPU &hash_table) {
  checkCudaErrors(cudaMemcpy(entries_, hash_table.entries,
                             sizeof(HashEntry) * entry_count_,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(heap_, hash_table.heap,
                             sizeof(uint) * block_count_,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(heap_counter_, hash_table.heap_counter,
                             sizeof(uint),
                             cudaMemcpyDeviceToHost));
}

void Debugger::CoreDump(BlocksGPU &blocks) {
  checkCudaErrors(cudaMemcpy(blocks_, blocks,
                             sizeof(Blocks) * block_count_,
                             cudaMemcpyDeviceToHost));
}

void Debugger::DebugHashToBlock() {
  std::vector<int3> block_pos;

  for (int i = 0; i < entry_count_; ++i) {
    HashEntry& entry = entries_[i];
    if (entry.ptr != FREE_ENTRY) {
      Block& block = blocks_[entry.ptr];
      block_pos.push_back(entry.pos);
    }
  }

  std::sort(block_pos.begin(), block_pos.end(),
            [](const int3& a, const int3 &b) {
              return (a.x < b.x)
                     || (a.x == b.x && a.y < b.y)
                     || (a.x == b.x && a.y == b.y && a.z < b.z);
  });

  for (auto& pos : block_pos) {
    std::cout << pos.x << ", " << pos.y << ", " << pos.z << std::endl;
  }

  //getchar();
}