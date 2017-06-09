//
// Created by wei on 17-6-4.
//

#include <glog/logging.h>
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
  for (int i = 0; i < entry_count_; ++i) {
    HashEntry& entry = entries_[i];
    if (entry.ptr != FREE_ENTRY) {
      Block& block = blocks_[entry.ptr];
      if (block_map_.find(entry.pos) == block_map_.end()) {
        std::vector<Block> vec;
        vec.push_back(block);
        block_map_.emplace(entry.pos, vec);
      } else {
        block_map_[entry.pos].push_back(block);
      }
    }
  }
}

void Debugger::PrintDebugInfo() {
  for (auto& item : block_map_) {
    int3 pos = item.first;
    LOG(INFO) << "(" << pos.x << " " << pos.y << " " << pos.z << "): "
              << item.second.size();
  }
}

