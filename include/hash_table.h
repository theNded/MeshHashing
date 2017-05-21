//
// Created by wei on 17-4-28.
//
// Host (CPU) wrapper for the hash table data allocated on the GPU
// Host operations: alloc, free, and reset

#ifndef VH_HASH_TABLE_H
#define VH_HASH_TABLE_H

#include "hash_table_gpu.h"

/// Generally, the template should be implemented entirely in a header
/// However, we need CUDA code that has to be in .cu
/// Hence, we separate the declaration and implementation
/// And specifically instantiate it with @typename Block in the .cu
class HashTable {
private:
  HashTableGPU gpu_data_;
  HashParams hash_params_;

  void Alloc(const HashParams &params);
  void Free();

public:
  HashTable();
  HashTable(const HashParams &params);
  ~HashTable();

  uint compacted_entry_count();
  void Resize(const HashParams &params);
  void Reset();
  void ResetMutexes();

  void CollectAllEntries();

  void Debug();

  HashTableGPU& gpu_data() {
    return gpu_data_;
  }
};

#endif //VH_HASH_TABLE_H
