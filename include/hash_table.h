//
// Created by wei on 17-4-28.
//

#ifndef VOXEL_HASHING_HASH_TABLE_H
#define VOXEL_HASHING_HASH_TABLE_H

#include "hash_table_gpu.h"

/// Generally, the template should be implemented entirely in a header
/// However, we need CUDA code that has to be in .cu
/// Hence, we separate the declaration and implementation
/// And specifically instantiate it in the .cu
template <typename T>
class HashTable {
private:
  HashTableGPU<T> gpu_data_;
  HashParams hash_params_;

  void Alloc(const HashParams &params);
  void Free();

public:
  HashTable();
  HashTable(const HashParams &params);
  ~HashTable();

  HashTableGPU<T>& gpu_data();

  void Resize(const HashParams &params);
  void Reset();
  void ResetMutexes();
};

#endif //VOXEL_HASHING_HASH_TABLE_H
