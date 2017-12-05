//
// Created by wei on 17-10-21.
//

#ifndef CORE_BLOCK_ARRAY_H
#define CORE_BLOCK_ARRAY_H

#include "core/block.h"

// Pre-allocated blocks to store the map
class BlockArray {
public:
  __host__ BlockArray() = default;
  __host__ explicit BlockArray(uint block_count);

  // We have to pass VALUE instead of REFERENCE to GPU,
  // therefore destructor will be called after a kernel launch,
  // and improper Free() will be triggered.
  // So if on GPU, disable destructor (temporarily),
  // and call Free() manually.
  // TODO: when CPU version is implemented, let it decide when to call Free()
  //__host__ ~BlockArray();

  __host__ void Alloc(uint block_count);
  __host__ void Resize(uint block_count);
  __host__ void Free();

  __host__ void Reset();

  __host__ __device__ Block& operator[] (uint i) {
    return blocks_[i];
  }
  __host__ __device__ const Block& operator[] (uint i) const {
    return blocks_[i];
  }

  __host__ Block* GetGPUPtr() const{
    return blocks_;
  }
private:
  bool is_allocated_on_gpu_ = false;
  // @param array
  Block*  blocks_;
  // @param const element
  uint    block_count_;
};

#endif // CORE_BLOCK_ARRAY_H
