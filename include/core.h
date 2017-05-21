//
// Created by wei on 17-3-12.
//
// Core data structures for VoxelHashing
// Lowest level structs

#ifndef VH_CORE_H
#define VH_CORE_H

#include "common.h"


/// Used by mesh
struct Vertex {
  float3 pos;
  int    ref_count;

  __device__
  void Clear() {
    pos = make_float3(0.0);
    ref_count = 0;
  }
};

struct Triangle {
  int3 vertex_ptrs;

  __device__
  void Clear() {
    vertex_ptrs = make_int3(-1, -1, -1);
  }
};
#endif //VH_CORE_H
