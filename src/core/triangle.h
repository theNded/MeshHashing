//
// Created by wei on 17-10-21.
//

#ifndef CORE_TRIANGLE_H
#define CORE_TRIANGLE_H

#include "core/common.h"
#include "helper_math.h"

struct __ALIGN__(4) Triangle {
  int3 vertex_ptrs;

  __device__
  void Clear() {
    vertex_ptrs = make_int3(-1, -1, -1);
  }
};

#endif //MESH_HASHING_TRIANGLE_H
