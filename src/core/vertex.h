//
// Created by wei on 17-10-21.
//

#ifndef CORE_VERTEX_H
#define CORE_VERTEX_H

#include "core/common.h"
#include "helper_math.h"

struct __ALIGN__(4) Vertex {
  float3 pos;
  float3 normal;
  float3 color;
  float  radius;
  int    ref_count;

  __device__
  void Clear() {
    pos = make_float3(0.0);
    normal = make_float3(0.0);
    color = make_float3(0);
    radius = 0;
    ref_count = 0;
  }
};

#endif //MESH_HASHING_VERTEX_H
