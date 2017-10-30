//
// Created by wei on 17-10-21.
//

#ifndef CORE_VOXEL_H
#define CORE_VOXEL_H

#include "core/common.h"
#include "helper_math.h"

// Statistics typically reserved for Voxels
// float: *Laplacian* and *entropy* are intuitive statistics
// float: *duration* is time-interval that the voxel exists
struct __ALIGN__(4) Stat {
  float laplacian;
  float entropy;
  float duration;

  __host__ __device__
  void Clear() {
    laplacian = 0;
    entropy = 0;
    duration = 0;
  }
};

//  + --------> //
//  | \         //
//  |  \        //
//  |   \       //
//  v　　 v      //
// A voxel holds:
// float: *sdf* and *weight* on its corner
// int[3]: *vertex_ptrs* on its 3 adjacent edges
// int[5]: *triangle_ptrs* for its potential triangles
struct __ALIGN__(8) Voxel {
  float  sdf;    // signed distance function
  float  weight;
  uchar3 color;  // color

  // !!!!!!
  // TODO: split them into mesh_unit, stats, additional variables, etc
  // !!!!!!
  // mesh
  int vertex_ptrs   [N_VERTEX];  // 3
  int vertex_mutexes[N_VERTEX];  // 3
  int triangle_ptrs [N_TRIANGLE];// 5

#ifdef STATS
  Stat   stats;
#endif
  short curr_cube_idx, prev_cube_idx;

//#ifdef PRIMAL_DUAL
  bool   mask;
  float  sdf0, sdf_bar;
  float3 p;
//#endif

  __host__ __device__
  void ResetMutexes() {
    vertex_mutexes[0] = FREE_PTR;
    vertex_mutexes[1] = FREE_PTR;
    vertex_mutexes[2] = FREE_PTR;
  }

  __host__ __device__
  int GetVertex(int idx) {
    return vertex_ptrs[idx];
  }

  __host__ __device__
  void Clear() {
    ClearSDF();
    ClearTriangle();
#ifdef STATS
    stats.Clear();
#endif
  }

  __host__ __device__
  void ClearSDF() {
    sdf = weight = 0.0f;
    color = make_uchar3(0, 0, 0);
    sdf0 = 0;
    sdf_bar = 0;
    p = make_float3(0);
    mask = false;
  }

  __host__ __device__
  void ClearTriangle() {
    vertex_ptrs[0] = vertex_mutexes[0] = FREE_PTR;
    vertex_ptrs[1] = vertex_mutexes[1] = FREE_PTR;
    vertex_ptrs[2] = vertex_mutexes[2] = FREE_PTR;

    triangle_ptrs[0] = FREE_PTR;
    triangle_ptrs[1] = FREE_PTR;
    triangle_ptrs[2] = FREE_PTR;
    triangle_ptrs[3] = FREE_PTR;
    triangle_ptrs[4] = FREE_PTR;

    curr_cube_idx = prev_cube_idx = 0;
  }

  __host__ __device__
  void Update(const Voxel &delta) {
    float3 c_prev  = make_float3(color.x, color.y, color.z);
    float3 c_delta = make_float3(delta.color.x, delta.color.y, delta.color.z);
    float3 c_curr  = 0.5f * c_prev + 0.5f * c_delta;
    color = make_uchar3(c_curr.x + 0.5f, c_curr.y + 0.5f, c_curr.z + 0.5f);

    sdf = (sdf * weight + delta.sdf * delta.weight) / (weight + delta.weight);
    weight = weight + delta.weight;
  }
};

#endif // CORE_VOXEL_H
