//
// Created by wei on 17-10-26.
//

#include "primal_dual.h"
#include <device_launch_parameters.h>
#include "geometry/primal_dual_query.h"

__global__
void PrimalDualIteratePass1Kernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    float lambda,
    float sigma,
    float tau
) {
  const HashEntry& entry = candidate_entries[blockIdx.x];
  Voxel &voxel = blocks[entry.ptr].voxels[threadIdx.x];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.IdxToVoxelLocalPos(local_idx));

  // Pass 1
  // p_{n+1} = \delta (p_n + \sigma \nabla \bar{x_n})
  float3 primal_gradient;
  float dual_divergence;
  bool valid = GetPrimalGradientDualDivergence(
      entry, voxel_pos,
      blocks,
      hash_table,
      geometry_helper,
      primal_gradient,
      dual_divergence
  );
  if (! valid) return;

  voxel.p = voxel.p + sigma * primal_gradient;
  voxel.p = voxel.p / fmaxf(1, length(voxel.p));
}

__global__
void PrimalDualIteratePass2Kernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    float lambda,
    float sigma,
    float tau
) {
  const HashEntry& entry = candidate_entries[blockIdx.x];
  Voxel &voxel = blocks[entry.ptr].voxels[threadIdx.x];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.IdxToVoxelLocalPos(local_idx));

  // Pass 2: should be separated in another function
  // x_{n+1} = prox (x_{n} - \tau (\nabla^T) p_{n+1})
  float3 primal_gradient;
  float dual_divergence;
  bool valid = GetPrimalGradientDualDivergence(
      entry, voxel_pos,
      blocks,
      hash_table,
      geometry_helper,
      primal_gradient,
      dual_divergence
  );
  float voxel_prev_x = voxel.x;
  voxel.x = voxel.x - tau * dual_divergence;
  voxel.x = (voxel.x + lambda * sigma * voxel.sdf) / (1 + lambda);

  voxel.x_bar = 2 * voxel.x - voxel_prev_x;
}

