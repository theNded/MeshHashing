//
// Created by wei on 17-10-26.
//

#include "primal_dual.h"
#include <device_launch_parameters.h>
#include <glog/logging.h>
#include "geometry/primal_dual_query.h"

__global__
void PrimalDualInitKernel(
    EntryArray candidate_entries,
    BlockArray blocks
) {
  const HashEntry& entry = candidate_entries[blockIdx.x];
  Voxel& voxel = blocks[entry.ptr].voxels[threadIdx.x];

  // primal
  voxel.x0 = voxel.sdf;
  voxel.x_bar = 0;
  voxel.p = make_float3(0);
}

__global__
/**
 * Primal dual: dual step
 * @param candidate_entries
 * @param blocks
 * @param hash_table
 * @param geometry_helper
 * @param lambda
 * @param sigma
 * @param tau
 */
void PrimalDualIteratePass1Kernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper,
    float lambda,
    float sigma,
    float tau
) {
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Voxel &voxel = blocks[entry.ptr].voxels[threadIdx.x];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  if (voxel.weight < EPSILON) return;

  //const int kMaxIterations = 100;
  //for (int i = 0; i < kMaxIterations; ++i) {
  // Dual step
  // p_{n+1} = prox_F* (p_{n} + \sigma \nabla x_bar{n})
  // prox_F* (y) = \delta (y) (projection function)
  float3 primal_gradient;
  bool valid = GetPrimalGradient(
      entry, voxel_pos,
      blocks, hash_table,
      geometry_helper, &primal_gradient);
  if (! valid) {
    voxel.p = make_float3(0);
    return;
  }
  voxel.p = voxel.p + sigma * primal_gradient;
  // huber
  float alpha = 0.5;
  voxel.p /= (1 + sigma * alpha);
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
  const HashEntry &entry = candidate_entries[blockIdx.x];
  Voxel &voxel = blocks[entry.ptr].voxels[threadIdx.x];

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint local_idx = threadIdx.x;  //inside of an SDF block
  int3 voxel_pos = voxel_base_pos + make_int3(geometry_helper.DevectorizeIndex(local_idx));

  if (voxel.weight < EPSILON) return;
  float voxel_prev_x = voxel.sdf;
    // Primal step
    // x_{n+1} = prox_G  (x_{n} - \tau Div p_{n+1})
  // prox_G = (1 + \lambda y) / (1 + \lambda)
  float dual_divergence = voxel.p.x + voxel.p.y + voxel.p.z;
  if (fabsf(voxel.p.x) < EPSILON
      && fabsf(voxel.p.y) < EPSILON
      && fabsf(voxel.p.z) < EPSILON)
    return;
  voxel.sdf = voxel.sdf - tau * dual_divergence;
  voxel.sdf = (voxel.sdf + lambda * tau * voxel.x0) / (1 + lambda * tau);
  //printf("sdf0: %f prev: %f curr: %f\n", voxel.x0, voxel_prev_x, voxel.sdf);
  // Extrapolation
  voxel.x_bar = 2 * voxel.sdf - voxel_prev_x;
  //}
}

void PrimalDualInit(
    EntryArray& candidate_entries,
    BlockArray& blocks
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return;

  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  PrimalDualInitKernel<<<grid_size, block_size>>> (candidate_entries, blocks);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void PrimalDualIterate(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    HashTable& hash_table,
    GeometryHelper& geometry_helper,
    const float lambda,
    const float sigma,
    const float tau
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return;

  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  PrimalDualIteratePass1Kernel <<<grid_size, block_size>>> (
    candidate_entries,
        blocks, hash_table,
        geometry_helper,
        lambda, sigma, tau);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
  PrimalDualIteratePass2Kernel <<<grid_size, block_size>>> (
      candidate_entries,
          blocks, hash_table,
          geometry_helper,
          lambda, sigma, tau);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}