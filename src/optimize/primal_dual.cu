//
// Created by wei on 17-10-26.
//

#include "primal_dual.h"
#include <device_launch_parameters.h>
#include <glog/logging.h>
#include "geometry/primal_dual_query.h"

__device__
inline float Huber(float x, float alpha) {
  return (fabsf(x) < alpha) ? 0.5f * (x*x)/alpha : (fabsf(x) - 0.5f*alpha);
}

__global__
void PrimalDualInitKernel(
    EntryArray candidate_entries,
    BlockArray blocks,
    HashTable hash_table,
    GeometryHelper geometry_helper
) {
  const HashEntry& entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];
  PrimalDualVariables& primal_dual_variables
      = block.primal_dual_variables[threadIdx.x];
  Voxel& voxel = block.voxels[threadIdx.x];

  // primal
  primal_dual_variables.Clear();
  primal_dual_variables.sdf0 = voxel.sdf;
  primal_dual_variables.mask = true;
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
    float tau,
    float* err_data,
    float* err_tv
) {
  const float alpha = 0.02;

  const HashEntry &entry = candidate_entries[blockIdx.x];
  Block& block = blocks[entry.ptr];

  Voxel &voxel = block.voxels[threadIdx.x];
  PrimalDualVariables &primal_dual_variable = block.primal_dual_variables[threadIdx.x];
  if (voxel.weight < EPSILON) return;

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3 voxel_pos = voxel_base_pos + make_int3(offset);

  // Compute error
  float data_diff = fabsf(voxel.sdf - primal_dual_variable.sdf0);
  data_diff *= data_diff;
  if (voxel.weight > EPSILON) {
    atomicAdd(err_data, data_diff);
  }

  float3 gradient;
  GetSDFGradient(entry, voxel_pos,
                 blocks, hash_table,
                 geometry_helper, &gradient);
  atomicAdd(err_tv, Huber(length(gradient), alpha));

  // Dual step
  // p_{n+1} = prox_F* (p_{n} + \sigma \nabla x_bar{n})
  // prox_F* (y) = \delta (y) (projection function)
  float3 primal_gradient;
  GetPrimalGradient(entry, voxel_pos,
                    blocks, hash_table,
                    geometry_helper,
                    &primal_gradient);

  //float tv_diff =
  primal_dual_variable.p = primal_dual_variable.p + sigma * primal_gradient;
  // huber
  primal_dual_variable.p /= (1 + sigma * alpha);
  primal_dual_variable.p /= fmaxf(1, length(primal_dual_variable.p));
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
  Block& block = blocks[entry.ptr];
  Voxel &voxel = block.voxels[threadIdx.x];
  PrimalDualVariables& primal_dual_variables = block.primal_dual_variables[threadIdx.x];
  if (voxel.weight < EPSILON)
    return;

  int3 voxel_base_pos = geometry_helper.BlockToVoxel(entry.pos);
  uint3 offset = geometry_helper.DevectorizeIndex(threadIdx.x);
  int3 voxel_pos = voxel_base_pos + make_int3(offset);

  float voxel_sdf_prev = voxel.sdf;
  // Primal step
  // x_{n+1} = prox_G  (x_{n} - \tau -Div p_{n+1})
  // prox_G = (1 + \lambda y) / (1 + \lambda)
  float dual_divergence = 0;
  GetDualDivergence(entry, voxel_pos,
                    blocks, hash_table,
                    geometry_helper, &dual_divergence);
  voxel.sdf = voxel.sdf - tau * dual_divergence;
  voxel.sdf = (voxel.sdf + lambda * tau * primal_dual_variables.sdf0)
              / (1 + lambda * tau);
  // Extrapolation
  primal_dual_variables.sdf_bar = 2 * voxel.sdf - voxel_sdf_prev;
}

void PrimalDualInit(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    HashTable& hash_table,
    GeometryHelper& geometry_helper
) {
  const uint threads_per_block = BLOCK_SIZE;

  uint candidate_entry_count = candidate_entries.count();
  if (candidate_entry_count <= 0)
    return;

  const dim3 grid_size(candidate_entry_count, 1);
  const dim3 block_size(threads_per_block, 1);

  PrimalDualInitKernel<<<grid_size, block_size>>> (candidate_entries,
      blocks, hash_table, geometry_helper);
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

  float* err_data, *err_tv;
  checkCudaErrors(cudaMalloc(&err_data, sizeof(float)));
  checkCudaErrors(cudaMemset(err_data, 0, sizeof(float)));
  checkCudaErrors(cudaMalloc(&err_tv, sizeof(float)));
  checkCudaErrors(cudaMemset(err_tv, 0, sizeof(float)));

  PrimalDualIteratePass1Kernel <<<grid_size, block_size>>> (
    candidate_entries,
        blocks, hash_table,
        geometry_helper,
        lambda, sigma, tau,
        err_data, err_tv);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  float err_data_cpu, err_tv_cpu;
  checkCudaErrors(cudaMemcpy(&err_data_cpu, err_data, sizeof(float),
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(&err_tv_cpu, err_tv, sizeof(float),
                             cudaMemcpyDeviceToHost));
  std::cout << err_data_cpu * lambda / 2 + err_tv_cpu << " "
            << err_data_cpu << " "
            << err_tv_cpu << std::endl;

  PrimalDualIteratePass2Kernel <<<grid_size, block_size>>> (
      candidate_entries,
          blocks, hash_table,
          geometry_helper,
          lambda, sigma, tau);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}