#include "matrix.h"

#include "map.h"
#include "sensor.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <unordered_set>
#include <vector>
#include <list>
#include <glog/logging.h>

#define PINF  __int_as_float(0x7f800000)

//////////
/// Kernel functions
/// Starve voxels (to determine outliers)
__global__
void StarveOccupiedVoxelsKernel(HashTableGPU<VoxelBlock> hash_table) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[idx];

  int weight = hash_table.values[entry.ptr](threadIdx.x).weight;
  weight = max(0, weight-1);
  hash_table.values[entry.ptr](threadIdx.x).weight = weight;
}

/// Collect dead voxels
__global__
void CollectInvalidBlockInfoKernel(HashTableGPU<VoxelBlock> hash_table) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[idx];

  Voxel v0 = hash_table.values[entry.ptr](2*threadIdx.x+0);
  Voxel v1 = hash_table.values[entry.ptr](2*threadIdx.x+1);

  if (v0.weight == 0)	v0.sdf = PINF;
  if (v1.weight == 0)	v1.sdf = PINF;

  __shared__ float	shared_min_sdf   [BLOCK_SIZE / 2];
  __shared__ uint		shared_max_weight[BLOCK_SIZE / 2];
  shared_min_sdf[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
  shared_max_weight[threadIdx.x] = max(v0.weight, v1.weight);

  /// reducing operation (?)
#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_min_sdf[threadIdx.x] = min(shared_min_sdf[threadIdx.x-stride/2],
                                        shared_min_sdf[threadIdx.x]);
      shared_max_weight[threadIdx.x] = max(shared_max_weight[threadIdx.x-stride/2],
                                           shared_max_weight[threadIdx.x]);
    }
  }
  __syncthreads();

  if (threadIdx.x == blockDim.x - 1) {
    float min_sdf = shared_min_sdf[threadIdx.x];
    uint max_weight = shared_max_weight[threadIdx.x];

    // TODO(wei): check this weird reference
    float t = truncate_distance(5.0f);

    hash_table.hash_entry_remove_flags[idx] =
            (min_sdf >= t || max_weight == 0) ? 1 : 0;
  }
}

__global__
void RecycleInvalidBlockKernel(HashTableGPU<VoxelBlock> hash_table) {
  const uint idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < (*hash_table.compacted_hash_entry_counter)
      && hash_table.hash_entry_remove_flags[idx] != 0) {
    const HashEntry& entry = hash_table.compacted_hash_entries[idx];
    if (hash_table.DeleteEntry(entry.pos)) {
      hash_table.values[entry.ptr].Clear();
    }
  }
}

__global__
void CollectAllBlocksKernel(HashTableGPU<VoxelBlock> hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count) {
    if (hash_table.hash_entries[idx].ptr != FREE_ENTRY) {
      addr_local = atomicAdd(&local_counter, 1);
    }
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(hash_table.compacted_hash_entry_counter,
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    hash_table.compacted_hash_entries[addr] = hash_table.hash_entries[idx];
  }
}

//////////
/// Member functions (CPU code)
Map::Map(const HashParams &hash_params) {
  hash_table_.Resize(hash_params);

  Reset();
}

Map::~Map() {}

void Map::Reset() {
  integrated_frame_count_ = 0;
  hash_table_.Reset();
}

void Map::Recycle() {
  // TODO(wei): change it via global parameters
  bool kRecycle = true;
  int garbage_collect_starve = 15;
  if (kRecycle) {
    if (integrated_frame_count_ > 0
        && integrated_frame_count_ % garbage_collect_starve == 0) {
      StarveOccupiedVoxels();
    }
    CollectInvalidBlockInfo();
    hash_table_.ResetMutexes();
    RecycleInvalidBlock();
  }
}

/// Member functions (CPU calling GPU kernels)
/// (__host__)
void Map::StarveOccupiedVoxels() {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count;
  checkCudaErrors(cudaMemcpy(&processing_block_count,
                             gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  StarveOccupiedVoxelsKernel<<<grid_size, block_size >>>(gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// (__host__)
void Map::CollectInvalidBlockInfo() {
  const uint threads_per_block = BLOCK_SIZE / 2;

  uint processing_block_count;
  checkCudaErrors(cudaMemcpy(&processing_block_count,
                             gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectInvalidBlockInfoKernel <<<grid_size, block_size >>>(gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// (__host__)
void Map::RecycleInvalidBlock() {
  const uint threads_per_block = 64;

  uint processing_block_count;
  checkCudaErrors(cudaMemcpy(&processing_block_count,
                             gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size((processing_block_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  RecycleInvalidBlockKernel << <grid_size, block_size >> >(gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Map::CollectAllBlocks(){
  const uint threads_per_block = 256;
  uint res = 0;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  checkCudaErrors(cudaMemset(gpu_data().compacted_hash_entry_counter,
                             0, sizeof(int)));
  CollectAllBlocksKernel<<<grid_size, block_size >>>(gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMemcpy(&res, gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  LOG(INFO) << "Block count in all: " << res;
}
