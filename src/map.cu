#include "matrix.h"

#include "map.h"
#include "sensor.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <unordered_set>
#include <vector>
#include <list>

// TODO(wei): put functions into hash_table.h
#define T_PER_BLOCK 8

#define PINF  __int_as_float(0x7f800000)

/// Input depth image as texture
/// Easier interpolation
/// Kernel functions
__global__
void CompactHashEntriesKernel(HashTableGPU<Block> hash_table, float4x4 c_T_w) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < hash_params.bucket_count * HASH_BUCKET_SIZE) {
    if (hash_table.hash_entries[idx].ptr != FREE_ENTRY) {
      if (IsBlockInCameraFrustum(c_T_w, hash_table.hash_entries[idx].pos))
      {
        addr_local = atomicAdd(&local_counter, 1);
      }
    }
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(hash_table.compacted_hash_entry_counter, local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const unsigned int addr = addr_global + addr_local;
    hash_table.compacted_hash_entries[addr] = hash_table.hash_entries[idx];
  }
}


//////////
/// Starve voxels (to determine outliers)
__global__
void StarveOccupiedVoxelsKernel(HashTableGPU<Block> hash_table) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[idx];

  //is typically exectued only every n'th frame
  int weight = hash_table.values[entry.ptr](threadIdx.x).weight;
  weight = max(0, weight-1);
  hash_table.values[entry.ptr](threadIdx.x).weight = weight;
}

__global__
void CollectInvalidBlockInfoKernel(HashTableGPU<Block> hash_table) {

  const unsigned int hashIdx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];

  Voxel v0 = hash_table.values[entry.ptr](2*threadIdx.x+0);
  Voxel v1 = hash_table.values[entry.ptr](2*threadIdx.x+1);

  if (v0.weight == 0)	v0.sdf = PINF;
  if (v1.weight == 0)	v1.sdf = PINF;

  __shared__ float	shared_min_sdf   [SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
  __shared__ uint		shared_max_weight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
  shared_min_sdf[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
  shared_max_weight[threadIdx.x] = max(v0.weight, v1.weight);

#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_min_sdf[threadIdx.x] = min(shared_min_sdf[threadIdx.x-stride/2], shared_min_sdf[threadIdx.x]);
      shared_max_weight[threadIdx.x] = max(shared_max_weight[threadIdx.x-stride/2], shared_max_weight[threadIdx.x]);
    }
  }

  __syncthreads();

  if (threadIdx.x == blockDim.x - 1) {
    float minSDF = shared_min_sdf[threadIdx.x];
    uint maxWeight = shared_max_weight[threadIdx.x];

    float t = truncate_distance(kSensorParams.max_depth_range);	//MATTHIAS TODO check whether this is a reasonable metric

    if (minSDF >= t || maxWeight == 0) {
      hash_table.hash_entry_remove_flags[hashIdx] = 1;
    } else {
      hash_table.hash_entry_remove_flags[hashIdx] = 0;
    }
  }
}

__global__
void RecycleInvalidBlockKernel(HashTableGPU<Block> hash_table) {
  const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;

  if (hashIdx < (*hash_table.compacted_hash_entry_counter)
      && hash_table.hash_entry_remove_flags[hashIdx] != 0) {	//decision to delete the hash entry

    const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];
    //if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

    if (hash_table.DeleteEntry(entry.pos)) {	//delete hash entry from hash (and performs heap append)
      hash_table.values[entry.ptr].Clear();
    }
  }
}

/// Member functions (pure CPU interface)
Map::Map(const HashParams &hash_params) {
  hash_params_ = hash_params;
  hash_table_.Resize(hash_params_);

  Reset();
}

Map::~Map() {}

void Map::Reset() {
  integrated_frame_count_ = 0;
  occupied_block_count_ = 0;
  hash_table_.Reset();
}

void Map::Recycle() {
  bool kRecycle = true;         /// false
  int garbage_collect_starve = 15;      /// 15
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

/// Member functions (kernel function callers)
void Map::CompactHashEntries(float4x4 c_T_w){
  const unsigned int threads_per_block = 256;
  const dim3 grid_size((HASH_BUCKET_SIZE * hash_params_.bucket_count + threads_per_block - 1) / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  checkCudaErrors(cudaMemset(hash_table_.gpu_data().compacted_hash_entry_counter, 0, sizeof(int)));
  CompactHashEntriesKernel<< <grid_size, block_size >> >(hash_table_.gpu_data(), c_T_w);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  unsigned int res = 0;
  checkCudaErrors(cudaMemcpy(&res, hash_table_.gpu_data().compacted_hash_entry_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
}

/// (__host__)
void Map::StarveOccupiedVoxels() {
  const unsigned int threads_per_block = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table_.gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  if (occupied_block_count > 0) {
    StarveOccupiedVoxelsKernel << <grid_size, block_size >> >(hash_table_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

/// (__host__)
void Map::CollectInvalidBlockInfo() {
  const unsigned int threads_per_block = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table_.gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 grid_size(occupied_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  if (occupied_block_count > 0) {
    CollectInvalidBlockInfoKernel <<<grid_size, block_size >>>(hash_table_.gpu_data());
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

/// (__host__)
void Map::RecycleInvalidBlock() {
  const unsigned int threads_per_block = 64;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table_.gpu_data().compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((occupied_block_count + threads_per_block - 1) / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  if (occupied_block_count > 0) {
    RecycleInvalidBlockKernel << <grid_size, block_size >> >(hash_table_.gpu_data());
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
