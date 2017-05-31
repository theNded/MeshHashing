#include "matrix.h"

#include "map.h"
#include "sensor.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <unordered_set>
#include <vector>
#include <list>
#include <glog/logging.h>
#include <device_launch_parameters.h>


#define PINF  __int_as_float(0x7f800000)

////////////////////
/// class Map - compress, recycle
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void StarveOccupiedBlocksKernel(CompactHashTableGPU compact_hash_table,
                                VoxelBlocksGPU      blocks) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = compact_hash_table.compacted_entries[idx];

  int weight = blocks[entry.ptr].voxels[threadIdx.x].weight;
  weight = max(0, weight-1);
  blocks[entry.ptr].voxels[threadIdx.x].weight = (uchar)weight;
}

/// Collect dead voxels
__global__
void CollectGarbageBlocksKernel(CompactHashTableGPU compact_hash_table,
                                VoxelBlocksGPU      blocks) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = compact_hash_table.compacted_entries[idx];

  Voxel v0 = blocks[entry.ptr].voxels[2*threadIdx.x+0];
  Voxel v1 = blocks[entry.ptr].voxels[2*threadIdx.x+1];

  if (v0.weight == 0)	v0.sdf = PINF;
  if (v1.weight == 0)	v1.sdf = PINF;

  __shared__ float	shared_min_sdf   [BLOCK_SIZE / 2];
  __shared__ uint		shared_max_weight[BLOCK_SIZE / 2];
  shared_min_sdf[threadIdx.x] = fminf(fabsf(v0.sdf), fabsf(v1.sdf));
  shared_max_weight[threadIdx.x] = max(v0.weight, v1.weight);

  /// reducing operation
#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {

    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_min_sdf[threadIdx.x] = fminf(shared_min_sdf[threadIdx.x-stride/2],
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

    compact_hash_table.entry_recycle_flags[idx] =
            (min_sdf >= t || max_weight == 0) ? 1 : 0;
  }
}

__global__
/// Their mesh not recycled
void RecycleGarbageBlocksKernel(HashTableGPU        hash_table,
                                CompactHashTableGPU compact_hash_table,
                                VoxelBlocksGPU      blocks) {
  const uint idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < (*compact_hash_table.compacted_entry_counter)
      && compact_hash_table.entry_recycle_flags[idx] != 0) {
    const HashEntry& entry = compact_hash_table.compacted_entries[idx];
    if (hash_table.DeleteEntry(entry.pos)) {
      blocks[entry.ptr].Clear();
    }
  }
}

/// Condition: IsBlockInCameraFrustum
__global__
void CollectInFrustumBlocksKernel(HashTableGPU        hash_table,
                                  CompactHashTableGPU compact_hash_table,
                                  SensorParams        sensor_params,
                                  float4x4            c_T_w) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count) {
    if (hash_table.entries[idx].ptr != FREE_ENTRY) {
      if (IsBlockInCameraFrustum(c_T_w, hash_table.entries[idx].pos,
                                 sensor_params)) {
        addr_local = atomicAdd(&local_counter, 1);
      }
    }
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_hash_table.compacted_entry_counter,
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    compact_hash_table.compacted_entries[addr] = hash_table.entries[idx];
  }
}

__global__
void CollectAllBlocksKernel(HashTableGPU        hash_table,
                            CompactHashTableGPU compact_hash_table) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count) {
    if (hash_table.entries[idx].ptr != FREE_ENTRY) {
      addr_local = atomicAdd(&local_counter, 1);
    }
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(compact_hash_table.compacted_entry_counter,
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    compact_hash_table.compacted_entries[addr] = hash_table.entries[idx];
  }
}

////////////////////
/// Host code
///////////////////

/// Life cycle
Map::Map(const HashParams &hash_params, const MeshParams &mesh_params) {
  hash_table_.Resize(hash_params);
  compact_hash_table_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);
  compact_mesh_.Resize(mesh_params);
}

Map::~Map() {}

/// Reset
void Map::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  compact_hash_table_.Reset();
  compact_mesh_.Reset();
}

/// Garbage collection
void Map::Recycle() {
  // TODO(wei): change it via global parameters
  bool kRecycle = true;
  int garbage_collect_starve = 15;

  if (kRecycle) {
    if (integrated_frame_count_ > 0
        && integrated_frame_count_ % garbage_collect_starve == 0) {
      StarveOccupiedBlocks();
    }
    CollectGarbageBlocks();
    hash_table_.ResetMutexes();
    RecycleGarbageBlocks();
  }
}

void Map::StarveOccupiedBlocks() {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = compact_hash_table_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  StarveOccupiedBlocksKernel<<<grid_size, block_size >>>(
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Map::CollectGarbageBlocks() {
  const uint threads_per_block = BLOCK_SIZE / 2;

  uint processing_block_count = compact_hash_table_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectGarbageBlocksKernel <<<grid_size, block_size >>>(
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Map::RecycleGarbageBlocks() {
  const uint threads_per_block = 64;

  uint processing_block_count = compact_hash_table_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size((processing_block_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  RecycleGarbageBlocksKernel <<<grid_size, block_size >>>(
          hash_table_.gpu_data(),
          compact_hash_table_.gpu_data(),
          blocks_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

/// Compress discrete hash table entries
void Map::CollectAllBlocks(){
  const uint threads_per_block = 256;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, hash_table_.gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  compact_hash_table_.reset_entry_count();
  CollectAllBlocksKernel<<<grid_size, block_size >>>(
          hash_table_.gpu_data(),
          compact_hash_table_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in all: "
            << compact_hash_table_.entry_count();
}

void Map::CollectInFrustumBlocks(Sensor &sensor){
  const uint threads_per_block = 256;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, hash_table_.gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  compact_hash_table_.reset_entry_count();
  CollectInFrustumBlocksKernel<<<grid_size, block_size >>>(
          hash_table_.gpu_data(),
          compact_hash_table_.gpu_data(),
          sensor.sensor_params(), sensor.c_T_w());

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in view frustum: "
            << compact_hash_table_.entry_count();
}

