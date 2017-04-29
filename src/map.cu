#include "matrix.h"

#include "hash_table.h"
#include "sensor_data.h"

#include <helper_cuda.h>
#include <helper_math.h>


// TODO(wei): put functions into hash_table.h
#define T_PER_BLOCK 8

#define PINF  __int_as_float(0x7f800000)

/// Input depth image as texture
/// Easier interpolation
extern texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
extern texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;

//////////
/// Compress hash entries
/// Private hash_entries.ptr, compacted_hash_entries
#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
__global__
void GenerateCompressedHashEntriesKernel(HashTableGPU<Block> hash_table, float4x4 c_T_w) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  __shared__ int localCounter;
  if (threadIdx.x == 0) localCounter = 0;
  __syncthreads();

  int addrLocal = -1;
  if (idx < hash_params.bucket_count * HASH_BUCKET_SIZE) {
    if (hash_table.hash_entries[idx].ptr != FREE_ENTRY) {
      if (IsBlockInCameraFrustum(c_T_w, hash_table.hash_entries[idx].pos))
      {
        addrLocal = atomicAdd(&localCounter, 1);
      }
    }
  }

  __syncthreads();

  __shared__ int addrGlobal;
  if (threadIdx.x == 0 && localCounter > 0) {
    addrGlobal = atomicAdd(hash_table.compacted_hash_entry_counter, localCounter);
  }
  __syncthreads();

  if (addrLocal != -1) {
    const unsigned int addr = addrGlobal + addrLocal;
    hash_table.compacted_hash_entries[addr] = hash_table.hash_entries[idx];
  }
}

unsigned int GenerateCompressedHashEntriesCudaHost(HashTableGPU<Block>& hash_table, const HashParams& hash_params,
                                                   float4x4 c_T_w) {
  const unsigned int threadsPerBlock = COMPACTIFY_HASH_THREADS_PER_BLOCK;
  const dim3 gridSize((HASH_BUCKET_SIZE * hash_params.bucket_count + threadsPerBlock - 1) / threadsPerBlock, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  checkCudaErrors(cudaMemset(hash_table.compacted_hash_entry_counter, 0, sizeof(int)));
  GenerateCompressedHashEntriesKernel << <gridSize, blockSize >> >(hash_table, c_T_w);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  unsigned int res = 0;
  checkCudaErrors(cudaMemcpy(&res, hash_table.compacted_hash_entry_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));

  return res;
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

__host__
void StarveOccupiedVoxelsCudaHost(HashTableGPU<Block>& hash_table, const HashParams& hash_params) {
  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table.compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 gridSize(occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (occupied_block_count > 0) {
    StarveOccupiedVoxelsKernel << <gridSize, blockSize >> >(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

//////////
/// Outlier blocks recycling
/// Private compacted_hash_entries, hash_entry_remove_flags
__shared__ float	shared_min_sdf   [SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint		shared_max_weight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];

__global__
void CollectInvalidBlockInfoKernel(HashTableGPU<Block> hash_table) {

  const unsigned int hashIdx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];


  Voxel v0 = hash_table.values[entry.ptr](2*threadIdx.x+0);
  Voxel v1 = hash_table.values[entry.ptr](2*threadIdx.x+1);

  if (v0.weight == 0)	v0.sdf = PINF;
  if (v1.weight == 0)	v1.sdf = PINF;

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

__host__
void CollectInvalidBlockInfoCudaHost(HashTableGPU<Block>& hash_table, const HashParams& hash_params) {
  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table.compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));
  const dim3 gridSize(occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (occupied_block_count > 0) {
    CollectInvalidBlockInfoKernel << <gridSize, blockSize >> >(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}


__global__
/// Private hash_entry_remove_flags
/// Public DeleteEntry && ClearVoxel
void RecycleInvalidBlockKernel(HashTableGPU<Block> hash_table) {

  //const uint hashIdx = blockIdx.x;
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

__host__
void RecycleInvalidBlockCudaHost(HashTableGPU<Block>& hash_table, const HashParams& hash_params) {
  const unsigned int threadsPerBlock = T_PER_BLOCK*T_PER_BLOCK;

  uint occupied_block_count;
  checkCudaErrors(cudaMemcpy(&occupied_block_count, hash_table.compacted_hash_entry_counter,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 gridSize((occupied_block_count + threadsPerBlock - 1) / threadsPerBlock, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (occupied_block_count > 0) {
    RecycleInvalidBlockKernel << <gridSize, blockSize >> >(hash_table);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}
