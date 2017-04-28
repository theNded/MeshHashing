#include "matrix.h"

#include "hash_table_gpu.h"
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
/// Reset cuda memory:
/// Private bucket_mutexes
__global__
void ResetBucketMutexesKernel(HashTable hash_table) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count) {
    hash_table.bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__host__
void ResetBucketMutexesCudaHost(HashTable& hash_table, const HashParams& hash_params) {
  const dim3 gridSize((hash_params.bucket_count + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
  const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

  ResetBucketMutexesKernel<<<gridSize, blockSize>>>(hash_table);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

__global__
/// Private heap
/// Public ClearVoxel
void ResetHeapKernel(HashTable hash_table) {
  const HashParams& hash_params = kHashParams;
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx == 0) {
    hash_table.heap_counter[0] = hash_params.value_capacity - 1;	//points to the last element of the array
  }

  if (idx < hash_params.value_capacity) {
    hash_table.heap[idx] = hash_params.value_capacity - idx - 1;
    hash_table.values[idx].Clear();
  }
}

/// Public ClearHashEntry
/// Private hash_entries, compacted_hash_entries
__global__
void ResetEntriesKernel(HashTable hash_table) {
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count * HASH_BUCKET_SIZE) {
    hash_table.ClearHashEntry(hash_table.hash_entries[idx]);
    hash_table.ClearHashEntry(hash_table.compacted_hash_entries[idx]);
  }
}

__host__
void ResetCudaHost(HashTable& hash_table, const HashParams& hash_params) {
  {
    //resetting the heap and SDF blocks
    const dim3 gridSize((hash_params.value_capacity + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
    const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

    ResetHeapKernel<<<gridSize, blockSize>>>(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    //resetting the hash
    const dim3 gridSize((HASH_BUCKET_SIZE * hash_params.bucket_count + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
    const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

    ResetEntriesKernel<<<gridSize, blockSize>>>(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }

  {
    //resetting the mutex
    const dim3 gridSize((hash_params.bucket_count + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
    const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

    ResetBucketMutexesKernel<<<gridSize, blockSize>>>(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

//////////
/// Compress hash entries
/// Private hash_entries.ptr, compacted_hash_entries
#define COMPACTIFY_HASH_THREADS_PER_BLOCK 256
__global__
void GenerateCompressedHashEntriesKernel(HashTable hash_table, float4x4 c_T_w) {
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

unsigned int GenerateCompressedHashEntriesCudaHost(HashTable& hash_table, const HashParams& hash_params,
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
void StarveOccupiedVoxelsKernel(HashTable hash_table) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[idx];

  //is typically exectued only every n'th frame
  int weight = hash_table.values[entry.ptr](threadIdx.x).weight;
  weight = max(0, weight-1);
  hash_table.values[entry.ptr](threadIdx.x).weight = weight;
}

__host__
void StarveOccupiedVoxelsCudaHost(HashTable& hash_table, const HashParams& hash_params) {
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
/// Private compacted_hash_entries, value_remove_flags
__shared__ float	shared_min_sdf   [SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint		shared_max_weight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];

__global__
void CollectInvalidBlockInfoKernel(HashTable hash_table) {

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
      hash_table.value_remove_flags[hashIdx] = 1;
    } else {
      hash_table.value_remove_flags[hashIdx] = 0;
    }
  }
}

__host__
void CollectInvalidBlockInfoCudaHost(HashTable& hash_table, const HashParams& hash_params) {
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
/// Private value_remove_flags
/// Public DeleteEntry && ClearVoxel
void RecycleInvalidBlockKernel(HashTable hash_table) {

  //const uint hashIdx = blockIdx.x;
  const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;


  if (hashIdx < (*hash_table.compacted_hash_entry_counter)
      && hash_table.value_remove_flags[hashIdx] != 0) {	//decision to delete the hash entry

    const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];
    //if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

    if (hash_table.DeleteEntry(entry.pos)) {	//delete hash entry from hash (and performs heap append)
      hash_table.values[entry.ptr].Clear();
    }
  }
}

__host__
void RecycleInvalidBlockCudaHost(HashTable& hash_table, const HashParams& hash_params) {
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

//////////////////////
/// For streaming usage
//__device__
//unsigned int linearizeChunkPos(const int3& chunkPos)
//{
//  int3 p = chunkPos-kHashParams.m_streamingMinGridPos;
//  return  p.z * kHashParams.m_streamingGridDimensions.x * kHashParams.m_streamingGridDimensions.y +
//          p.y * kHashParams.m_streamingGridDimensions.x +
//          p.x;
//}
//
//__device__
//int3 worldToChunks(const float3& posWorld)
//{
//  float3 p;
//  p.x = posWorld.x/kHashParams.m_streamingVoxelExtents.x;
//  p.y = posWorld.y/kHashParams.m_streamingVoxelExtents.y;
//  p.z = posWorld.z/kHashParams.m_streamingVoxelExtents.z;
//
//  float3 s;
//  s.x = (float)sign(p.x);
//  s.y = (float)sign(p.y);
//  s.z = (float)sign(p.z);
//
//  return make_int3(p+s*0.5f);
//}
//
//__device__
//bool isSDFBlockStreamedOut(const int3& sdfBlock, const HashTable& hash_table, const unsigned int* is_streamed_mask)	//TODO MATTHIAS (-> move to HashTable)
//{
//  float3 posWorld = VoxelToWorld(BlockToVoxel(sdfBlock)); // sdfBlock is assigned to chunk by the bottom right sample pos
//
//  uint index = linearizeChunkPos(worldToChunks(posWorld));
//  uint nBitsInT = 32;
//  return ((is_streamed_mask[index/nBitsInT] & (0x1 << (index%nBitsInT))) != 0x0);
//}
