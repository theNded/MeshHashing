#include "matrix.h"

#include "hash_table.h"
#include "sensor_data.h"

#include <helper_cuda.h>
#include <helper_math.h>

#define T_PER_BLOCK 8

#define PINF  __int_as_float(0x7f800000)

/// Input depth image as texture
/// Easier interpolation
extern texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
extern texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;

//////////
/// Reset cuda memory
__global__
void ResetBucketMutexesKernel(HashTable hash_table)
{
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count) {
    hash_table.bucket_mutexes[idx] = FREE_ENTRY;
  }
}

__host__
void ResetBucketMutexesCudaHost(HashTable& hash_table, const HashParams& hash_params)
{
  const dim3 gridSize((hash_params.bucket_count + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
  const dim3 blockSize((T_PER_BLOCK*T_PER_BLOCK), 1);

  ResetBucketMutexesKernel<<<gridSize, blockSize>>>(hash_table);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

__global__
void ResetHeapKernel(HashTable hash_table) {
  const HashParams& hash_params = kHashParams;
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx == 0) {
    hash_table.heap_counter[0] = hash_params.block_count - 1;	//points to the last element of the array
  }

  if (idx < hash_params.block_count) {

    hash_table.heap[idx] = hash_params.block_count - idx - 1;
    uint blockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;
    uint base_idx = idx * blockSize;
    for (uint i = 0; i < blockSize; i++) {
      hash_table.DeleteVoxel(base_idx+i);
    }
  }
}

__global__
void ResetEntriesKernel(HashTable hash_table)
{
  const HashParams& hash_params = kHashParams;
  const unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx < hash_params.bucket_count * HASH_BUCKET_SIZE) {
    hash_table.DeleteHashEntry(hash_table.hash_entries[idx]);
    hash_table.DeleteHashEntry(hash_table.compacted_hash_entries[idx]);
  }
}

__host__
void ResetCudaHost(HashTable& hash_table, const HashParams& hash_params) {
  {
    //resetting the heap and SDF blocks
    const dim3 gridSize((hash_params.block_count + (T_PER_BLOCK*T_PER_BLOCK) - 1)/(T_PER_BLOCK*T_PER_BLOCK), 1);
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
/// Alloc blocks in the frustum
__global__
void AllocBlocksKernel(HashTable hash_table, SensorData cameraData,
                                  float4x4 w_T_c, const unsigned int* d_bitMask)
{
  const HashParams& hash_params = kHashParams;
  const SensorParams& cameraParams = kSensorParams;

  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x < cameraParams.width && y < cameraParams.height)
  {
    /// TODO(wei): change it here, too ugly
    float d = tex2D(depthTextureRef, x, y);

    //if (d == MINF || d < cameraParams.min_depth_range || d > cameraParams.max_depth_range)	return;
    if (d == MINF || d == 0.0f)	return;

    if (d >= hash_params.sdf_upper_bound) return;

    float t = truncate_distance(d);
    float minDepth = min(hash_params.sdf_upper_bound, d-t);
    float maxDepth = min(hash_params.sdf_upper_bound, d+t);
    if (minDepth >= maxDepth) return;

    float3 rayMin = ImageReprojectToCamera(x, y, minDepth);
    rayMin = w_T_c * rayMin;
    float3 rayMax = ImageReprojectToCamera(x, y, maxDepth);
    rayMax = w_T_c * rayMax;


    float3 rayDir = normalize(rayMax - rayMin);

    int3 idCurrentVoxel = WorldToBlock(rayMin);
    int3 idEnd = WorldToBlock(rayMax);

    float3 step = make_float3(sign(rayDir));
    float3 boundaryPos = BlockToWorld(idCurrentVoxel+make_int3(clamp(step, 0.0, 1.0f)))-0.5f*hash_params.voxel_size;
    float3 tMax = (boundaryPos-rayMin)/rayDir;
    float3 tDelta = (step*SDF_BLOCK_SIZE*hash_params.voxel_size)/rayDir;
    int3 idBound = make_int3(make_float3(idEnd)+step);

    //#pragma unroll
    //for(int c = 0; c < 3; c++) {
    //	if (rayDir[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
    //	if (boundaryPos[c] - rayMin[c] == 0.0f) { tMax[c] = PINF; tDelta[c] = PINF; }
    //}
    if (rayDir.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }
    if (boundaryPos.x - rayMin.x == 0.0f) { tMax.x = PINF; tDelta.x = PINF; }

    if (rayDir.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }
    if (boundaryPos.y - rayMin.y == 0.0f) { tMax.y = PINF; tDelta.y = PINF; }

    if (rayDir.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }
    if (boundaryPos.z - rayMin.z == 0.0f) { tMax.z = PINF; tDelta.z = PINF; }


    unsigned int iter = 0; // iter < g_MaxLoopIterCount
    unsigned int g_MaxLoopIterCount = 1024;	//TODO MATTHIAS MOVE TO GLOBAL APP STATE
#pragma unroll 1
    while(iter < g_MaxLoopIterCount) {

      //check if it's in the frustum and not checked out
      if (IsBlockInCameraFrustum(w_T_c.getInverse(), idCurrentVoxel)) {
        /// Disable streaming at current
        // && !isSDFBlockStreamedOut(idCurrentVoxel, hash_table, d_bitMask)) {
        hash_table.AllocBlock(idCurrentVoxel);
      }

      // Traverse voxel grid
      if(tMax.x < tMax.y && tMax.x < tMax.z)	{
        idCurrentVoxel.x += step.x;
        if(idCurrentVoxel.x == idBound.x) return;
        tMax.x += tDelta.x;
      }
      else if(tMax.z < tMax.y) {
        idCurrentVoxel.z += step.z;
        if(idCurrentVoxel.z == idBound.z) return;
        tMax.z += tDelta.z;
      }
      else	{
        idCurrentVoxel.y += step.y;
        if(idCurrentVoxel.y == idBound.y) return;
        tMax.y += tDelta.y;
      }

      iter++;
    }
  }
}

__host__
void AllocBlocksCudaHost(HashTable& hash_table, const HashParams& hash_params,
               const SensorData& sensor_data, const SensorParams& depthCameraParams,
               const float4x4& w_T_c,
               const unsigned int* d_bitMask) {

  const dim3 gridSize((depthCameraParams.width + T_PER_BLOCK - 1)/T_PER_BLOCK, (depthCameraParams.height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  AllocBlocksKernel<<<gridSize, blockSize>>>(hash_table, sensor_data, w_T_c, d_bitMask);

#ifdef _DEBUG
  cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
}

//////////
/// Compress hash entries
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
  int weight = hash_table.blocks[entry.ptr + threadIdx.x].weight;
  weight = max(0, weight-1);
  hash_table.blocks[entry.ptr + threadIdx.x].weight = weight;
}

void StarveOccupiedVoxelsCudaHost(HashTable& hash_table, const HashParams& hash_params) {
  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
  const dim3 gridSize(hash_params.occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (hash_params.occupied_block_count > 0) {
    StarveOccupiedVoxelsKernel << <gridSize, blockSize >> >(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}

//////////
/// Outlier blocks recycling
__shared__ float	shared_MinSDF[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];
__shared__ uint		shared_MaxWeight[SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2];

__global__
void CollectInvalidBlockInfoKernel(HashTable hash_table) {

  const unsigned int hashIdx = blockIdx.x;
  const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];

  //uint h = hash_table.HashBucketForBlockPos(entry.pos);
  //hash_table.block_remove_flags[hashIdx] = 1;
  //if (hash_table.bucket_mutexes[h] == LOCK_ENTRY)	return;

  //if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before
  //const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

  const unsigned int idx0 = entry.ptr + 2*threadIdx.x+0;
  const unsigned int idx1 = entry.ptr + 2*threadIdx.x+1;

  Voxel v0 = hash_table.blocks[idx0];
  Voxel v1 = hash_table.blocks[idx1];

  if (v0.weight == 0)	v0.sdf = PINF;
  if (v1.weight == 0)	v1.sdf = PINF;

  shared_MinSDF[threadIdx.x] = min(fabsf(v0.sdf), fabsf(v1.sdf));	//init shared memory
  shared_MaxWeight[threadIdx.x] = max(v0.weight, v1.weight);

#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {
    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_MinSDF[threadIdx.x] = min(shared_MinSDF[threadIdx.x-stride/2], shared_MinSDF[threadIdx.x]);
      shared_MaxWeight[threadIdx.x] = max(shared_MaxWeight[threadIdx.x-stride/2], shared_MaxWeight[threadIdx.x]);
    }
  }

  __syncthreads();

  if (threadIdx.x == blockDim.x - 1) {
    float minSDF = shared_MinSDF[threadIdx.x];
    uint maxWeight = shared_MaxWeight[threadIdx.x];

    float t = truncate_distance(kSensorParams.max_depth_range);	//MATTHIAS TODO check whether this is a reasonable metric

    if (minSDF >= t || maxWeight == 0) {
      hash_table.block_remove_flags[hashIdx] = 1;
    } else {
      hash_table.block_remove_flags[hashIdx] = 0;
    }
  }
}

void CollectInvalidBlockInfoCudaHost(HashTable& hash_table, const HashParams& hash_params) {

  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE / 2;
  const dim3 gridSize(hash_params.occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (hash_params.occupied_block_count > 0) {
    CollectInvalidBlockInfoKernel << <gridSize, blockSize >> >(hash_table);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
  }
}


__global__
void RecycleInvalidBlockKernel(HashTable hash_table) {

  //const uint hashIdx = blockIdx.x;
  const uint hashIdx = blockIdx.x*blockDim.x + threadIdx.x;


  if (hashIdx < kHashParams.occupied_block_count && hash_table.block_remove_flags[hashIdx] != 0) {	//decision to delete the hash entry

    const HashEntry& entry = hash_table.compacted_hash_entries[hashIdx];
    //if (entry.ptr == FREE_ENTRY) return; //should never happen since we did compactify before

    if (hash_table.DeleteHashEntryElement(entry.pos)) {	//delete hash entry from hash (and performs heap append)
      const uint linBlockSize = SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE;

#pragma unroll 1
      for (uint i = 0; i < linBlockSize; i++) {	//clear sdf block: CHECK TODO another kernel?
        hash_table.DeleteVoxel(entry.ptr + i);
      }
    }
  }
}

void RecycleInvalidBlockCudaHost(HashTable& hash_table, const HashParams& hash_params) {

  const unsigned int threadsPerBlock = T_PER_BLOCK*T_PER_BLOCK;
  const dim3 gridSize((hash_params.occupied_block_count + threadsPerBlock - 1) / threadsPerBlock, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (hash_params.occupied_block_count > 0) {
    RecycleInvalidBlockKernel << <gridSize, blockSize >> >(hash_table);
  }
#ifdef _DEBUG
  cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

//////////////////////
/// For streaming usage
__device__
unsigned int linearizeChunkPos(const int3& chunkPos)
{
  int3 p = chunkPos-kHashParams.m_streamingMinGridPos;
  return  p.z * kHashParams.m_streamingGridDimensions.x * kHashParams.m_streamingGridDimensions.y +
          p.y * kHashParams.m_streamingGridDimensions.x +
          p.x;
}

__device__
int3 worldToChunks(const float3& posWorld)
{
  float3 p;
  p.x = posWorld.x/kHashParams.m_streamingVoxelExtents.x;
  p.y = posWorld.y/kHashParams.m_streamingVoxelExtents.y;
  p.z = posWorld.z/kHashParams.m_streamingVoxelExtents.z;

  float3 s;
  s.x = (float)sign(p.x);
  s.y = (float)sign(p.y);
  s.z = (float)sign(p.z);

  return make_int3(p+s*0.5f);
}

__device__
bool isSDFBlockStreamedOut(const int3& sdfBlock, const HashTable& hash_table, const unsigned int* d_bitMask)	//TODO MATTHIAS (-> move to HashTable)
{
  float3 posWorld = VoxelToWorld(BlockToVoxel(sdfBlock)); // sdfBlock is assigned to chunk by the bottom right sample pos

  uint index = linearizeChunkPos(worldToChunks(posWorld));
  uint nBitsInT = 32;
  return ((d_bitMask[index/nBitsInT] & (0x1 << (index%nBitsInT))) != 0x0);
}