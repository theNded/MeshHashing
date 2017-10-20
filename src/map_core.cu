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
#include "mc_tables.h"


#define PINF  __int_as_float(0x7f800000)

////////////////////
/// class Map - compress, recycle
////////////////////

////////////////////
/// Device code
////////////////////
__global__
void StarveOccupiedBlocksKernel(CandidateEntryPoolGPU candidate_entries,
                                BlocksGPU      blocks) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = candidate_entries.entries[idx];
  float weight = blocks[entry.ptr].voxels[threadIdx.x].weight;
  weight = fmaxf(0, weight - 1);
  blocks[entry.ptr].voxels[threadIdx.x].weight = weight;
}

/// Collect dead voxels
__global__
void CollectGarbageBlocksKernel(CandidateEntryPoolGPU candidate_entries,
                                BlocksGPU      blocks,
                                CoordinateConverter converter) {

  const uint idx = blockIdx.x;
  const HashEntry& entry = candidate_entries.entries[idx];

  Voxel v0 = blocks[entry.ptr].voxels[2*threadIdx.x+0];
  Voxel v1 = blocks[entry.ptr].voxels[2*threadIdx.x+1];

  float sdf0 = v0.sdf, sdf1 = v1.sdf;
  if (v0.weight < EPSILON)	sdf0 = PINF;
  if (v1.weight < EPSILON)	sdf1 = PINF;

  __shared__ float	shared_min_sdf   [BLOCK_SIZE / 2];
  __shared__ float	shared_max_weight[BLOCK_SIZE / 2];
  shared_min_sdf[threadIdx.x] = fminf(fabsf(sdf0), fabsf(sdf1));
  shared_max_weight[threadIdx.x] = fmaxf(v0.weight, v1.weight);

  /// reducing operation
#pragma unroll 1
  for (uint stride = 2; stride <= blockDim.x; stride <<= 1) {

    __syncthreads();
    if ((threadIdx.x  & (stride-1)) == (stride-1)) {
      shared_min_sdf[threadIdx.x] = fminf(shared_min_sdf[threadIdx.x-stride/2],
                                          shared_min_sdf[threadIdx.x]);
      shared_max_weight[threadIdx.x] = fmaxf(shared_max_weight[threadIdx.x-stride/2],
                                             shared_max_weight[threadIdx.x]);
    }
  }
  __syncthreads();

  if (threadIdx.x == blockDim.x - 1) {
    float min_sdf = shared_min_sdf[threadIdx.x];
    float max_weight = shared_max_weight[threadIdx.x];

    // TODO(wei): check this weird reference
    float t = converter.truncate_distance(5.0f);

    // TODO(wei): add || valid_triangles == 0 when memory leak is dealt with
    candidate_entries.entry_recycle_flags[idx] =
            (min_sdf >= t || max_weight < EPSILON) ? 1 : 0;

  }
}

/// !!! Their mesh not recycled
__global__
void RecycleGarbageBlocksTrianglesKernel(HashTableGPU        hash_table,
                                CandidateEntryPoolGPU candidate_entries,
                                BlocksGPU           blocks,
                                MeshGPU             mesh) {
  const uint idx = blockIdx.x;
  if (candidate_entries.entry_recycle_flags[idx] == 0) return;

  const HashEntry& entry = candidate_entries.entries[idx];
  const uint local_idx = threadIdx.x;  //inside an SDF block
  Voxel &voxel = blocks[entry.ptr].voxels[local_idx];

  for (int i = 0; i < N_TRIANGLE; ++i) {
    int triangle_ptr = voxel.triangle_ptrs[i];
    if (triangle_ptr == FREE_PTR) continue;

    // Clear ref_count of its pointed vertices
    mesh.ReleaseTriangle(mesh.triangles[triangle_ptr]);
    mesh.triangles[triangle_ptr].Clear();
    mesh.FreeTriangle(triangle_ptr);
    voxel.triangle_ptrs[i] = FREE_PTR;
  }
}

__global__
void RecycleGarbageBlocksVerticesKernel(HashTableGPU        hash_table,
                                         CandidateEntryPoolGPU candidate_entries,
                                         BlocksGPU           blocks,
                                         MeshGPU             mesh) {
  if (candidate_entries.entry_recycle_flags[blockIdx.x] == 0) return;
  const HashEntry &entry = candidate_entries.entries[blockIdx.x];
  const uint local_idx = threadIdx.x;

  Voxel &cube = blocks[entry.ptr].voxels[local_idx];

  __shared__ int valid_vertex_count;
  if (threadIdx.x == 0) valid_vertex_count = 0;
  __syncthreads();

#pragma unroll 1
  for (int i = 0; i < 3; ++i) {
    if (cube.vertex_ptrs[i] != FREE_PTR) {
      if (mesh.vertices[cube.vertex_ptrs[i]].ref_count <= 0) {
        mesh.vertices[cube.vertex_ptrs[i]].Clear();
        mesh.FreeVertex(cube.vertex_ptrs[i]);
        cube.vertex_ptrs[i] = FREE_PTR;
      }
      else {
        atomicAdd(&valid_vertex_count, 1);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0 && valid_vertex_count == 0) {
    if (hash_table.FreeEntry(entry.pos)) {
      blocks[entry.ptr].Clear();
    }
  }
}

/// Condition: IsBlockInCameraFrustum
__global__
void CollectInFrustumBlocksKernel(HashTableGPU        hash_table,
                                  CandidateEntryPoolGPU candidate_entries,
                                  SensorParams        sensor_params,
                                  float4x4            c_T_w,
                                  CoordinateConverter converter) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count
    && hash_table.entries[idx].ptr != FREE_ENTRY
    && converter.IsBlockInCameraFrustum(c_T_w, hash_table.entries[idx].pos,
                                        sensor_params)) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(candidate_entries.candidate_entry_counter,
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    candidate_entries.entries[addr] = hash_table.entries[idx];
  }
}

__global__
void CollectAllBlocksKernel(HashTableGPU        hash_table,
                            CandidateEntryPoolGPU candidate_entries) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < *hash_table.entry_count
      && hash_table.entries[idx].ptr != FREE_ENTRY) {
    addr_local = atomicAdd(&local_counter, 1);
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(candidate_entries.candidate_entry_counter,
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    candidate_entries.entries[addr] = hash_table.entries[idx];
  }
}

////////////////////
/// Host code
///////////////////

/// Life cycle
Map::Map(const HashParams &hash_params,
         const MeshParams &mesh_params,
         const SDFParams &sdf_params,
         const std::string& time_profile,
         const std::string& memo_profile) {
  hash_table_.Resize(hash_params);
  candidate_entries_.Resize(hash_params.entry_count);
  blocks_.Resize(hash_params.value_capacity);

  mesh_.Resize(mesh_params);
  compact_mesh_.Resize(mesh_params);
  bbox_.Resize(hash_params.value_capacity * 24);

  time_profile_.open(time_profile, std::ios::out);
  memo_profile_.open(memo_profile, std::ios::out);

  coordinate_converter_.voxel_size = sdf_params.voxel_size;
  coordinate_converter_.truncation_distance_scale =
      sdf_params.truncation_distance_scale;
  coordinate_converter_.truncation_distance =
      sdf_params.truncation_distance;
  coordinate_converter_.sdf_upper_bound = sdf_params.sdf_upper_bound;
  coordinate_converter_.weight_sample = sdf_params.weight_sample;
}

Map::~Map() {
  time_profile_.close();
  memo_profile_.close();
}

/// Reset
void Map::Reset() {
  integrated_frame_count_ = 0;

  hash_table_.Reset();
  blocks_.Reset();
  mesh_.Reset();

  candidate_entries_.Reset();
  compact_mesh_.Reset();
  bbox_.Reset();
}

/// Garbage collection
void Map::Recycle(int frame_count) {
  // TODO(wei): change it via global parameters

  int kRecycleGap = 15;
  if (frame_count % kRecycleGap == kRecycleGap - 1) {
    StarveOccupiedBlocks();

    CollectGarbageBlocks();
    hash_table_.ResetMutexes();
    RecycleGarbageBlocks();
  }
}

void Map::StarveOccupiedBlocks() {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  StarveOccupiedBlocksKernel<<<grid_size, block_size >>>(
          candidate_entries_.gpu_data(),
          blocks_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

void Map::CollectGarbageBlocks() {
  const uint threads_per_block = BLOCK_SIZE / 2;

  uint processing_block_count = candidate_entries_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  CollectGarbageBlocksKernel <<<grid_size, block_size >>>(
          candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
              coordinate_converter_);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

// TODO(wei): Check vertex / triangles in detail
// including garbage collection
void Map::RecycleGarbageBlocks() {
  const uint threads_per_block = BLOCK_SIZE;

  uint processing_block_count = candidate_entries_.entry_count();
  if (processing_block_count <= 0)
    return;

  const dim3 grid_size(processing_block_count, 1);
  const dim3 block_size(threads_per_block, 1);

  RecycleGarbageBlocksTrianglesKernel <<<grid_size, block_size >>>(
          hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  RecycleGarbageBlocksVerticesKernel <<<grid_size, block_size >>>(
      hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          blocks_.gpu_data(),
          mesh_.gpu_data());
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

  candidate_entries_.reset_entry_count();
  CollectAllBlocksKernel<<<grid_size, block_size >>>(
          hash_table_.gpu_data(),
          candidate_entries_.gpu_data());
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in all: "
            << candidate_entries_.entry_count();
}

void Map::CollectInFrustumBlocks(Sensor &sensor){
  const uint threads_per_block = 256;

  uint entry_count;
  checkCudaErrors(cudaMemcpy(&entry_count, hash_table_.gpu_data().entry_count,
                             sizeof(uint), cudaMemcpyDeviceToHost));

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  candidate_entries_.reset_entry_count();
  CollectInFrustumBlocksKernel<<<grid_size, block_size >>>(
      hash_table_.gpu_data(),
          candidate_entries_.gpu_data(),
          sensor.sensor_params(),
          sensor.c_T_w(),
          coordinate_converter_);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in view frustum: "
            << candidate_entries_.entry_count();
}

