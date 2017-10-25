#include "matrix.h"

#include "engine/main_engine.h"
#include "sensor/rgbd_sensor.h"

#include <helper_cuda.h>
#include <helper_math.h>

#include <unordered_set>
#include <vector>
#include <list>
#include <glog/logging.h>
#include <device_launch_parameters.h>
#include "meshing/mc_tables.h"


#define PINF  __int_as_float(0x7f800000)

////////////////////
/// class MappingEngine - compress, recycle
////////////////////

/// Condition: IsBlockInCameraFrustum
__global__
void CollectBlocksInFrustumKernel(HashTable hash_table,
                                  EntryArray candidate_entries,
                                  SensorParams sensor_params,
                                  float4x4 c_T_w,
                                  CoordinateConverter converter) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < hash_table.entry_count
    && hash_table.entry(idx).ptr != FREE_ENTRY
    && converter.IsBlockInCameraFrustum(c_T_w, hash_table.entry(idx).pos,
                                        sensor_params)) {
    addr_local = atomicAdd(&local_counter, 1);
  }
  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(&candidate_entries.counter(),
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    candidate_entries[addr] = hash_table.entry(idx);
  }
}

__global__
void CollectAllBlocksKernel(HashTable hash_table,
                            EntryArray candidate_entries) {
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ int local_counter;
  if (threadIdx.x == 0) local_counter = 0;
  __syncthreads();

  int addr_local = -1;
  if (idx < hash_table.entry_count
      && hash_table.entry(idx).ptr != FREE_ENTRY) {
    addr_local = atomicAdd(&local_counter, 1);
  }

  __syncthreads();

  __shared__ int addr_global;
  if (threadIdx.x == 0 && local_counter > 0) {
    addr_global = atomicAdd(&candidate_entries.counter(),
                            local_counter);
  }
  __syncthreads();

  if (addr_local != -1) {
    const uint addr = addr_global + addr_local;
    candidate_entries[addr] = hash_table.entry(idx);
  }
}

////////////////////
/// Host code
///////////////////

/// Compress discrete hash table entries
void CollectAllBlocks(EntryArray &candidate_entries, HashTable &hash_table) {
  const uint threads_per_block = 256;

  uint entry_count = hash_table.entry_count;
  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  candidate_entries.reset_count();
  CollectAllBlocksKernel <<<grid_size, block_size >>>(
          hash_table,
          candidate_entries);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in all: "
            << candidate_entries.count();
}

void CollectBlocksInFrustum(HashTable &hash_table,
                            EntryArray &candidate_entries,
                            Sensor &sensor,
                            CoordinateConverter &converter) {
  const uint threads_per_block = 256;

  uint entry_count = hash_table.entry_count;

  const dim3 grid_size((entry_count + threads_per_block - 1)
                       / threads_per_block, 1);
  const dim3 block_size(threads_per_block, 1);

  candidate_entries.reset_count();
  CollectBlocksInFrustumKernel <<<grid_size, block_size >>>(
      hash_table,
          candidate_entries,
          sensor.sensor_params(),
          sensor.c_T_w(),
          converter);

  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  LOG(INFO) << "Block count in view frustum: "
            << candidate_entries.count();
}

