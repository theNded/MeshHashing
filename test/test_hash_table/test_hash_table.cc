//
// Created by wei on 17-3-13.
//

#include <glog/logging.h>

#include <cuda_runtime.h>
#include "hash_table_gpu.h"
#include "test_hash_table.h"

#include <helper_cuda.h>

uint CpuRun(int3 pos) {
  const int p0 = 73856093;
  const int p1 = 19349669;
  const int p2 = 83492791;

  int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2)) % 500000;

  if (res < 0) res += 500000;
  return (uint)res;
}

int main(int argc, const char **argv) {
  HashTable hash_data;

  findCudaDevice(argc, argv);
  HashParams params;
  params.value_capacity = 256 * 256 * 4;
  params.bucket_count = 500000;
  params.bucket_size = 10;
  params.block_size = 8;

  hash_data.Alloc(params, true);
  LOG(INFO) << "Hash data allocated";

  TestHashTable test;
  test.Run(hash_data, make_int3(142, 33, 5));
  LOG(INFO) << CpuRun(make_int3(142, 33, 5));

  test.Run(hash_data, make_int3(0, 0, 0));
  LOG(INFO) << CpuRun(make_int3(0, 0, 0));

  test.Run(hash_data, make_int3(-1, -1, 2323));
  LOG(INFO) << CpuRun(make_int3(-1, -1, 2323));

  test.Run(hash_data, make_int3(34, 52, 142));
  LOG(INFO) << CpuRun(make_int3(34, 52, 142));

  hash_data.Free();
  LOG(INFO) << "Hash data freed";
}