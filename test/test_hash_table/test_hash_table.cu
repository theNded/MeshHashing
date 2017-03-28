#include "hash_table.h"
#include "test_hash_table.h"
#include <helper_cuda.h>

__global__ void ComputeHashPos(HashData hash_data, int3 pos) {
  printf("__global__: ComputeHashPos\n");
  uint id = hash_data.computeHashPos(pos);
  printf("Hash(%d, %d, %d) = %d\n", pos.x, pos.y, pos.z, id);
}

void TestHashData::Run(HashData &hash_data, int3 pos) {
  ComputeHashPos<<<1, 1>>>(hash_data, pos);
}