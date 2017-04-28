//
// Created by wei on 17-4-5.
//

#include "map.h"
#include "sensor.h"
#include <unordered_set>
#include <vector>
#include <list>

Map::Map(const HashParams &hash_params) {
  hash_params_ = hash_params;
  hash_table_.Alloc(hash_params_);

  Reset();
}

Map::~Map() {
  hash_table_.Free();
}

void Map::Reset() {
  integrated_frame_count_ = 0;
  occupied_block_count_ = 0;
  ResetCudaHost(hash_table_, hash_params_);
}

void Map::GenerateCompressedHashEntries(float4x4 c_T_w) {
  occupied_block_count_ = GenerateCompressedHashEntriesCudaHost(hash_table_, hash_params_, c_T_w);
  //this version uses atomics over prefix sums, which has a much better performance
  std::cout << "Occupied Blocks: " << occupied_block_count_ << std::endl;
}

void Map::RecycleInvalidBlocks() {
  bool kRecycle = true;         /// false
  int garbage_collect_starve = 15;      /// 15
  if (kRecycle) {

    if (integrated_frame_count_ > 0
        && integrated_frame_count_ % garbage_collect_starve == 0) {
      StarveOccupiedVoxelsCudaHost(hash_table_, hash_params_);
    }

    CollectInvalidBlockInfoCudaHost(hash_table_, hash_params_);
    ResetBucketMutexesCudaHost(hash_table_, hash_params_);
    //needed if linked lists are enabled -> for memeory deletion
    RecycleInvalidBlockCudaHost(hash_table_, hash_params_);
  }
}


//! debug only!
unsigned int Map::getHeapFreeCount() {
  unsigned int count;
  checkCudaErrors(cudaMemcpy(&count, hash_table_.heap_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  return count + 1;  //there is one more free than the address suggests (0 would be also a valid address)
}

void Map::debugHash() {
  HashEntry *hashCPU = new HashEntry[hash_params_.bucket_size * hash_params_.bucket_count];
  HashEntry *hashCompCPU = new HashEntry[occupied_block_count_];
  Voxel *voxelCPU = new Voxel[hash_params_.value_capacity * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
  unsigned int *heapCPU = new unsigned int[hash_params_.value_capacity];
  unsigned int heapCounterCPU;

  checkCudaErrors(cudaMemcpy(&heapCounterCPU, hash_table_.heap_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  heapCounterCPU++;  //points to the first free entry: number of blocks is one more

  checkCudaErrors(cudaMemcpy(heapCPU, hash_table_.heap, sizeof(unsigned int) * hash_params_.value_capacity, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCPU, hash_table_.hash_entries,
                             sizeof(HashEntry) * hash_params_.bucket_size * hash_params_.bucket_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCompCPU, hash_table_.compacted_hash_entries,
                             sizeof(HashEntry) * occupied_block_count_, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(voxelCPU, hash_table_.values,
                             sizeof(Voxel) * hash_params_.value_capacity * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE,
                             cudaMemcpyDeviceToHost));

  std::cout << "Compactified" << std::endl;

  //Check for duplicates
  class myint3Voxel {
  public:
    myint3Voxel() {}

    ~myint3Voxel() {}

    bool operator<(const myint3Voxel &other) const {
      if (x == other.x) {
        if (y == other.y) {
          return z < other.z;
        }
        return y < other.y;
      }
      return x < other.x;
    }

    bool operator==(const myint3Voxel &other) const {
      return x == other.x && y == other.y && z == other.z;
    }

    int x, y, z, i;
    int offset;
    int ptr;
  };


  std::unordered_set<unsigned int> pointersFreeHash;
  std::vector<int> pointersFreeVec(hash_params_.value_capacity, 0);
  for (unsigned int i = 0; i < heapCounterCPU; i++) {
    pointersFreeHash.insert(heapCPU[i]);
    pointersFreeVec[heapCPU[i]] = FREE_ENTRY;
  }
  if (pointersFreeHash.size() != heapCounterCPU) {
    std::cerr << "Heap check invalid" << std::endl;
  }


  unsigned int numOccupied = 0;
  unsigned int numMinusOne = 0;
  //unsigned int listOverallFound = 0;

  std::list<myint3Voxel> l;
  //std::vector<myint3Voxel> v;

  for (unsigned int i = 0; i < hash_params_.bucket_size * hash_params_.bucket_count; i++) {
    if (hashCPU[i].ptr == -1) {
      numMinusOne++;
    }

    if (hashCPU[i].ptr != -2) {
      numOccupied++;  // != FREE_ENTRY
      myint3Voxel a;
      a.x = hashCPU[i].pos.x;
      a.y = hashCPU[i].pos.y;
      a.z = hashCPU[i].pos.z;
      //std::cout << a.x << " " << a.y << " " << a.z << " " << voxelCPU[hashCPU[i].ptr] << std::endl;
      l.push_back(a);
      //v.push_back(a);


      if (pointersFreeHash.find(hashCPU[i].ptr) != pointersFreeHash.end()) {
        std::cerr << ("ERROR: ptr is on free heap, but also marked as an allocated entry");
      }
      pointersFreeVec[hashCPU[i].ptr] = LOCK_ENTRY;
    }
  }

  unsigned int numHeapFree = 0;
  unsigned int numHeapOccupied = 0;
  for (unsigned int i = 0; i < hash_params_.value_capacity; i++) {
    if (pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
    else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
    else {
      std::cerr << "memory leak detected: neither free nor allocated";
      exit(-1);
    }
  }
  if (numHeapFree + numHeapOccupied == hash_params_.value_capacity) std::cout << "HEAP OK!" << std::endl;
  else {
    std::cerr << "HEAP CORRUPTED";
    exit(-1);
  }

  l.sort();
  size_t sizeBefore = l.size();
  l.unique();
  size_t sizeAfter = l.size();


  std::cout << "diff: " << sizeBefore - sizeAfter << std::endl;
  std::cout << "minOne: " << numMinusOne << std::endl;
  std::cout << "numOccupied: " << numOccupied << "\t numFree: " << getHeapFreeCount() << std::endl;
  std::cout << "numOccupied + free: " << numOccupied + getHeapFreeCount() << std::endl;
  std::cout << "numInFrustum: " << occupied_block_count_ << std::endl;

  delete [] hashCPU;
  delete [] hashCompCPU;
  delete [] voxelCPU;
  delete [] heapCPU;
}
