//
// Created by wei on 17-3-16.
//

#include "mapper.h"
#include "sensor_data.h"
#include <unordered_set>
#include <vector>
#include <list>

Mapper::Mapper(Map* voxel_map) {
  map_ = voxel_map;
}

Mapper::~Mapper() {}

void Mapper::bindDepthCameraTextures(const SensorData &sensor_data) {
  bindInputDepthColorTextures(sensor_data);
}

void Mapper::integrate(const float4x4 &lastRigidTransform, const SensorData &sensor_data,
               const SensorParams &depthCameraParams, unsigned int *d_bitMask) {

  setLastRigidTransform(lastRigidTransform);
  /// transform ok ?

  //make the rigid transform available on the GPU
  map_->hash_table().updateParams(map_->hash_params());
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  map_->AllocBlocks(sensor_data, depthCameraParams);
  /// DIFFERENT: d_bitMask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  map_->GenerateCompressedHashEntries();
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  integrateDepthMap(sensor_data, depthCameraParams);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  map_->RecycleInvalidBlocks();
  /// not processed, ok
  map_->integrated_frame_count_++;
}

void Mapper::setLastRigidTransform(const float4x4 &lastRigidTransform) {
  map_->hash_params().m_rigidTransform = lastRigidTransform;
  map_->hash_params().m_rigidTransformInverse
          = map_->hash_params().m_rigidTransform.getInverse();
}

//! debug only!
unsigned int Mapper::getHeapFreeCount() {
  unsigned int count;
  checkCudaErrors(cudaMemcpy(&count, map_->hash_table().heap_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  return count + 1;  //there is one more free than the address suggests (0 would be also a valid address)
}

//! debug only!
void Mapper::debugHash() {
  HashTable  hash_table_ = map_->hash_table();
  HashParams hash_params_ = map_->hash_params();
  HashEntry *hashCPU = new HashEntry[hash_params_.bucket_size * hash_params_.bucket_count];
  HashEntry *hashCompCPU = new HashEntry[hash_params_.occupied_block_count];
  Voxel *voxelCPU = new Voxel[hash_params_.block_count * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
  unsigned int *heapCPU = new unsigned int[hash_params_.block_count];
  unsigned int heapCounterCPU;

  checkCudaErrors(cudaMemcpy(&heapCounterCPU, hash_table_.heap_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  heapCounterCPU++;  //points to the first free entry: number of blocks is one more

  checkCudaErrors(cudaMemcpy(heapCPU, hash_table_.heap, sizeof(unsigned int) * hash_params_.block_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCPU, hash_table_.hash_entries,
             sizeof(HashEntry) * hash_params_.bucket_size * hash_params_.bucket_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCompCPU, hash_table_.compacted_hash_entries,
                             sizeof(HashEntry) * hash_params_.occupied_block_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(voxelCPU, hash_table_.blocks,
                  sizeof(Voxel) * hash_params_.block_count * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE,
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
  std::vector<int> pointersFreeVec(hash_params_.block_count, 0);
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

      unsigned int linearBlockSize =
              hash_params_.block_size * hash_params_.block_size * hash_params_.block_size;
      if (pointersFreeHash.find(hashCPU[i].ptr / linearBlockSize) != pointersFreeHash.end()) {
        std::cerr << ("ERROR: ptr is on free heap, but also marked as an allocated entry");
      }
      pointersFreeVec[hashCPU[i].ptr / linearBlockSize] = LOCK_ENTRY;
    }
  }

  unsigned int numHeapFree = 0;
  unsigned int numHeapOccupied = 0;
  for (unsigned int i = 0; i < hash_params_.block_count; i++) {
    if (pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
    else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
    else {
      std::cerr << "memory leak detected: neither free nor allocated";
      exit(-1);
    }
  }
  if (numHeapFree + numHeapOccupied == hash_params_.block_count) std::cout << "HEAP OK!" << std::endl;
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
  std::cout << "numInFrustum: " << hash_params_.occupied_block_count << std::endl;

  delete [] hashCPU;
  delete [] hashCompCPU;
  delete [] voxelCPU;
  delete [] heapCPU;
}

void Mapper::integrateDepthMap(const SensorData &sensor_data, const SensorParams &depthCameraParams) {
  integrateDepthMapCUDA(map_->hash_table(), map_->hash_params(), sensor_data, depthCameraParams);
}