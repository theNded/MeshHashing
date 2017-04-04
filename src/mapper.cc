//
// Created by wei on 17-3-16.
//

#include "mapper.h"
#include "sensor_data.h"
#include <unordered_set>
#include <vector>
#include <list>

Mapper::Mapper(const HashParams &params) {
  create(params);
}

Mapper::~Mapper() {
  destroy();
}

void Mapper::bindDepthCameraTextures(const SensorData &sensor_data) {
  bindInputDepthColorTextures(sensor_data);
}

void Mapper::integrate(const float4x4 &lastRigidTransform, const SensorData &sensor_data,
               const SensorParams &depthCameraParams, unsigned int *d_bitMask) {

  setLastRigidTransform(lastRigidTransform);
  /// transform ok ?

  //make the rigid transform available on the GPU
  hash_table_.updateParams(hash_params_);
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  alloc(sensor_data, depthCameraParams, d_bitMask);
  /// DIFFERENT: d_bitMask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  compactifyHashEntries(sensor_data);
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  integrateDepthMap(sensor_data, depthCameraParams);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  garbageCollect(sensor_data);
  /// not processed, ok

  m_numIntegratedFrames++;
}

void Mapper::setLastRigidTransform(const float4x4 &lastRigidTransform) {
  hash_params_.m_rigidTransform = lastRigidTransform;
  hash_params_.m_rigidTransformInverse = hash_params_.m_rigidTransform.getInverse();
}

void Mapper::setLastRigidTransformAndCompactify(const float4x4 &lastRigidTransform, const SensorData &sensor_data) {
  setLastRigidTransform(lastRigidTransform);
  compactifyHashEntries(sensor_data);
}

const float4x4 Mapper::getLastRigidTransform() const {
  return hash_params_.m_rigidTransform;
}

//! resets the hash to the initial state (i.e., clears all data)
void Mapper::reset() {
  m_numIntegratedFrames = 0;

  hash_params_.m_rigidTransform.setIdentity();
  hash_params_.m_rigidTransformInverse.setIdentity();
  hash_params_.occupied_block_count = 0;
  hash_table_.updateParams(hash_params_);
  resetCUDA(hash_table_, hash_params_);
}


HashTable& Mapper::getHashTable() {
  return hash_table_;
}

const HashParams& Mapper::getHashParams() const {
  return hash_params_;
}


//! debug only!
unsigned int Mapper::getHeapFreeCount() {
  unsigned int count;
  checkCudaErrors(cudaMemcpy(&count, hash_table_.heap_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  return count + 1;  //there is one more free than the address suggests (0 would be also a valid address)
}

//! debug only!
void Mapper::debugHash() {
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


void Mapper::create(const HashParams &params) {
  hash_params_ = params;
  hash_table_.Alloc(hash_params_);

  reset();
}

void Mapper::destroy() {
  hash_table_.free();
}

void Mapper::alloc(const SensorData &sensor_data, const SensorParams &depthCameraParams,
           const unsigned int *d_bitMask) {

  bool offline_processing = false; /// assumed to be false
  if (offline_processing) {
    //allocate until all blocks are allocated
    unsigned int prevFree = getHeapFreeCount();
    while (1) {
      resetHashBucketMutexCUDA(hash_table_, hash_params_);
      allocCUDA(hash_table_, hash_params_, sensor_data, depthCameraParams, d_bitMask);

      unsigned int currFree = getHeapFreeCount();

      if (prevFree != currFree) {
        prevFree = currFree;
      } else {
        break;
      }
    }
  } else {
    //this version is faster, but it doesn't guarantee that all blocks are allocated (staggers alloc to the next frame)
    resetHashBucketMutexCUDA(hash_table_, hash_params_);
    allocCUDA(hash_table_, hash_params_, sensor_data, depthCameraParams, d_bitMask);
    /// !!! NOBODY IS ALLOCATED
  }
}


void Mapper::compactifyHashEntries(const SensorData &sensor_data) {

  hash_params_.occupied_block_count = compactifyHashAllInOneCUDA(hash_table_,
                                                                hash_params_);    //this version uses atomics over prefix sums, which has a much better performance
  std::cout << "Occupied Blocks: " << hash_params_.occupied_block_count << std::endl;
  hash_table_.updateParams(hash_params_);  //make sure numOccupiedBlocks is updated on the GPU
}

void Mapper::integrateDepthMap(const SensorData &sensor_data, const SensorParams &depthCameraParams) {
  integrateDepthMapCUDA(hash_table_, hash_params_, sensor_data, depthCameraParams);
}

void Mapper::garbageCollect(const SensorData &sensor_data) {
  //only perform if enabled by global app state
  bool garbage_collect = true;         /// false
  int garbage_collect_starve = 15;      /// 15
  if (garbage_collect) {

    if (m_numIntegratedFrames > 0 && m_numIntegratedFrames % garbage_collect_starve == 0) {
      starveVoxelsKernelCUDA(hash_table_, hash_params_);
    }

    garbageCollectIdentifyCUDA(hash_table_, hash_params_);
    resetHashBucketMutexCUDA(hash_table_, hash_params_);  //needed if linked lists are enabled -> for memeory deletion
    garbageCollectFreeCUDA(hash_table_, hash_params_);
  }
}