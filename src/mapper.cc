//
// Created by wei on 17-3-16.
//

#include "mapper.h"
#include "sensor_data.h"
#include <unordered_set>
#include <vector>
#include <list>

CUDASceneRepHashSDF::CUDASceneRepHashSDF(const HashParams &params) {
  create(params);
}

CUDASceneRepHashSDF::~CUDASceneRepHashSDF() {
  destroy();
}

void CUDASceneRepHashSDF::bindDepthCameraTextures(const DepthCameraData &depthCameraData) {
  bindInputDepthColorTextures(depthCameraData);
}

void CUDASceneRepHashSDF::integrate(const float4x4 &lastRigidTransform, const DepthCameraData &depthCameraData,
               const DepthCameraParams &depthCameraParams, unsigned int *d_bitMask) {

  setLastRigidTransform(lastRigidTransform);
  /// transform ok ?

  //make the rigid transform available on the GPU
  m_hashData.updateParams(m_hashParams);
  /// seems OK

  //allocate all hash blocks which are corresponding to depth map entries
  alloc(depthCameraData, depthCameraParams, d_bitMask);
  /// DIFFERENT: d_bitMask now empty
  /// seems OK now, supported by MATLAB scatter3

  //generate a linear hash array with only occupied entries
  compactifyHashEntries(depthCameraData);
  /// seems OK, supported by MATLAB scatter3

  //volumetrically integrate the depth data into the depth SDFBlocks
  integrateDepthMap(depthCameraData, depthCameraParams);
  /// cuda kernel launching ok
  /// seems ok according to CUDA output

  garbageCollect(depthCameraData);
  /// not processed, ok

  m_numIntegratedFrames++;
}

void CUDASceneRepHashSDF::setLastRigidTransform(const float4x4 &lastRigidTransform) {
  m_hashParams.m_rigidTransform = lastRigidTransform;
  m_hashParams.m_rigidTransformInverse = m_hashParams.m_rigidTransform.getInverse();
}

void CUDASceneRepHashSDF::setLastRigidTransformAndCompactify(const float4x4 &lastRigidTransform, const DepthCameraData &depthCameraData) {
  setLastRigidTransform(lastRigidTransform);
  compactifyHashEntries(depthCameraData);
}

const float4x4 CUDASceneRepHashSDF::getLastRigidTransform() const {
  return m_hashParams.m_rigidTransform;
}

//! resets the hash to the initial state (i.e., clears all data)
void CUDASceneRepHashSDF::reset() {
  m_numIntegratedFrames = 0;

  m_hashParams.m_rigidTransform.setIdentity();
  m_hashParams.m_rigidTransformInverse.setIdentity();
  m_hashParams.occupied_block_count = 0;
  m_hashData.updateParams(m_hashParams);
  resetCUDA(m_hashData, m_hashParams);
}


HashTable& CUDASceneRepHashSDF::getHashTable() {
  return m_hashData;
}

const HashParams& CUDASceneRepHashSDF::getHashParams() const {
  return m_hashParams;
}


//! debug only!
unsigned int CUDASceneRepHashSDF::getHeapFreeCount() {
  unsigned int count;
  checkCudaErrors(cudaMemcpy(&count, m_hashData.heap_counter, sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
  return count + 1;  //there is one more free than the address suggests (0 would be also a valid address)
}

//! debug only!
void CUDASceneRepHashSDF::debugHash() {
  HashEntry *hashCPU = new HashEntry[m_hashParams.bucket_size * m_hashParams.bucket_count];
  HashEntry *hashCompCPU = new HashEntry[m_hashParams.occupied_block_count];
  Voxel *voxelCPU = new Voxel[m_hashParams.block_count * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE];
  unsigned int *heapCPU = new unsigned int[m_hashParams.block_count];
  unsigned int heapCounterCPU;

  checkCudaErrors(cudaMemcpy(&heapCounterCPU, m_hashData.heap_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost));
  heapCounterCPU++;  //points to the first free entry: number of blocks is one more

  checkCudaErrors(cudaMemcpy(heapCPU, m_hashData.heap, sizeof(unsigned int) * m_hashParams.block_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCPU, m_hashData.hash_entries,
             sizeof(HashEntry) * m_hashParams.bucket_size * m_hashParams.bucket_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(hashCompCPU, m_hashData.compacted_hash_entries,
                             sizeof(HashEntry) * m_hashParams.occupied_block_count, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(voxelCPU, m_hashData.blocks,
                  sizeof(Voxel) * m_hashParams.block_count * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE * SDF_BLOCK_SIZE,
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
  std::vector<int> pointersFreeVec(m_hashParams.block_count, 0);
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

  for (unsigned int i = 0; i < m_hashParams.bucket_size * m_hashParams.bucket_count; i++) {
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
              m_hashParams.block_size * m_hashParams.block_size * m_hashParams.block_size;
      if (pointersFreeHash.find(hashCPU[i].ptr / linearBlockSize) != pointersFreeHash.end()) {
        std::cerr << ("ERROR: ptr is on free heap, but also marked as an allocated entry");
      }
      pointersFreeVec[hashCPU[i].ptr / linearBlockSize] = LOCK_ENTRY;
    }
  }

  unsigned int numHeapFree = 0;
  unsigned int numHeapOccupied = 0;
  for (unsigned int i = 0; i < m_hashParams.block_count; i++) {
    if (pointersFreeVec[i] == FREE_ENTRY) numHeapFree++;
    else if (pointersFreeVec[i] == LOCK_ENTRY) numHeapOccupied++;
    else {
      std::cerr << "memory leak detected: neither free nor allocated";
      exit(-1);
    }
  }
  if (numHeapFree + numHeapOccupied == m_hashParams.block_count) std::cout << "HEAP OK!" << std::endl;
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
  std::cout << "numInFrustum: " << m_hashParams.occupied_block_count << std::endl;

  delete [] hashCPU;
  delete [] hashCompCPU;
  delete [] voxelCPU;
  delete [] heapCPU;
}


void CUDASceneRepHashSDF::create(const HashParams &params) {
  m_hashParams = params;
  m_hashData.allocate(m_hashParams);

  reset();
}

void CUDASceneRepHashSDF::destroy() {
  m_hashData.free();
}

void CUDASceneRepHashSDF::alloc(const DepthCameraData &depthCameraData, const DepthCameraParams &depthCameraParams,
           const unsigned int *d_bitMask) {

  bool offline_processing = false; /// assumed to be false
  if (offline_processing) {
    //allocate until all blocks are allocated
    unsigned int prevFree = getHeapFreeCount();
    while (1) {
      resetHashBucketMutexCUDA(m_hashData, m_hashParams);
      allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);

      unsigned int currFree = getHeapFreeCount();

      if (prevFree != currFree) {
        prevFree = currFree;
      } else {
        break;
      }
    }
  } else {
    //this version is faster, but it doesn't guarantee that all blocks are allocated (staggers alloc to the next frame)
    resetHashBucketMutexCUDA(m_hashData, m_hashParams);
    allocCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams, d_bitMask);
    /// !!! NOBODY IS ALLOCATED
  }
}


void CUDASceneRepHashSDF::compactifyHashEntries(const DepthCameraData &depthCameraData) {

  m_hashParams.occupied_block_count = compactifyHashAllInOneCUDA(m_hashData,
                                                                m_hashParams);    //this version uses atomics over prefix sums, which has a much better performance
  std::cout << "Occupied Blocks: " << m_hashParams.occupied_block_count << std::endl;
  m_hashData.updateParams(m_hashParams);  //make sure numOccupiedBlocks is updated on the GPU
}

void CUDASceneRepHashSDF::integrateDepthMap(const DepthCameraData &depthCameraData, const DepthCameraParams &depthCameraParams) {
  integrateDepthMapCUDA(m_hashData, m_hashParams, depthCameraData, depthCameraParams);
}

void CUDASceneRepHashSDF::garbageCollect(const DepthCameraData &depthCameraData) {
  //only perform if enabled by global app state
  bool garbage_collect = true;         /// false
  int garbage_collect_starve = 15;      /// 15
  if (garbage_collect) {

    if (m_numIntegratedFrames > 0 && m_numIntegratedFrames % garbage_collect_starve == 0) {
      starveVoxelsKernelCUDA(m_hashData, m_hashParams);
    }

    garbageCollectIdentifyCUDA(m_hashData, m_hashParams);
    resetHashBucketMutexCUDA(m_hashData, m_hashParams);  //needed if linked lists are enabled -> for memeory deletion
    garbageCollectFreeCUDA(m_hashData, m_hashParams);
  }
}