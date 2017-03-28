//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_HASH_TABLE_MANAGER_H
#define MRF_VH_HASH_TABLE_MANAGER_H

#include "common.h"

#include "hash_param.h"
#include "hash_table.h"
#include "sensor_data.h"
#include "sensor_param.h"

/// CUDA functions
extern void resetCUDA(HashData& hashData, const HashParams& hashParams);
extern void resetHashBucketMutexCUDA(HashData& hashData, const HashParams& hashParams);
extern void allocCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask);

/// Assume it is only used for streaming
extern void fillDecisionArrayCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData);
extern void compactifyHashCUDA(HashData& hashData, const HashParams& hashParams);
extern unsigned int compactifyHashAllInOneCUDA(HashData& hashData, const HashParams& hashParams);

/// ! FUSION PART !
extern void integrateDepthMapCUDA(HashData& hashData, const HashParams& hashParams, const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
extern void bindInputDepthColorTextures(const DepthCameraData& depthCameraData);

/// Garbage collection
extern void starveVoxelsKernelCUDA(HashData& hashData, const HashParams& hashParams);
extern void garbageCollectIdentifyCUDA(HashData& hashData, const HashParams& hashParams);
extern void garbageCollectFreeCUDA(HashData& hashData, const HashParams& hashParams);

/// CUDA / C++ shared class
class CUDASceneRepHashSDF {
public:
  /// Construct and deconstruct
  CUDASceneRepHashSDF(const HashParams& params);
  ~CUDASceneRepHashSDF();

  void reset();

  /// Set input (image)
  void bindDepthCameraTextures(const DepthCameraData& depthCameraData);

  /// SDF fusion
  void integrate(const float4x4& lastRigidTransform, const DepthCameraData& depthCameraData,
                 const DepthCameraParams& depthCameraParams, unsigned int* d_bitMask);

  /// Set pose
  void setLastRigidTransform(const float4x4& lastRigidTransform);
  void setLastRigidTransformAndCompactify(const float4x4& lastRigidTransform, const DepthCameraData& depthCameraData);
  const float4x4 getLastRigidTransform() const;

  /// Member accessor
  HashData& getHashData();
  const HashParams& getHashParams() const;

  //! debug only!
  unsigned int getHeapFreeCount();
  void debugHash();

private:
  void create(const HashParams& params);
  void destroy();
  void alloc(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams, const unsigned int* d_bitMask);

  void compactifyHashEntries(const DepthCameraData& depthCameraData);
  void integrateDepthMap(const DepthCameraData& depthCameraData, const DepthCameraParams& depthCameraParams);
  void garbageCollect(const DepthCameraData& depthCameraData);

  HashParams	m_hashParams;
  HashData		m_hashData;

  // CUDAScan		m_cudaScan; disable at current
  unsigned int	m_numIntegratedFrames;	//used for garbage collect
};

#endif //MRF_VH_HASH_TABLE_MANAGER_H
