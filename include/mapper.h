//
// Created by wei on 17-3-16.
//

#ifndef MRF_VH_HASH_TABLE_MANAGER_H
#define MRF_VH_HASH_TABLE_MANAGER_H

#include "common.h"

#include "geometry_util.h"
#include "hash_param.h"
#include "hash_table.h"
#include "map.h"
#include "sensor_data.h"
#include "sensor_param.h"

/// CUDA functions
extern void resetCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void resetHashBucketMutexCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void allocCUDA(HashTable& hash_table, const HashParams& hash_params, const SensorData& sensor_data, const SensorParams& depthCameraParams, const unsigned int* d_bitMask);

/// Assume it is only used for streaming
extern void fillDecisionArrayCUDA(HashTable& hash_table, const HashParams& hash_params, const SensorData& sensor_data);
extern void compactifyHashCUDA(HashTable& hash_table, const HashParams& hash_params);
extern unsigned int compactifyHashAllInOneCUDA(HashTable& hash_table, const HashParams& hash_params);

/// ! FUSION PART !
extern void integrateDepthMapCUDA(HashTable& hash_table, const HashParams& hash_params, const SensorData& sensor_data, const SensorParams& depthCameraParams);
extern void bindInputDepthColorTextures(const SensorData& sensor_data);

/// Garbage collection
extern void starveVoxelsKernelCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void garbageCollectIdentifyCUDA(HashTable& hash_table, const HashParams& hash_params);
extern void garbageCollectFreeCUDA(HashTable& hash_table, const HashParams& hash_params);

/// CUDA / C++ shared class
class Mapper {
public:
  /// Construct and deconstruct
  Mapper(const HashParams& params);
  ~Mapper();

  void reset();

  /// Set input (image)
  void bindDepthCameraTextures(const SensorData& sensor_data);

  /// SDF fusion
  void integrate(const float4x4& lastRigidTransform, const SensorData& sensor_data,
                 const SensorParams& depthCameraParams, unsigned int* d_bitMask);

  /// Set pose
  void setLastRigidTransform(const float4x4& lastRigidTransform);
  void setLastRigidTransformAndCompactify(const float4x4& lastRigidTransform, const SensorData& sensor_data);
  const float4x4 getLastRigidTransform() const;

  /// Member accessor
  HashTable& getHashTable();
  const HashParams& getHashParams() const;

  //! debug only!
  unsigned int getHeapFreeCount();
  void debugHash();

private:
  void create(const HashParams& params);
  void destroy();
  void alloc(const SensorData& sensor_data,
             const SensorParams& depthCameraParams,
             const unsigned int* d_bitMask);

  void compactifyHashEntries(const SensorData& sensor_data);
  void integrateDepthMap(const SensorData& sensor_data, const SensorParams& depthCameraParams);
  void garbageCollect(const SensorData& sensor_data);

  HashParams	hash_params_;
  HashTable		hash_table_;

  // CUDAScan		m_cudaScan; disable at current
  unsigned int	m_numIntegratedFrames;	//used for garbage collect
};

#endif //MRF_VH_HASH_TABLE_MANAGER_H
