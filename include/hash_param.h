//
// Created by wei on 17-3-12.
//

/// Parameters for HashTable
/// Shared in __constant__ form
/// Update it ONLY in hash_table
#ifndef VH_HASH_PARAM_H
#define VH_HASH_PARAM_H

#include "common.h"

#include <matrix.h>

struct __ALIGN__(16) HashParams {
  /// Latest rigid transform of the sensor
  /// TODO: move it elsewhere (maybe sensor?), or not __constant__
  float4x4		m_rigidTransform;
  float4x4		m_rigidTransformInverse;

  //////////////////////////////////////////////////
  /// TODO: add m_num_voxels and m_num_hash_entries
  /// Currently used parameters
  unsigned int	m_hashNumBuckets;                 // 500000
  unsigned int	m_hashBucketSize;                 // 10
  unsigned int	m_hashMaxCollisionLinkedListSize; // 7
  unsigned int	m_numSDFBlocks;                   // 256 * 256 * 4 -> 1000000 (8x8x8 per block)

  int				    m_SDFBlockSize;                   // 8 (8x8x8)
  float			    m_virtualVoxelSize;               // 0.004 (m)
  unsigned int	m_numOccupiedBlocks;	            // occupied blocks in the viewing frustum

  float			    m_maxIntegrationDistance;         // 4.0 (m)
  float			    m_truncScale;                     // 0.01 (m / m)
  float		    	m_truncation;                     // 0.02 (m)
  unsigned int	m_integrationWeightSample;        // 10,  TODO: change it!
  unsigned int	m_integrationWeightMax;           // 255
  //////////////////////////////////////////////////

  /// Stream from GPU to CPU (external storage)
  /// Go through these later
  float3		m_streamingVoxelExtents;
  int3			m_streamingGridDimensions;
  int3			m_streamingMinGridPos;
  unsigned int	m_streamingInitialChunkListSize;
  uint2			m_dummy;

};
#endif //VH_HASH_PARAM_H
