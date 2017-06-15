//
// Created by wei on 17-6-9.
//

#ifndef VH_BLOCK_ANALYZER_H
#define VH_BLOCK_ANALYZER_H

#include "block.h"
#include "renderer.h"

class BlockAnalyzer {
public:
  /// Host code:
  std::vector<float3> voxels;
  std::vector<float2> ssdf;
  std::vector<uchar2> sweight;

  std::vector<float3> vertices;
  std::vector<float3> colors;
  std::vector<int3>   triangles;

  void Save(int3 pos) {

  }
};


#endif //VH_BLOCK_ANALYZER_H
