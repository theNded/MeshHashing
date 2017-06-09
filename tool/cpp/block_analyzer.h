//
// Created by wei on 17-6-9.
//

#ifndef VH_BLOCK_ANALYZER_H
#define VH_BLOCK_ANALYZER_H

#include "block.h"
#include "renderer.h"

/// We have to render points and meshes
/// Hence we have to dump a block along with its mesh
class BlockObject : public GLObjectBase {
protected:
  static const GLfloat kVertices[8];
  static const GLubyte kIndices[6];

  cudaGraphicsResource* cuda_voxels_;
  cudaGraphicsResource* cuda_vertices_;
  cudaGraphicsResource* cuda_normals_;
  cudaGraphicsResource* cuda_colors_;
  cudaGraphicsResource* cuda_triangles_;

  uint width_;
  uint height_;

public:
  BlockObject();
  ~BlockObject();

  void Render(glm::mat4 m, glm::mat4 v, glm::mat4 p);
  void SetData(Voxel* voxels,
               float3* vertices,
               float3* colors,
               float3* triangles);
};

class BlockAnalyzer {
  Block block;

};


#endif //VH_BLOCK_ANALYZER_H
