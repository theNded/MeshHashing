//
// Created by wei on 17-10-21.
//

#ifndef MESH_HASHING_BOUNDING_BOX_H
#define MESH_HASHING_BOUNDING_BOX_H

//////////////////////
/// Bonuding Box, used for debugging
/////////////////////
struct BBoxGPU {
  float3* vertices;
  uint*   vertex_counter;
};

class BBox {
private:
  BBoxGPU gpu_memory_;
  int max_vertex_count_;

  void Alloc(int max_vertex_count);

  void Free();

public:
  BBox();

  ~BBox();

  uint vertex_count();

  float3 *vertices() {
    return gpu_memory_.vertices;
  }

  void Resize(int amx_vertex_count);

  void Reset();

  BBoxGPU &gpu_memory() {
    return gpu_memory_;
  }
};



#endif //MESH_HASHING_BOUNDING_BOX_H
