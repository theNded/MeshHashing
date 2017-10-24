//
// Created by wei on 17-10-21.
//

#ifndef MESH_HASHING_BOUNDING_BOX_H
#define MESH_HASHING_BOUNDING_BOX_H

class BoundingBox {
public:
  BoundingBox() = default;
  //~BoundingBox();
  void Alloc(int max_vertex_count);
  void Free();

  uint vertex_count();
  void Resize(int max_vertex_count);
  void Reset();

  __device__
  float3* vertices() {
    return vertices_;
  }
  __device__
  uint* vertex_counter() {
    return vertex_counter_;
  }
private:
  float3* vertices_;
  uint*   vertex_counter_;

  int max_vertex_count_;
};

#endif //MESH_HASHING_BOUNDING_BOX_H
