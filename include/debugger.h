//
// Created by wei on 17-6-4.
//

#ifndef VOXEL_HASHING_DEBUGGER_H
#define VOXEL_HASHING_DEBUGGER_H

#include <vector>
#include <unordered_map>
#include "hash_table.h"
#include "block.h"
#include "mesh.h"

struct Hash3D {
  const static int bucket_count = 1000000;
  std::size_t operator() (const int3& pos) const {
    const int p0 = 73856093;
    const int p1 = 19349669;
    const int p2 = 83492791;

    int res = ((pos.x * p0) ^ (pos.y * p1) ^ (pos.z * p2))
              % (bucket_count);
    if (res < 0) res += (bucket_count);
    return (size_t) res;
  }
};

class Debugger {
private:

  std::unordered_map<int3, std::vector<Block>, Hash3D> block_map_;

  HashEntry *entries_;
  uint      *heap_;
  uint      *heap_counter_;
  ///           |
  ///           v
  Block     *blocks_;
  ///           |
  ///           v
  uint*   vertex_heap;
  uint*   vertex_heap_counter;
  Vertex* vertices;

  uint*     triangle_heap;
  uint*     triangle_heap_counter;
  Triangle* triangles;

  int entry_count_;
  int block_count_;

public:
  Debugger(int entry_count, int block_count);
  ~Debugger();

  void CoreDump(HashTableGPU& hash_table);
  void CoreDump(BlocksGPU&    blocks);
  void CoreDump(MeshGPU&      mesh);

  void DebugHashToBlock();
  void DebugBlockToMesh();

  void PrintDebugInfo();
};


#endif //VOXEL_HASHING_DEBUGGER_H
