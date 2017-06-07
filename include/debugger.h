//
// Created by wei on 17-6-4.
//

#ifndef VOXEL_HASHING_DEBUGGER_H
#define VOXEL_HASHING_DEBUGGER_H

#include "hash_table.h"
#include "block.h"
#include "mesh.h"

class Debugger {
private:
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
};


#endif //VOXEL_HASHING_DEBUGGER_H
