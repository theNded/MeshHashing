//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_RECYCLE_H
#define MESH_HASHING_RECYCLE_H

#include "core/common.h"
#include "core/hash_table.h"
#include "core/mesh.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "geometry/geometry_helper.h"

void StarveOccupiedBlockArray(EntryArray& candidate_entries,
                              BlockArray& blocks);
void CollectGarbageBlockArray(EntryArray& candidate_entries,
                              BlockArray& blocks,
                              GeometryHelper& geoemtry_helper);


// TODO(wei): Check vertex / triangles in detail
// including garbage collection
void RecycleGarbageBlockArray(HashTable& hash_table,
                              EntryArray &candidate_entries,
                              BlockArray& blocks,
                              Mesh& mesh);

#endif //MESH_HASHING_RECYCLE_H
