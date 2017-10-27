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

// @function
// Enumerate @param candidate_entries
// operate over correspondent @param blocks
void StarveOccupiedBlockArray(
    EntryArray& candidate_entries,
    BlockArray& blocks
);
// @function
// Enumerate @param candidate_entries
// operate over correspondent @param blocks
// set flag for incoming @param candidate_entries
void CollectGarbageBlockArray(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    GeometryHelper& geometry_helper
);

// TODO(wei): Check vertex / triangles in detail
// @function
// Enumerate @param candidate_entries
// recycle correspondent @param blocks
//                   and @param mesh
// also free entry in @param hash_table if needed
void RecycleGarbageBlockArray(
    EntryArray &candidate_entries,
    BlockArray& blocks,
    Mesh& mesh,
    HashTable& hash_table
);

#endif //MESH_HASHING_RECYCLE_H
