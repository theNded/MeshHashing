//
// Created by wei on 17-10-26.
//

#ifndef MESH_HASHING_PRIMAL_DUAL_H
#define MESH_HASHING_PRIMAL_DUAL_H

#include "core/common.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/hash_table.h"
#include "geometry/geometry_helper.h"

void PrimalDualInit(
    EntryArray& candidate_entries,
    BlockArray& blocks
);

void PrimalDualIterate(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    HashTable& hash_table,
    GeometryHelper& geometry_helper,
    const float lambda,
    const float sigma,
    const float tau
);
#endif //MESH_HASHING_PRIMAL_DUAL_H
