//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_COMPRESS_MESH_H
#define MESH_HASHING_COMPRESS_MESH_H

#include "core/common.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "core/mesh.h"
#include "visualization/compact_mesh.h"

void CompressMesh(EntryArray& candidate_entries, BlockArray& blocks,
                  Mesh& mesh,
                  CompactMesh & compact_mesh, int3& stats);

#endif //MESH_HASHING_COMPRESS_MESH_H
