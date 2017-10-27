//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_MESH_WRITER_H
#define MESH_HASHING_MESH_WRITER_H

#include <string>
#include "core/common.h"
#include "visualization/compact_mesh.h"

void SaveObj(CompactMesh& compact_mesh, std::string path);
void SavePly(CompactMesh& compact_mesh, std::string path);

#endif //MESH_HASHING_MESH_WRITER_H
