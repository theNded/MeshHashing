//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_FUSE_H
#define MESH_HASHING_FUSE_H

#include "core/hash_table.h"
#include "core/block_array.h"
#include "core/entry_array.h"
#include "core/mesh.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

void UpdateBlockArray(EntryArray& candidate_entries,
                      HashTable&  hash_table,
                      BlockArray& blocks,
                      Mesh& mesh,
                      Sensor &sensor,
                      GeometryHelper& geoemtry_helper);

#endif //MESH_HASHING_FUSE_H
