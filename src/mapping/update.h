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

// @function
// Enumerate @param candidate_entries
// change the value of @param blocks
// according to the existing @param mesh
//                 and input @param sensor data
// with the help of hash_table and geometry_helper
void UpdateBlockArray(
    EntryArray& candidate_entries,
    BlockArray& blocks,
    Mesh& mesh,
    Sensor& sensor,
    HashTable& hash_table,
    GeometryHelper& geometry_helper
);

#endif //MESH_HASHING_FUSE_H
