//
// Created by wei on 17-10-25.
//

#ifndef MAPPING_UPDATE_PROBABILISTIC_H
#define MAPPING_UPDATE_PROBABILISTIC_H

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
void RefineSensorData(
    EntryArray &candidate_entries,
    BlockArray &blocks,
    Mesh &mesh,
    Sensor &sensor,
    HashTable &hash_table,
    GeometryHelper &geometry_helper
);

#endif //MESH_HASHING_UPDATE_PROBABILISTIC_H
