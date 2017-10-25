//
// Created by wei on 17-10-22.
//

#ifndef CORE_COLLECT_H
#define CORE_COLLECT_H

#include "core/entry_array.h"
#include "core/hash_table.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

// @function
// Read the entries in @param hash_table
// Write to the @param candidate_entries (for parallel computation)
void CollectAllBlocks(
    HashTable &hash_table,
    EntryArray &candidate_entries
);

// @function
// Read the entries in @param hash_table
// Filter the positions with @param sensor info (pose and params),
//                       and @param geometry helper
// Write to the @param candidate_entries (for parallel computation)
void CollectBlocksInFrustum(
    HashTable &hash_table,
    Sensor &sensor,
    GeometryHelper &geometry_helper,
    EntryArray &candidate_entries
);

#endif //CORE_COLLECT_H
