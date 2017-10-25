//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_COLLECT_H
#define MESH_HASHING_COLLECT_H

#include "core/entry_array.h"
#include "core/hash_table.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

void CollectAllBlocks(EntryArray &candidate_entries, HashTable &hash_table);
void CollectBlocksInFrustum(HashTable &hash_table,
                            EntryArray &candidate_entries,
                            Sensor &sensor,
                            GeometryHelper &geoemtry_helper);

#endif //MESH_HASHING_COLLECT_H
