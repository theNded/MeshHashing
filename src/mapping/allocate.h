//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_ALLOCATE_H
#define MESH_HASHING_ALLOCATE_H

#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"

void AllocBlockArray(HashTable& hash_table,
                     Sensor& sensor,
                     GeometryHelper& geoemtry_helper);

#endif //MESH_HASHING_ALLOCATE_H
