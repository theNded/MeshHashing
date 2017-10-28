//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_ALLOCATE_H
#define MESH_HASHING_ALLOCATE_H

#include "core/hash_table.h"
#include "geometry/geometry_helper.h"
#include "sensor/rgbd_sensor.h"

// @function
// See what entries of @param hash_table
// was affected by @param sensor
// with the help of @param geometry_helper
double AllocBlockArray(
    HashTable& hash_table,
    Sensor& sensor,
    GeometryHelper& geometry_helper
);

#endif //MESH_HASHING_ALLOCATE_H
