//
// Created by wei on 17-10-22.
//

#ifndef MESH_HASHING_ALLOCATE_H
#define MESH_HASHING_ALLOCATE_H

#include "core/hash_table.h"
#include "geometry/coordinate_utils.h"
#include "sensor/rgbd_sensor.h"

void AllocBlockArray(HashTable& hash_table,
                     Sensor& sensor,
                     CoordinateConverter& converter);

#endif //MESH_HASHING_ALLOCATE_H
