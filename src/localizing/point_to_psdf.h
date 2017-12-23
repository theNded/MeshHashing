//
// Created by wei on 17-12-23.
//

#ifndef MESH_HASHING_POINT_TO_PSDF_H
#define MESH_HASHING_POINT_TO_PSDF_H

#include "core/common.h"
#include "core/hash_table.h"
#include "core/mesh.h"
#include "core/entry_array.h"
#include "core/block_array.h"
#include "sensor/rgbd_sensor.h"
#include "geometry/geometry_helper.h"

float PointToSurface(
    BlockArray &blocks,
    Sensor &sensor,
    HashTable &hash_table,
    GeometryHelper &geometry_helper,
    mat6x6 &A,
    mat6x1 &b,
    int& count);
#endif //MESH_HASHING_POINT_TO_PSDF_H
