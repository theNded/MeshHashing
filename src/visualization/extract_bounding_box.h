//
// Created by wei on 17-10-23.
//

#ifndef MESH_HASHING_EXTRACT_BOUNDING_BOX_H
#define MESH_HASHING_EXTRACT_BOUNDING_BOX_H

#include "core/common.h"
#include "core/entry_array.h"

#include "visualization/bounding_box.h"
#include "visualization/extract_bounding_box.h"
#include "geometry/geometry_helper.h"

void ExtractBoundingBox(EntryArray& candidate_entries,
                        BoundingBox& bounding_box,
                        GeometryHelper& geoemtry_helper);

#endif //MESH_HASHING_EXTRACT_BOUNDING_BOX_H
