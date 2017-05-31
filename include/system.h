//
// Created by wei on 17-5-31.
//

#ifndef VOXEL_HASHING_SYSTEM_H
#define VOXEL_HASHING_SYSTEM_H

#include "dataset_manager.h"
#include "map.h"
#include "renderer.h"

class System {
private:
  Map*          map_;
  MeshRenderer* renderer_;

public:
  System(Dataset dataset,
          // Render options
         bool free_walk     = false,
         bool line_only     = true,
         bool new_mesh_only = false,
         bool fine_gradient = true,
         bool record_video  = false,
         bool ray_casting   = true);

  System(std::string dataset_path,
         DatasetType dataset_type,
         bool free_walk     = false,
         bool line_only     = true,
         bool new_mesh_only = false,
         bool fine_gradient = true,
         bool record_video  = false,
         bool ray_cating    = true);
};


#endif //VOXEL_HASHING_SYSTEM_H
