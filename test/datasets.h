//
// Created by wei on 17-5-31.
//

#ifndef VH_DATASETS_H
#define VH_DATASETS_H

#include "dataset_manager.h"

static const Dataset datasets[] = {
        {ICL,            "/home/wei/data/ICL/lv1/"},
        {TUM1,           "/home/wei/data/TUM/rgbd_dataset_freiburg1_xyz/"},
        {TUM2,           "/home/wei/data/TUM/rgbd_dataset_freiburg2_xyz/"},
        {TUM3,           "/home/wei/data/TUM/rgbd_dataset_freiburg3_long_office_household/"},
        {SUN3D,          "/home/wei/data/SUN3D/copyroom/"},
        {SUN3D_ORIGINAL, "/home/wei/data/SUN3D-Princeton/hotel_umd/maryland_hotel3/"},
        {PKU,            "/home/wei/data/3DVCR/lab3/"}
};

static const std::string orb_configs[] = {
        "", // ICL nill
        "../config/ORB/TUM1.yaml",
        "../config/ORB/TUM2.yaml",
        "../config/ORB/TUM3.yaml",
        "../config/ORB/SUN3D.yaml",
        "../config/ORB/SUN3D_ORIGINAL.yaml",
        "../config/ORB/PKU.yaml"
};

#endif //VH_DATASETS_H
