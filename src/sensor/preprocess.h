//
// Created by wei on 17-10-25.
//

#ifndef MESH_HASHING_PREPROCESS_H
#define MESH_HASHING_PREPROCESS_H

#include <opencv2/opencv.hpp>
#include <helper_math.h>
#include "core/params.h"

__host__
void ConvertDepthFormat(
    cv::Mat& depth_img,
    short* depth_buffer,
    float* depth_data,
    SensorParams& params
);

__host__
void ConvertColorFormat(
    cv::Mat &color_img,
    uchar4* color_buffer,
    float4* color_data,
    SensorParams& params
);



#endif //MESH_HASHING_PREPROCESS_H
