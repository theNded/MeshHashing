//
// Created by wei on 17-6-6.
//

#ifndef VH_COLOR_UTIL_H
#define VH_COLOR_UTIL_H

#include "common.h"
#include "helper_math.h"

// http://paulbourke.net/texture_colour/colourspace/
__host__ __device__
inline float3 ValToRGB(float v, const float vmin, const float vmax) {
  float3 c = make_float3(1.0f);

  float dv;
  if (v < vmin)
    v = vmin;
  if (v > vmax)
    v = vmax;
  dv = vmax - vmin;

  if (v < (vmin + 0.25 * dv)) {
    c.x = 0;
    c.y = 4 * (v - vmin) / dv;
  } else if (v < (vmin + 0.5 * dv)) {
    c.x = 0;
    c.z = 1 + 4 * (vmin + 0.25 * dv - v) / dv;
  } else if (v < (vmin + 0.75 * dv)) {
    c.x = 4 * (v - vmin - 0.5 * dv) / dv;
    c.z = 0;
  } else {
    c.y = 1 + 4 * (vmin + 0.75 * dv - v) / dv;
    c.z = 0;
  }
  return c;
};

/// Util: Depth to RGB
__device__
inline float3 HSVToRGB(const float3& hsv) {
  float H = hsv.x;
  float S = hsv.y;
  float V = hsv.z;

  float hd = H/60.0f;
  uint h = (uint)hd;
  float f = hd-h;

  float p = V*(1.0f-S);
  float q = V*(1.0f-S*f);
  float t = V*(1.0f-S*(1.0f-f));

  if(h == 0 || h == 6) {
    return make_float3(V, t, p);
  }
  else if(h == 1) {
    return make_float3(q, V, p);
  }
  else if(h == 2) {
    return make_float3(p, V, t);
  }
  else if(h == 3) {
    return make_float3(p, q, V);
  }
  else if(h == 4) {
    return make_float3(t, p, V);
  } else {
    return make_float3(V, p, q);
  }
}


#endif //VH_COLOR_UTIL_H
