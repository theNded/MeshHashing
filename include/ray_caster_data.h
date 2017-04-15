//
// Created by wei on 17-3-17.
//

#ifndef MRF_VH_RAY_CASTER_DATA_H
#define MRF_VH_RAY_CASTER_DATA_H

#include "common.h"
#include <helper_cuda.h>

#include "hash_table.h"
#include "ray_caster_param.h"

#ifndef MINF
#define MINF __int_as_float(0xff800000)
#endif

/// constant.cu
extern __constant__ RayCastParams kRayCastParams;
extern void UpdateConstantRayCastParams(const RayCastParams &params);

struct RayCastSample {
  float sdf;
  float alpha;
  uint weight;
};

struct RayCasterData {
  ///////////////
  // Host part //
  ///////////////

  __device__ __host__
  RayCasterData() {
    d_depth = NULL;
    d_depth4 = NULL;
    d_normals = NULL;
    d_colors = NULL;
  }

  __host__
  void Alloc(const RayCastParams &params) {
    checkCudaErrors(cudaMalloc(&d_depth, sizeof(float) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&d_depth4, sizeof(float4) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&d_normals, sizeof(float4) * params.m_width * params.m_height));
    checkCudaErrors(cudaMalloc(&d_colors, sizeof(float4) * params.m_width * params.m_height));

    /// wei: should it happen here?
    //updateParams(params);
  }

  __host__
  void updateParams(const RayCastParams &params) {
    UpdateConstantRayCastParams(params);
  }

  __host__
  void Free() {
    checkCudaErrors(cudaFree(d_depth));
    checkCudaErrors(cudaFree(d_depth4));
    checkCudaErrors(cudaFree(d_normals));
    checkCudaErrors(cudaFree(d_colors));
  }

  /////////////////
  // Device part //
  /////////////////
#ifdef __CUDACC__
  __device__
    const RayCastParams& params() const {
      return kRayCastParams;
  }

  __device__
  float frac(float val) const {
    return (val - floorf(val));
  }
  __device__
  float3 frac(const float3& val) const {
      return make_float3(frac(val.x), frac(val.y), frac(val.z));
  }

  __device__
  bool trilinearInterpolationSimpleFastFast(const HashTable& hash, const float3& pos, float& dist, uchar3& color) const {
    const float oSet = kHashParams.voxel_size;
    const float3 posDual = pos-make_float3(oSet/2.0f, oSet/2.0f, oSet/2.0f);
      float3 weight = frac(WorldToVoxelf(pos));

    dist = 0.0f;
    float3 colorFloat = make_float3(0.0f, 0.0f, 0.0f);
    Voxel v = hash.GetVoxel(posDual+make_float3(0.0f, 0.0f, 0.0f)); if(v.weight == 0) return false; float3 vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*v.sdf; colorFloat+= (1.0f-weight.x)*(1.0f-weight.y)*(1.0f-weight.z)*vColor;
          v = hash.GetVoxel(posDual+make_float3(oSet, 0.0f, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*v.sdf; colorFloat+=	   weight.x *(1.0f-weight.y)*(1.0f-weight.z)*vColor;
          v = hash.GetVoxel(posDual+make_float3(0.0f, oSet, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*v.sdf; colorFloat+= (1.0f-weight.x)*	   weight.y *(1.0f-weight.z)*vColor;
          v = hash.GetVoxel(posDual+make_float3(0.0f, 0.0f, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *v.sdf; colorFloat+= (1.0f-weight.x)*(1.0f-weight.y)*	   weight.z *vColor;
          v = hash.GetVoxel(posDual+make_float3(oSet, oSet, 0.0f)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *	   weight.y *(1.0f-weight.z)*v.sdf; colorFloat+=	   weight.x *	   weight.y *(1.0f-weight.z)*vColor;
          v = hash.GetVoxel(posDual+make_float3(0.0f, oSet, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+= (1.0f-weight.x)*	   weight.y *	   weight.z *v.sdf; colorFloat+= (1.0f-weight.x)*	   weight.y *	   weight.z *vColor;
          v = hash.GetVoxel(posDual+make_float3(oSet, 0.0f, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *(1.0f-weight.y)*	   weight.z *v.sdf; colorFloat+=	   weight.x *(1.0f-weight.y)*	   weight.z *vColor;
          v = hash.GetVoxel(posDual+make_float3(oSet, oSet, oSet)); if(v.weight == 0) return false;		   vColor = make_float3(v.color.x, v.color.y, v.color.z); dist+=	   weight.x *	   weight.y *	   weight.z *v.sdf; colorFloat+=	   weight.x *	   weight.y *	   weight.z *vColor;

    color = make_uchar3(colorFloat.x, colorFloat.y, colorFloat.z);//v.color;

    return true;
  }


  __device__
  float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) const
  {
    return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
  }

  static const unsigned int nIterationsBisection = 3;

  // d0 near, d1 far
  __device__
    bool findIntersectionBisection(const HashTable& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) const {
    float a = r0; float aDist = d0;
    float b = r1; float bDist = d1;
    float c = 0.0f;

#pragma unroll 1
    for(uint i = 0; i<nIterationsBisection; i++) {
      c = findIntersectionLinear(a, b, aDist, bDist);

      float cDist;
      if(!trilinearInterpolationSimpleFastFast(hash, worldCamPos+c*worldDir, cDist, color)) return false;

      if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
      else { b = c; bDist = cDist; }
    }

    alpha = c;

    return true;
  }


  __device__
  float3 gradientForPoint(const HashTable& hash, const float3& pos) const {
    const float voxelSize = kHashParams.voxel_size;
    float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

    float distp00; uchar3 colorp00; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
    float dist0p0; uchar3 color0p0; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
    float dist00p; uchar3 color00p; trilinearInterpolationSimpleFastFast(hash, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

    float dist100; uchar3 color100; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100, color100);
    float dist010; uchar3 color010; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010, color010);
    float dist001; uchar3 color001; trilinearInterpolationSimpleFastFast(hash, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001, color001);

    float3 grad = make_float3((distp00-dist100)/offset.x, (dist0p0-dist010)/offset.y, (dist00p-dist001)/offset.z);

    float l = length(grad);
    if(l == 0.0f) {
      return make_float3(0.0f, 0.0f, 0.0f);
    }

    return -grad/l;
  }

  __device__
  void traverseCoarseGridSimpleSampleAll(const HashTable& hash, const float3& worldCamPos, const float3& worldDir, const float3& camDir, const int3& dTid, float minInterval, float maxInterval) const {
    int x = dTid.x, y = dTid.y;
    bool flag = (x == 589 && y == 477);

    const RayCastParams& rayCastParams = kRayCastParams;

    // Last Sample
    RayCastSample lastSample; lastSample.sdf = 0.0f; lastSample.alpha = 0.0f; lastSample.weight = 0; // lastSample.color = int3(0, 0, 0);
    const float depthToRayLength = 1.0f/camDir.z; // scale factor to convert from depth to ray length

    float rayCurrent = depthToRayLength * max(rayCastParams.m_minDepth, minInterval);	// Convert depth to raylength
    float rayEnd = depthToRayLength * min(rayCastParams.m_maxDepth, maxInterval);		// Convert depth to raylength
    if (flag) printf("%f %f\n", rayCurrent, rayEnd);
    //float rayCurrent = depthToRayLength * rayCastParams.m_minDepth;	// Convert depth to raylength
    //float rayEnd = depthToRayLength * rayCastParams.m_maxDepth;		// Convert depth to raylength
#pragma unroll 1
    while(rayCurrent < rayEnd)
    {
      float3 currentPosWorld = worldCamPos+rayCurrent*worldDir;
      float dist;	uchar3 color;

      if(trilinearInterpolationSimpleFastFast(hash, currentPosWorld, dist, color))
      {

        if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here
        {

          float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
          uchar3 color2;
          bool b = findIntersectionBisection(hash, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);

          float3 currentIso = worldCamPos+alpha*worldDir;
          if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
          {
            if(abs(dist) < rayCastParams.m_thresDist)
            {
              float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

              d_depth[dTid.y*rayCastParams.m_width+dTid.x] = depth;
              d_depth4[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(ImageReprojectToCamera(dTid.x, dTid.y, depth), 1.0f);
              d_colors[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(color2.x/255.f, color2.y/255.f, color2.z/255.f, 1.0f);

              if(rayCastParams.m_useGradients)
              {
                float3 normal = gradientForPoint(hash, currentIso);
                normal = - normal;
                float4 n = rayCastParams.m_viewMatrix * make_float4(normal, 0.0f);
                d_normals[dTid.y*rayCastParams.m_width+dTid.x] = make_float4(n.x, n.y, n.z, 1.0f);
              }

              return;
            }
          }
        }

        lastSample.sdf = dist;
        lastSample.alpha = rayCurrent;
        // lastSample.color = color;
        lastSample.weight = 1;
        rayCurrent += rayCastParams.m_rayIncrement;
      } else {
        lastSample.weight = 0;
        rayCurrent += rayCastParams.m_rayIncrement;
      }


    }

  }

#endif // __CUDACC__

  /// WRITE in ray_caster,
  /// specifically in traverseCoarseGridSimpleSampleAll
  float *d_depth;
  float4 *d_depth4;
  float4 *d_normals;
  float4 *d_colors;
};


#endif //MRF_VH_RAY_CASTER_DATA_H
