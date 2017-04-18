#include <matrix.h>

#include "sensor_data.h"
#include "hash_table.h"
#include "ray_caster.h"
#include "ray_caster_data.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024

__device__
inline float frac(float val) {
  return (val - floorf(val));
}
__device__
inline float3 frac(const float3& val)  {
  return make_float3(frac(val.x), frac(val.y), frac(val.z));
}

__device__
bool TrilinearInterpolation(const HashTable& hash, const float3& pos, float& dist, uchar3& color) {
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
float findIntersectionLinear(float tNear, float tFar, float dNear, float dFar) {
  return tNear+(dNear/(dNear-dFar))*(tFar-tNear);
}

static const unsigned int nIterationsBisection = 3;

// d0 near, d1 far
__device__
bool findIntersectionBisection(const HashTable& hash, const float3& worldCamPos, const float3& worldDir, float d0, float r0, float d1, float r1, float& alpha, uchar3& color) {
  float a = r0; float aDist = d0;
  float b = r1; float bDist = d1;
  float c = 0.0f;

#pragma unroll 1
  for(uint i = 0; i<nIterationsBisection; i++) {
    c = findIntersectionLinear(a, b, aDist, bDist);

    float cDist;
    if(!TrilinearInterpolation(hash, worldCamPos+c*worldDir, cDist, color)) return false;

    if(aDist*cDist > 0.0) { a = c; aDist = cDist; }
    else { b = c; bDist = cDist; }
  }

  alpha = c;

  return true;
}


__device__
float3 GradientAtPoint(const HashTable& hash, const float3& pos) {
  const float voxelSize = kHashParams.voxel_size;
  float3 offset = make_float3(voxelSize, voxelSize, voxelSize);

  float distp00; uchar3 colorp00; TrilinearInterpolation(hash, pos-make_float3(0.5f*offset.x, 0.0f, 0.0f), distp00, colorp00);
  float dist0p0; uchar3 color0p0; TrilinearInterpolation(hash, pos-make_float3(0.0f, 0.5f*offset.y, 0.0f), dist0p0, color0p0);
  float dist00p; uchar3 color00p; TrilinearInterpolation(hash, pos-make_float3(0.0f, 0.0f, 0.5f*offset.z), dist00p, color00p);

  float dist100; uchar3 color100; TrilinearInterpolation(hash, pos+make_float3(0.5f*offset.x, 0.0f, 0.0f), dist100, color100);
  float dist010; uchar3 color010; TrilinearInterpolation(hash, pos+make_float3(0.0f, 0.5f*offset.y, 0.0f), dist010, color010);
  float dist001; uchar3 color001; TrilinearInterpolation(hash, pos+make_float3(0.0f, 0.0f, 0.5f*offset.z), dist001, color001);

  float3 grad = make_float3((distp00-dist100)/offset.x, (dist0p0-dist010)/offset.y, (dist00p-dist001)/offset.z);

  float l = length(grad);
  if(l == 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }

  return -grad/l;
}

__global__ void CastKernel(const HashTable hash_table, RayCasterData rayCasterData) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  const RayCasterParams& rayCastParams = kRayCasterParams;

  //const HashParams& hash_params = kHashParams;

  if (x < rayCastParams.m_width && y < rayCastParams.m_height) {
    rayCasterData.depth_image_[y*rayCastParams.m_width+x] = MINF;
    rayCasterData.vertex_image_[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
    rayCasterData.normal_image_[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
    rayCasterData.color_image_[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);

    /// WRONG! this uses the sensor's parameter instead of the viewer's
    float3 camDir = normalize(ImageReprojectToCamera(x, y, 1.0f));
    float3 worldCamPos = rayCastParams.w_T_c * make_float3(0.0f, 0.0f, 0.0f);
    float4 w = rayCastParams.w_T_c * make_float4(camDir, 0.0f);
    float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

    float minInterval = rayCastParams.m_minDepth;
    float maxInterval = rayCastParams.m_maxDepth;

    bool flag = (x == 589 && y == 477);

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

      if(TrilinearInterpolation(hash_table, currentPosWorld, dist, color))
      {

        if(lastSample.weight > 0 && lastSample.sdf > 0.0f && dist < 0.0f) // current sample is always valid here
        {

          float alpha; // = findIntersectionLinear(lastSample.alpha, rayCurrent, lastSample.sdf, dist);
          uchar3 color2;
          bool b = findIntersectionBisection(hash_table, worldCamPos, worldDir, lastSample.sdf, lastSample.alpha, dist, rayCurrent, alpha, color2);

          float3 currentIso = worldCamPos+alpha*worldDir;
          if(b && abs(lastSample.sdf - dist) < rayCastParams.m_thresSampleDist)
          {
            if(abs(dist) < rayCastParams.m_thresDist)
            {
              float depth = alpha / depthToRayLength; // Convert ray length to depth depthToRayLength

              rayCasterData.depth_image_[y*rayCastParams.m_width+x] = depth;
              rayCasterData.vertex_image_[y*rayCastParams.m_width+x] = make_float4(ImageReprojectToCamera(x, y, depth), 1.0f);
              rayCasterData.color_image_[y*rayCastParams.m_width+x] = make_float4(color2.x/255.f, color2.y/255.f, color2.z/255.f, 1.0f);

              if(rayCastParams.m_useGradients)
              {
                float3 normal = GradientAtPoint(hash_table, currentIso);
                normal = - normal;
                float4 n = rayCastParams.c_T_w * make_float4(normal, 0.0f);
                rayCasterData.normal_image_[y*rayCastParams.m_width+x] = make_float4(n.x, n.y, n.z, 1.0f);
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
}

__host__
void CastCudaHost(const HashTable        &hash_table,   const RayCasterData   &rayCastData,
              const RayCasterParams &rayCastParams) {

  const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1)/T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  CastKernel<<<gridSize, blockSize>>>(hash_table, rayCastData);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}