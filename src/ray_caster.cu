#include <matrix.h>

#include "sensor_data.h"
#include "hash_table.h"
#include "ray_caster.h"
#include "ray_caster_data.h"

#define T_PER_BLOCK 8
#define NUM_GROUPS_X 1024

__global__ void renderKernel(const HashTable HashTable, const RayCastData rayCastData, const DepthCameraData cameraData) {
  const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

  const RayCastParams& rayCastParams = c_rayCastParams;

  const HashParams& hashParams = kHashParams;

  if (x < rayCastParams.m_width && y < rayCastParams.m_height) {
    rayCastData.d_depth[y*rayCastParams.m_width+x] = MINF;
    rayCastData.d_depth4[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
    rayCastData.d_normals[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);
    rayCastData.d_colors[y*rayCastParams.m_width+x] = make_float4(MINF,MINF,MINF,MINF);

    float3 camDir = normalize(cameraData.kinectProjToCamera(x, y, 1.0f));
    float3 worldCamPos = rayCastParams.m_viewMatrixInverse * make_float3(0.0f, 0.0f, 0.0f);
    float4 w = rayCastParams.m_viewMatrixInverse * make_float4(camDir, 0.0f);
    float3 worldDir = normalize(make_float3(w.x, w.y, w.z));

    ////use ray interval splatting
    //float minInterval = tex2D(rayMinTextureRef, x, y);
    //float maxInterval = tex2D(rayMaxTextureRef, x, y);

    //don't use ray interval splatting
    float minInterval = rayCastParams.m_minDepth;
    float maxInterval = rayCastParams.m_maxDepth;

    //if (minInterval == 0 || minInterval == MINF) minInterval = rayCastParams.m_minDepth;
    //if (maxInterval == 0 || maxInterval == MINF) maxInterval = rayCastParams.m_maxDepth;
    //TODO MATTHIAS: shouldn't this return in the case no interval is found?
    //if (minInterval == 0 || minInterval == MINF) return;
    //if (maxInterval == 0 || maxInterval == MINF) return;

    // debugging
    //if (maxInterval < minInterval) {
    //	printf("ERROR (%d,%d): [ %f, %f ]\n", x, y, minInterval, maxInterval);
    //}

    rayCastData.traverseCoarseGridSimpleSampleAll(HashTable, cameraData, worldCamPos, worldDir, camDir, make_int3(x,y,1), minInterval, maxInterval);
  }
}

void renderCS(const HashTable        &HashTable,   const RayCastData   &rayCastData,
              const DepthCameraData &cameraData, const RayCastParams &rayCastParams) {

  const dim3 gridSize((rayCastParams.m_width + T_PER_BLOCK - 1)/T_PER_BLOCK, (rayCastParams.m_height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  renderKernel<<<gridSize, blockSize>>>(HashTable, rayCastData, cameraData);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}