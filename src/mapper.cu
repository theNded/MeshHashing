/// Input depth image as texture
/// Easier interpolation

#include "hash_table.h"
#include "sensor_data.h"

//////////
/// Bind texture
texture<float, cudaTextureType2D, cudaReadModeElementType> depthTextureRef;
texture<float4, cudaTextureType2D, cudaReadModeElementType> colorTextureRef;
__host__
void BindSensorDataToTextureCudaHost(const SensorData& sensor_data) {
  checkCudaErrors(cudaBindTextureToArray(depthTextureRef, sensor_data.d_depthArray, sensor_data.h_depthChannelDesc));
  checkCudaErrors(cudaBindTextureToArray(colorTextureRef, sensor_data.d_colorArray, sensor_data.h_colorChannelDesc));
  depthTextureRef.filterMode = cudaFilterModePoint;
  colorTextureRef.filterMode = cudaFilterModePoint;
}

//////////
/// Integrate depth map
__global__
void IntegrateCudaKernel(HashTable hash_table, SensorData cameraData, float4x4 c_T_w) {
  const HashParams& hash_params = kHashParams;
  const SensorParams& cameraParams = kSensorParams;

  //TODO check if we should load this in shared memory
  const HashEntry& entry = hash_table.compacted_hash_entries[blockIdx.x];

  int3 pi_base = BlockToVoxel(entry.pos);

  uint i = threadIdx.x;	//inside of an SDF block
  int3 pi = pi_base + make_int3(IdxToVoxelLocalPos(i));
  float3 pf = VoxelToWorld(pi);

  pf = c_T_w * pf;
  uint2 screenPos = make_uint2(CameraProjectToImagei(pf));


  if (screenPos.x < cameraParams.width && screenPos.y < cameraParams.height) {	//on screen

    //float depth = g_InputDepth[screenPos];
    float depth = tex2D(depthTextureRef, screenPos.x, screenPos.y);
    float4 color  = make_float4(MINF, MINF, MINF, MINF);
    if (cameraData.d_colorData) {
      color = tex2D(colorTextureRef, screenPos.x, screenPos.y);
      //color = bilinearFilterColor(cameraData.CameraProjectToImagef(pf));
    }

    if (color.x != MINF && depth != MINF) { // valid depth and color
      //if (depth != MINF) {	//valid depth

      if (depth < hash_params.sdf_upper_bound) {
        float depthZeroOne = NormalizeDepth(depth);

        float sdf = depth - pf.z;
        float truncation = truncate_distance(depth);
        if (sdf > -truncation) // && depthZeroOne >= 0.0f && depthZeroOne <= 1.0f) //check if in truncation range should already be made in depth map computation
        {
          if (sdf >= 0.0f) {
            sdf = fminf(truncation, sdf);
          } else {
            sdf = fmaxf(-truncation, sdf);
          }

          //float weightUpdate = g_WeightSample;
          //weightUpdate = (1-depthZeroOne)*5.0f + depthZeroOne*0.05f;
          //weightUpdate *= g_WeightSample;
          float weightUpdate = max(hash_params.weight_sample * 1.5f * (1.0f-depthZeroOne), 1.0f);

          Voxel curr;	//construct current voxel
          curr.sdf = sdf;
          curr.weight = weightUpdate;

          //float3 c = g_InputColor[screenPos].xyz;
          //curr.color = (int3)(c * 255.0f);
          if (cameraData.d_colorData) {
            //const float4& c = tex2D(colorTextureRef, screenPos.x, screenPos.y);
            curr.color = make_uchar3(255*color.x, 255*color.y, 255*color.z);
          } else {
            //TODO MATTHIAS make sure there is always consistent color data
            curr.color = make_uchar3(0,255,0);
          }

          uint idx = entry.ptr + i;

          Voxel newVoxel;
          //if (color.x == MINF) hash_table.combineVoxelDepthOnly(hash_table.blocks[idx], curr, newVoxel);
          //else hash_table.combineVoxel(hash_table.blocks[idx], curr, newVoxel);
          hash_table.combineVoxel(hash_table.blocks[idx], curr, newVoxel);
          hash_table.blocks[idx] = newVoxel;
          //Voxel prev = GetVoxel(g_SDFBlocksSDFUAV, g_SDFBlocksRGBWUAV, idx);
          //Voxel newVoxel = combineVoxel(curr, prev);
          //SetVoxel(g_SDFBlocksSDFUAV, g_SDFBlocksRGBWUAV, idx, newVoxel);
        }
      }
    }
  }
}

__host__
void IntegrateCudaHost(HashTable& hash_table, const HashParams& hash_params,
                       const SensorData& sensor_data, const SensorParams& depthCameraParams,
                       float4x4 c_T_w) {
  const unsigned int threadsPerBlock = SDF_BLOCK_SIZE*SDF_BLOCK_SIZE*SDF_BLOCK_SIZE;
  const dim3 gridSize(hash_params.occupied_block_count, 1);
  const dim3 blockSize(threadsPerBlock, 1);

  if (hash_params.occupied_block_count > 0) {	//this guard is important if there is no depth in the current frame (i.e., no blocks were allocated)
    IntegrateCudaKernel << <gridSize, blockSize >> >(hash_table, sensor_data, c_T_w);
  }
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());
}

