#include <matrix.h>
#include <geometry_util.h>

#include "gradient.h"
#include "ray_caster.h"

//////////
/// Device code required by kernel functions


//////////
/// Kernel function
__global__
void CastKernel(const HashTableGPU hash_table,
                Block *blocks,
                RayCasterDataGPU gpu_data,
                RayCasterParams ray_caster_params,
                const float4x4 c_T_w,
                const float4x4 w_T_c) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= ray_caster_params.width || y >= ray_caster_params.height)
    return;

  int pixel_idx = y * ray_caster_params.width + x;
  gpu_data.depth_image [pixel_idx] = MINF;
  gpu_data.vertex_image[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  gpu_data.normal_image[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  gpu_data.color_image [pixel_idx] = make_float4(1, 1, 1, 1);
  gpu_data.surface_image[pixel_idx] = make_float4(0, 0, 0, 0);

  /// 1. Determine ray direction
  float3 camera_dir = normalize(ImageReprojectToCamera(x, y, 1.0f,
                                                       ray_caster_params.fx,
                                                       ray_caster_params.fy,
                                                       ray_caster_params.cx,
                                                       ray_caster_params.cy));

  float ray_length_per_depth_unit = 1.0f / camera_dir.z;
  float t_min = ray_caster_params.min_raycast_depth / camera_dir.z;
  float t_max = ray_caster_params.max_raycast_depth / camera_dir.z;

  float3 world_pos_camera_origin = w_T_c * make_float3(0.0f, 0.0f, 0.0f);
  float4 world_dir_homo = w_T_c * make_float4(camera_dir, 0.0f);
  float3 world_dir = normalize(make_float3(world_dir_homo.x,
                                           world_dir_homo.y,
                                           world_dir_homo.z));

  RayCasterSample prev_sample;
  prev_sample.sdf = 0.0f;
  prev_sample.t = 0.0f;
  prev_sample.weight = 0;

  /// NOT zig-zag; just evenly sampling along the ray
#pragma unroll 1
  for (float t = t_min; t < t_max; t += ray_caster_params.raycast_step) {
    float3 world_pos_sample = world_pos_camera_origin + t * world_dir;
    float sdf;
    uchar3 color;

    if (TrilinearInterpolation(hash_table, blocks, world_pos_sample,
                               sdf, color)) {
      /// Zero crossing
      if (prev_sample.weight > 0 && prev_sample.sdf > 0.0f && sdf < 0.0f) {

        float interpolated_t;
        uchar3 interpolated_color;
        /// Find isosurface
        bool is_isosurface_found = BisectionIntersection(
                hash_table, blocks,
                world_pos_camera_origin, world_dir,
                prev_sample.sdf, prev_sample.t, sdf, t,
                interpolated_t, interpolated_color);

        float3 world_pos_isosurface =
                world_pos_camera_origin + interpolated_t * world_dir;

        /// Good enough sample
        if (is_isosurface_found
            && abs(prev_sample.sdf - sdf) < ray_caster_params.sample_sdf_threshold) {
          /// Trick from the original author of voxel-hashing
          if (abs(sdf) < ray_caster_params.sdf_threshold) {
            float depth = interpolated_t / ray_length_per_depth_unit;

            gpu_data.depth_image [pixel_idx] = depth;
            gpu_data.vertex_image[pixel_idx]
                    = make_float4(ImageReprojectToCamera(
                    x, y, depth,
                    ray_caster_params.fx,
                    ray_caster_params.fy,
                    ray_caster_params.cx,
                    ray_caster_params.cy), 1.0f);

            gpu_data.color_image [pixel_idx]
                    = make_float4(interpolated_color.x / 255.f,
                                  interpolated_color.y / 255.f,
                                  interpolated_color.z / 255.f, 1.0f);

            if (ray_caster_params.enable_gradients) {
              float3 normal = GradientAtPoint(hash_table, blocks, world_pos_isosurface);
              normal = -normal;
              float4 n = c_T_w * make_float4(normal, 0.0f);
              gpu_data.normal_image[pixel_idx]
                      = make_float4(n.x, n.y, n.z, 1.0f);

              /// Shading here
              // TODO(wei): light position uniform
              float3 light_pos = make_float3(0, -2, -3);
              float3 n3 = normalize(make_float3(normal.x, normal.y, normal.z));
              float3 l3 = normalize(light_pos - world_pos_isosurface);
              float distance = length(light_pos - world_pos_isosurface);
              float3 c3 = make_float3(0.62f, 0.72f, 0.88) * dot(-n3, l3)
                          * 20.0f / (distance * distance);
              gpu_data.surface_image[pixel_idx]
                      = make_float4(c3, 1.0f);
            }

            return;
          }
        }
      }

      /// No zero crossing || not good
      prev_sample.sdf = sdf;
      prev_sample.t = t;
      prev_sample.weight = 1;
    }
  }
}

/// Member function: (CPU code)
RayCaster::RayCaster(const RayCasterParams& params) {
  ray_caster_params_ = params;
  uint image_size = params.width * params.height;
  checkCudaErrors(cudaMalloc(&gpu_data_.depth_image, sizeof(float) * image_size));
  checkCudaErrors(cudaMalloc(&gpu_data_.vertex_image, sizeof(float4) * image_size));
  checkCudaErrors(cudaMalloc(&gpu_data_.normal_image, sizeof(float4) * image_size));
  checkCudaErrors(cudaMalloc(&gpu_data_.color_image, sizeof(float4) * image_size));
  checkCudaErrors(cudaMalloc(&gpu_data_.surface_image, sizeof(float4) * image_size));

  normal_image_ = cv::Mat(params.height, params.width, CV_32FC4);
  color_image_  = cv::Mat(params.height, params.width, CV_32FC4);
  surface_image_ = cv::Mat(params.height, params.width, CV_32FC4);
}

RayCaster::~RayCaster() {
  checkCudaErrors(cudaFree(gpu_data_.depth_image));
  checkCudaErrors(cudaFree(gpu_data_.vertex_image));
  checkCudaErrors(cudaFree(gpu_data_.normal_image));
  checkCudaErrors(cudaFree(gpu_data_.color_image));
  checkCudaErrors(cudaFree(gpu_data_.surface_image));
}

//////////
/// Member function: (CPU calling GPU kernels)
/// Major function, extract surface and normal from the volumes
void RayCaster::Cast(Map& map, const float4x4& c_T_w) {
  const uint threads_per_block = 8;
  const float4x4 w_T_c = c_T_w.getInverse();

  const dim3 grid_size((ray_caster_params_.width + threads_per_block - 1)
                       /threads_per_block,
                       (ray_caster_params_.height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  CastKernel<<<grid_size, block_size>>>(
          map.hash_table().gpu_data(),
                  map.blocks().gpu_data(),
                  gpu_data_, ray_caster_params_, c_T_w, w_T_c);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  uint image_size = ray_caster_params_.height * ray_caster_params_.width;
  checkCudaErrors(cudaMemcpy(normal_image_.data, gpu_data_.normal_image,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(color_image_.data, gpu_data_.color_image,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(surface_image_.data, gpu_data_.surface_image,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));

}