#include <matrix.h>
#include <glog/logging.h>
#include "color_util.h"

#include "geometry/geometry_helper.h"
#include "geometry/gradient.h"
#include "visualization/ray_caster.h"

//////////
/// Device code required by kernel functions


//////////
/// Kernel function
__global__
void CastKernel(const HashTable hash_table,
                const BlockArray    blocks,
                RayCasterData  ray_caster_data,
                const RayCasterParams ray_caster_params,
                const float4x4 c_T_w,
                const float4x4 w_T_c,
                GeometryHelper geoemtry_helper) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= ray_caster_params.width || y >= ray_caster_params.height)
    return;


  int pixel_idx = y * ray_caster_params.width + x;
  ray_caster_data.depth [pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  ray_caster_data.vertex[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  ray_caster_data.normal[pixel_idx] = make_float4(MINF, MINF, MINF, MINF);
  ray_caster_data.color [pixel_idx] = make_float4(1, 1, 1, 1);
  ray_caster_data.surface[pixel_idx] = make_float4(0, 0, 0, 0);

  /// 1. Determine ray direction
  float3 camera_ray_dir = normalize(
          geoemtry_helper.ImageReprojectToCamera(x, y, 1.0f,
                                 ray_caster_params.fx, ray_caster_params.fy,
                                 ray_caster_params.cx, ray_caster_params.cy));

  float t_min = ray_caster_params.min_raycast_depth / camera_ray_dir.z;
  float t_max = ray_caster_params.max_raycast_depth / camera_ray_dir.z;

  float3 world_cam_pos = w_T_c * make_float3(0.0f);
  float4 world_ray_dir_homo = w_T_c * make_float4(camera_ray_dir, 0.0f);
  float3 world_ray_dir = normalize(make_float3(world_ray_dir_homo.x,
                                               world_ray_dir_homo.y,
                                               world_ray_dir_homo.z));

  RayCasterSample prev_sample;
  prev_sample.entropy = 0;
  prev_sample.sdf = 0.0f;
  prev_sample.t = 0.0f;
  prev_sample.weight = 0;

  /// NOT zig-zag; just evenly sampling along the ray
  // TODO: Check this weird case : return causes illegal address failure
  bool return_flag = false;

#pragma unroll 1
  for (float t = t_min; t < t_max & !return_flag; t += ray_caster_params.raycast_step) {
    float3 world_sample_pos = world_cam_pos + t * world_ray_dir;
    float  sdf;
    uchar3 color;
#ifdef STATS
    Stat stats;
#endif
    /// a voxel surrounded by valid voxels
    if (  TrilinearInterpolation(hash_table, blocks, world_sample_pos,
                               sdf,
#ifdef STATS
                                 stats,
#endif
                                 color, geoemtry_helper)) {

      /// Zero crossing exist
      if (prev_sample.weight > 0 // valid previous sample
          && prev_sample.sdf > 0.0f && sdf < 0.0f) { // zero-crossing

        float interpolated_t;
        uchar3 interpolated_color;
        /// Find exact zero crossing
        bool is_isosurface_found = BisectionIntersection(
                hash_table, blocks,
                world_cam_pos, world_ray_dir,
                prev_sample.sdf, prev_sample.t,
                sdf, t,
                interpolated_t, interpolated_color, geoemtry_helper);

        float3 world_pos_isosurface =
                world_cam_pos + interpolated_t * world_ray_dir;

        /// Good enough sample
        if (is_isosurface_found
            && abs(prev_sample.sdf - sdf) < ray_caster_params.sample_sdf_threshold) {
          /// Trick from the original author of voxel-hashing
          if (abs(sdf) < ray_caster_params.sdf_threshold) {
            float depth = interpolated_t * camera_ray_dir.z;

            float3 rgb = ValToRGB(depth, 0.3, 5.0);
            //printf("%f %f %f\n", rgb.x, rgb.y, rgb.z);
            ray_caster_data.depth[pixel_idx] = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
            //break;
            ray_caster_data.vertex[pixel_idx]
                    = make_float4(geoemtry_helper.ImageReprojectToCamera(x, y, depth,
                                                         ray_caster_params.fx,
                                                         ray_caster_params.fy,
                                                         ray_caster_params.cx,
                                                         ray_caster_params.cy), 1.0f);

            ray_caster_data.color [pixel_idx]
                    = make_float4(interpolated_color.x / 255.f,
                                  interpolated_color.y / 255.f,
                                  interpolated_color.z / 255.f, 1.0f);

            if (ray_caster_params.enable_gradients) {
              float3 normal = GradientAtPoint(hash_table, blocks, world_pos_isosurface, geoemtry_helper);
              normal = -normal;
              float4 n = c_T_w * make_float4(normal, 0.0f);
              ray_caster_data.normal[pixel_idx]
                      = make_float4(n.x, n.y, n.z, 1.0f);

              /// Shading here
              // TODO(wei): light position uniform
              float3 light_pos = make_float3(0, -2, -3);
              float3 n3 = normalize(make_float3(normal.x, normal.y, normal.z));
              float3 l3 = normalize(light_pos - world_pos_isosurface);
              float distance = length(light_pos - world_pos_isosurface);
              // bgr
              float3 c3 = make_float3(0.62f, 0.72f, 0.88) * dot(-n3, l3)
                          * 20.0f / (distance * distance);

              // Uncertainty: low -> entropy high
              //c3 = ValToRGB(stats.entropy, 0.0f, 1.0f);
              ray_caster_data.surface[pixel_idx]
                      = make_float4(c3.x, c3.y, c3.z, 1.0f);
            }

            return_flag = true;
          }
        }
      }

      /// No zero crossing || not good
      prev_sample.sdf = sdf;
#ifdef STATS
      prev_sample.entropy = stats.entropy;
#endif
      prev_sample.t = t;
      prev_sample.weight = 1;
    }
  }
}

/// Member function: (CPU code)
RayCaster::RayCaster(const RayCasterParams &params) {
  Alloc(params);
}

void RayCaster::Alloc(const RayCasterParams &params) {
  if (! is_allocated_on_gpu_) {
    ray_caster_params_ = params;
    uint image_size = params.width * params.height;
    checkCudaErrors(cudaMalloc(&ray_caster_data_.depth, sizeof(float4) * image_size));
    checkCudaErrors(cudaMalloc(&ray_caster_data_.vertex, sizeof(float4) * image_size));
    checkCudaErrors(cudaMalloc(&ray_caster_data_.normal, sizeof(float4) * image_size));
    checkCudaErrors(cudaMalloc(&ray_caster_data_.color, sizeof(float4) * image_size));
    checkCudaErrors(cudaMalloc(&ray_caster_data_.surface, sizeof(float4) * image_size));

    depth_image_ = cv::Mat(params.height, params.width, CV_32FC4);
    normal_image_ = cv::Mat(params.height, params.width, CV_32FC4);
    color_image_ = cv::Mat(params.height, params.width, CV_32FC4);
    surface_image_ = cv::Mat(params.height, params.width, CV_32FC4);

    is_allocated_on_gpu_ = true;
  }
}

void RayCaster::Free() {
  if (is_allocated_on_gpu_) {
    checkCudaErrors(cudaFree(ray_caster_data_.depth));
    checkCudaErrors(cudaFree(ray_caster_data_.vertex));
    checkCudaErrors(cudaFree(ray_caster_data_.normal));
    checkCudaErrors(cudaFree(ray_caster_data_.color));
    checkCudaErrors(cudaFree(ray_caster_data_.surface));
    is_allocated_on_gpu_ = false;
  }
}

//////////
/// Member function: (CPU calling GPU kernels)
/// Major function, extract surface and normal from the volumes
void RayCaster::Cast(HashTable& hash_table, BlockArray& blocks,
                     RayCasterData& ray_caster_data,
                     GeometryHelper& geoemtry_helper,
                     const float4x4& c_T_w) {
  const uint threads_per_block = 8;
  const float4x4 w_T_c = c_T_w.getInverse();

  const dim3 grid_size((ray_caster_params_.width + threads_per_block - 1)
                       /threads_per_block,
                       (ray_caster_params_.height + threads_per_block - 1)
                       /threads_per_block);
  const dim3 block_size(threads_per_block, threads_per_block);

  CastKernel<<<grid_size, block_size>>>(
      hash_table,
          blocks,
          ray_caster_data,
          ray_caster_params_, c_T_w, w_T_c, geoemtry_helper);
  checkCudaErrors(cudaDeviceSynchronize());
  checkCudaErrors(cudaGetLastError());

  uint image_size = ray_caster_params_.height * ray_caster_params_.width;
  checkCudaErrors(cudaMemcpy(depth_image_.data, ray_caster_data.depth,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(normal_image_.data, ray_caster_data.normal,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(color_image_.data, ray_caster_data.color,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(surface_image_.data, ray_caster_data_.surface,
                             sizeof(float) * 4 * image_size,
                             cudaMemcpyDeviceToHost));
}