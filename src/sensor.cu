/// 16 threads per block
#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

#include <helper_cuda.h>
#include <helper_math.h>
#include <driver_types.h>

/// Input short* depth (cpu) to float* depth (gpu)
__global__ void convertDepthRawToFloatKernel(float *d_output, short *d_input,
                                          unsigned int width, unsigned int height,
                                          float minDepth, float maxDepth) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;
  /// Convert mm -> m
  const float depth = 0.0002f * d_input[idx]; // (1 / 5000)
  bool is_valid = (depth >= minDepth && depth <= maxDepth);
  d_output[idx] = is_valid ? depth : MINF;
}

void convertDepthRawToFloat(float* d_output, short* d_input,
                            unsigned int width, unsigned int height,
                            float minDepth, float maxDepth) {
  /// First copy cpu data in to cuda short
  short *cuda_input;
  checkCudaErrors(cudaMalloc(&cuda_input, sizeof(short) * width * height));
  checkCudaErrors(cudaMemcpy(cuda_input, d_input, sizeof(short) * width * height, cudaMemcpyHostToDevice));

  const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  convertDepthRawToFloatKernel<<<gridSize, blockSize>>>(d_output, cuda_input,
          width, height, minDepth, maxDepth);
}

///
/// Input uchar* color (cpu) to float4* color (gpu)
__global__ void convertColorRawToFloat4Kernel(float4* d_output, unsigned char *d_input,
                                        unsigned int width, unsigned int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) return;
  const int idx = y * width + x;

  uchar4 c = make_uchar4(d_input[4 * idx + 0], d_input[4 * idx + 1],
                         d_input[4 * idx + 2], d_input[4 * idx + 3]);
  bool is_valid = (c.x != 0 && c.y != 0 && c.z != 0);
  d_output[idx] = is_valid ? make_float4(c.x / 255.0f, c.y / 255.0f,
                                         c.z / 255.0f, c.w / 255.0f)
                           : make_float4(MINF, MINF, MINF, MINF);
}

void convertColorRawToFloat4(float4* d_output, unsigned char* d_input,
                             unsigned int width, unsigned int height) {
  unsigned char *cuda_input;
  checkCudaErrors(cudaMalloc(&cuda_input, 4 * sizeof(unsigned char) * width * height));
  checkCudaErrors(cudaMemcpy(cuda_input, d_input, 4 * sizeof(unsigned char) * width * height, cudaMemcpyHostToDevice));

  const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  convertColorRawToFloat4Kernel<<<gridSize, blockSize>>>(d_output, cuda_input, width, height);
}

/// Util: Depth to RGB
__device__ float3 convertHSVToRGB(const float3& hsv) {
  float H = hsv.x;
  float S = hsv.y;
  float V = hsv.z;

  float hd = H/60.0f;
  unsigned int h = (unsigned int)hd;
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

__device__ float3 convertDepthToRGB(float depth, float depthMin, float depthMax) {
  float depthZeroOne = (depth - depthMin)/(depthMax - depthMin);
  float x = 1.0f-depthZeroOne;
  if (x < 0.0f)	x = 0.0f;
  if (x > 1.0f)	x = 1.0f;

  x = 360.0f*x - 120.0f;
  if (x < 0.0f) x += 359.0f;
  return convertHSVToRGB(make_float3(x, 1.0f, 0.5f));
}

__global__ void depthToHSVDevice(float4* d_output, float* d_input,
                                 unsigned int width, unsigned int height,
                                 float minDepth, float maxDepth) {
  const int x = blockIdx.x*blockDim.x + threadIdx.x;
  const int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= 0 && x < width && y >= 0 && y < height) {

    float depth = d_input[y*width + x];
    if (depth != MINF && depth != 0.0f && depth >= minDepth && depth <= maxDepth) {
      float3 c = convertDepthToRGB(depth, minDepth, maxDepth);
      d_output[y*width + x] = make_float4(c, 1.0f);
    } else {
      d_output[y*width + x] = make_float4(0.0f);
    }
  }
}

void depthToHSV(float4* d_output, float* d_input, unsigned int width, unsigned int height, float minDepth, float maxDepth) {
  const dim3 gridSize((width + T_PER_BLOCK - 1)/T_PER_BLOCK, (height + T_PER_BLOCK - 1)/T_PER_BLOCK);
  const dim3 blockSize(T_PER_BLOCK, T_PER_BLOCK);

  depthToHSVDevice<<<gridSize, blockSize>>>(d_output, d_input, width, height, minDepth, maxDepth);
}