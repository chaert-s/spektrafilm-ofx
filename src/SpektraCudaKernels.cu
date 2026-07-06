#include "SpektraCudaKernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace spektrafilm {
namespace {

constexpr uint32_t kColorAdaptationInputCompression = 1u << 0u;
constexpr uint32_t kColorAdaptationCurveSmoothing = 1u << 1u;
constexpr uint32_t kColorAdaptationOutputLightnessCompression = 1u << 2u;
constexpr uint32_t kColorAdaptationOutputChromaCompression = 1u << 3u;
constexpr uint32_t kOutputGamutCompressionStride = 18u;

// host-facing error text stays here so launch wrappers remain small
void setError(char *error, size_t errorSize, const char *message, cudaError_t status = cudaSuccess) {
  if (!error || errorSize == 0u) {
    return;
  }
  if (status == cudaSuccess) {
    std::snprintf(error, errorSize, "%s", message ? message : "");
  } else {
    std::snprintf(error, errorSize, "%s: %s", message ? message : "CUDA error", cudaGetErrorString(status));
  }
}

// Host image bridge: OFX rowBytes/origin/half around internal float RGBA.
__global__ void copyFloatKernel(const float *source, float *destination, size_t floatCount) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < floatCount) {
    destination[index] = source[index];
  }
}

__global__ void packDeviceImageToFloatKernel(
  const void *source,
  int sourceX1,
  int sourceY1,
  int sourceRowBytes,
  int sourceBytesPerComponent,
  int windowX1,
  int windowY1,
  int width,
  int height,
  float *destination
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const int imageX = windowX1 + x;
  const int imageY = windowY1 + y;
  const auto *row = static_cast<const char *>(source) +
    static_cast<ptrdiff_t>(imageY - sourceY1) * sourceRowBytes;
  const auto *pixel = row +
    static_cast<ptrdiff_t>(imageX - sourceX1) * 4 * sourceBytesPerComponent;
  const size_t output = static_cast<size_t>(index) * 4u;
  if (sourceBytesPerComponent == 4) {
    const auto *rgba = reinterpret_cast<const float *>(pixel);
    destination[output] = rgba[0];
    destination[output + 1u] = rgba[1];
    destination[output + 2u] = rgba[2];
    destination[output + 3u] = rgba[3];
  } else {
    const auto *rgba = reinterpret_cast<const __half *>(pixel);
    destination[output] = __half2float(rgba[0]);
    destination[output + 1u] = __half2float(rgba[1]);
    destination[output + 2u] = __half2float(rgba[2]);
    destination[output + 3u] = __half2float(rgba[3]);
  }
}

__global__ void unpackFloatToDeviceImageKernel(
  const float *source,
  void *destination,
  int destinationX1,
  int destinationY1,
  int destinationRowBytes,
  int destinationBytesPerComponent,
  int windowX1,
  int windowY1,
  int width,
  int height
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const int imageX = windowX1 + x;
  const int imageY = windowY1 + y;
  auto *row = static_cast<char *>(destination) +
    static_cast<ptrdiff_t>(imageY - destinationY1) * destinationRowBytes;
  auto *pixel = row +
    static_cast<ptrdiff_t>(imageX - destinationX1) * 4 * destinationBytesPerComponent;
  const size_t input = static_cast<size_t>(index) * 4u;
  if (destinationBytesPerComponent == 4) {
    auto *rgba = reinterpret_cast<float *>(pixel);
    rgba[0] = source[input];
    rgba[1] = source[input + 1u];
    rgba[2] = source[input + 2u];
    rgba[3] = source[input + 3u];
  } else {
    auto *rgba = reinterpret_cast<__half *>(pixel);
    rgba[0] = __float2half_rn(source[input]);
    rgba[1] = __float2half_rn(source[input + 1u]);
    rgba[2] = __float2half_rn(source[input + 2u]);
    rgba[3] = __float2half_rn(source[input + 3u]);
  }
}

__device__ float clampf(float value, float lo, float hi) {
  return fminf(fmaxf(value, lo), hi);
}

__device__ float mixf(float a, float b, float t) {
  return a + (b - a) * t;
}

__device__ uint32_t minu32(uint32_t a, uint32_t b) {
  return a < b ? a : b;
}

__device__ float mitchellWeight(float t) {
  constexpr float b = 1.0f / 3.0f;
  constexpr float c = 1.0f / 3.0f;
  const float x = fabsf(t);
  if (x < 1.0f) {
    return (1.0f / 6.0f) * ((12.0f - 9.0f * b - 6.0f * c) * x * x * x +
                            (-18.0f + 12.0f * b + 6.0f * c) * x * x +
                            (6.0f - 2.0f * b));
  }
  if (x < 2.0f) {
    return (1.0f / 6.0f) * ((-b - 6.0f * c) * x * x * x +
                            (6.0f * b + 30.0f * c) * x * x +
                            (-12.0f * b - 48.0f * c) * x +
                            (8.0f * b + 24.0f * c));
  }
  return 0.0f;
}

__device__ uint32_t mirroredIndex(int index, uint32_t size) {
  if (size <= 1u) {
    return 0u;
  }
  const int period = static_cast<int>(size) * 2 - 2;
  int mirrored = index % period;
  if (mirrored < 0) {
    mirrored += period;
  }
  if (mirrored >= static_cast<int>(size)) {
    mirrored = period - mirrored;
  }
  return static_cast<uint32_t>(mirrored);
}

__device__ uint32_t colorSpaceIndex(int colorSpace, const KernelColorInfo &colorInfo) {
  if (colorSpace < 0 || colorSpace >= static_cast<int>(colorInfo.colorSpaceCount)) {
    return 0u;
  }
  return static_cast<uint32_t>(colorSpace);
}

__device__ bool colorAdaptationEnabled(const KernelParams &params, uint32_t flag) {
  return (params.colorAdaptationFlags & flag) != 0u;
}

__device__ float sampleTransferLut(float value, uint32_t colorSpace, const KernelColorInfo &colorInfo, const float *lut) {
  const uint32_t lutSize = colorInfo.transferLutSize;
  if (!lut || lutSize <= 1u) {
    return value;
  }
  const uint32_t offset = colorSpace * lutSize;
  const float range = fmaxf(colorInfo.decodeMax - colorInfo.decodeMin, 1.0e-6f);
  const float step = range / static_cast<float>(lutSize - 1u);
  if (value <= colorInfo.decodeMin) {
    const float y0 = lut[offset];
    const float y1 = lut[offset + 1u];
    return y0 + (value - colorInfo.decodeMin) * ((y1 - y0) / fmaxf(step, 1.0e-12f));
  }
  if (value >= colorInfo.decodeMax) {
    const float y0 = lut[offset + lutSize - 2u];
    const float y1 = lut[offset + lutSize - 1u];
    return y1 + (value - colorInfo.decodeMax) * ((y1 - y0) / fmaxf(step, 1.0e-12f));
  }
  const float t = (value - colorInfo.decodeMin) / range;
  const float position = t * static_cast<float>(lutSize - 1u);
  const uint32_t lo = static_cast<uint32_t>(floorf(position));
  const uint32_t hi = minu32(lo + 1u, lutSize - 1u);
  const float f = position - static_cast<float>(lo);
  return mixf(lut[offset + lo], lut[offset + hi], f);
}

__device__ float3 decodeInputRgb(float3 rgb, const KernelParams &params, const KernelColorInfo &colorInfo, const float *lut, const uint32_t *transferKind) {
  const uint32_t colorSpace = colorSpaceIndex(params.inputColorSpace, colorInfo);
  if (!transferKind || transferKind[colorSpace] == 0u) {
    return rgb;
  }
  return make_float3(
    sampleTransferLut(rgb.x, colorSpace, colorInfo, lut),
    sampleTransferLut(rgb.y, colorSpace, colorInfo, lut),
    sampleTransferLut(rgb.z, colorSpace, colorInfo, lut)
  );
}

__device__ float3 mulColorMatrix(float3 rgb, int colorSpace, const KernelColorInfo &colorInfo, const float *matrix) {
  const uint32_t offset = colorSpaceIndex(colorSpace, colorInfo) * 9u;
  return make_float3(
    matrix[offset] * rgb.x + matrix[offset + 1u] * rgb.y + matrix[offset + 2u] * rgb.z,
    matrix[offset + 3u] * rgb.x + matrix[offset + 4u] * rgb.y + matrix[offset + 5u] * rgb.z,
    matrix[offset + 6u] * rgb.x + matrix[offset + 7u] * rgb.y + matrix[offset + 8u] * rgb.z
  );
}

__device__ float3 hanatosRaw(
  float3 xyz,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *hanatosRawResponse
) {
  const float b = xyz.x + xyz.y + xyz.z;
  const float invB = 1.0f / fmaxf(b, 1.0e-10f);
  const float xyX = clampf(xyz.x * invB, 0.0f, 1.0f);
  const float xyY = clampf(xyz.y * invB, 0.0f, 1.0f);
  const float tx = clampf((1.0f - xyX) * (1.0f - xyX), 0.0f, 1.0f);
  const float ty = clampf(xyY / fmaxf(1.0f - xyX, 1.0e-10f), 0.0f, 1.0f);
  const float xCoord = tx * static_cast<float>(info.hanatosWidth - 1u);
  const float yCoord = ty * static_cast<float>(info.hanatosHeight - 1u);
  const int xBase = xCoord >= static_cast<float>(info.hanatosWidth - 1u)
    ? static_cast<int>(info.hanatosWidth - 2u)
    : static_cast<int>(floorf(xCoord));
  const int yBase = yCoord >= static_cast<float>(info.hanatosHeight - 1u)
    ? static_cast<int>(info.hanatosHeight - 2u)
    : static_cast<int>(floorf(yCoord));
  const float xFrac = xCoord >= static_cast<float>(info.hanatosWidth - 1u) ? 1.0f : xCoord - static_cast<float>(xBase);
  const float yFrac = yCoord >= static_cast<float>(info.hanatosHeight - 1u) ? 1.0f : yCoord - static_cast<float>(yBase);
  const float wx[4] = {
    mitchellWeight(xFrac + 1.0f),
    mitchellWeight(xFrac),
    mitchellWeight(xFrac - 1.0f),
    mitchellWeight(xFrac - 2.0f)
  };
  const float wy[4] = {
    mitchellWeight(yFrac + 1.0f),
    mitchellWeight(yFrac),
    mitchellWeight(yFrac - 1.0f),
    mitchellWeight(yFrac - 2.0f)
  };

  float3 raw = make_float3(0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  const uint32_t responseBase = colorAdaptationEnabled(params, kColorAdaptationInputCompression)
    ? info.hanatosWidth * info.hanatosHeight * 3u
    : 0u;
  for (uint32_t i = 0u; i < 4u; ++i) {
    const uint32_t xi = mirroredIndex(xBase - 1 + static_cast<int>(i), info.hanatosWidth);
    for (uint32_t j = 0u; j < 4u; ++j) {
      const uint32_t yj = mirroredIndex(yBase - 1 + static_cast<int>(j), info.hanatosHeight);
      const float weight = wx[i] * wy[j];
      weightSum += weight;
      const uint32_t offset = responseBase + (xi * info.hanatosHeight + yj) * 3u;
      raw.x += weight * hanatosRawResponse[offset];
      raw.y += weight * hanatosRawResponse[offset + 1u];
      raw.z += weight * hanatosRawResponse[offset + 2u];
    }
  }
  if (weightSum != 0.0f) {
    raw.x /= weightSum;
    raw.y /= weightSum;
    raw.z /= weightSum;
  }
  const float scale = fmaxf(b, 0.0f);
  raw.x *= scale;
  raw.y *= scale;
  raw.z *= scale;
  return raw;
}

__device__ float3 mallettRaw(float3 linearSrgb, const float *mallettBasisIlluminant) {
  const float r = fmaxf(linearSrgb.x, 0.0f);
  const float g = fmaxf(linearSrgb.y, 0.0f);
  const float b = fmaxf(linearSrgb.z, 0.0f);
  return make_float3(
    mallettBasisIlluminant[0u] * r + mallettBasisIlluminant[1u] * g + mallettBasisIlluminant[2u] * b,
    mallettBasisIlluminant[3u] * r + mallettBasisIlluminant[4u] * g + mallettBasisIlluminant[5u] * b,
    mallettBasisIlluminant[6u] * r + mallettBasisIlluminant[7u] * g + mallettBasisIlluminant[8u] * b
  );
}

__device__ float3 filmRawFromRgb(
  float3 rgb,
  const KernelParams &params,
  const KernelSpectralInfo &spectralInfo,
  const KernelColorInfo &colorInfo,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  const float *inputToSrgb,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind
) {
  const float3 decoded = decodeInputRgb(rgb, params, colorInfo, colorDecodeLut, colorTransferKind);
  float3 raw;
  if (params.rgbToRawMethod == 1) {
    raw = mallettRaw(mulColorMatrix(decoded, params.inputColorSpace, colorInfo, inputToSrgb), mallettBasisIlluminant);
  } else {
    raw = hanatosRaw(
      mulColorMatrix(decoded, params.inputColorSpace, colorInfo, inputToReferenceXyz),
      params,
      spectralInfo,
      hanatosRawResponse);
  }
  const float exposure = exp2f(params.filmExposureEv + params.autoExposureEv);
  return make_float3(fmaxf(raw.x * exposure, 0.0f), fmaxf(raw.y * exposure, 0.0f), fmaxf(raw.z * exposure, 0.0f));
}

__device__ int safePixelCoord(int value, int count) {
  return min(max(value, 0), max(count, 1) - 1);
}

__device__ float4 sampleFloat4Clamped(const float *source, int x, int y, int width, int height) {
  const int sx = safePixelCoord(x, width);
  const int sy = safePixelCoord(y, height);
  const size_t offset = (static_cast<size_t>(sy) * static_cast<size_t>(width) + static_cast<size_t>(sx)) * 4u;
  return make_float4(source[offset], source[offset + 1u], source[offset + 2u], source[offset + 3u]);
}

// Input and negative exposure setup.
__global__ void enlargerResampleKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const KernelParams p = *params;
  const float scale = fmaxf(p.enlargerScale, 1.0f);
  const float safeWidth = static_cast<float>(max(width, 1));
  const float safeHeight = static_cast<float>(max(height, 1));
  const float outputUvX = (static_cast<float>(x) + 0.5f) / safeWidth;
  const float outputUvY = (static_cast<float>(y) + 0.5f) / safeHeight;
  const float sourceUvX = 0.5f + (outputUvX - 0.5f) / scale + p.enlargerOffsetXPercent * (0.01f / scale);
  const float sourceUvY = 0.5f + (outputUvY - 0.5f) / scale + p.enlargerOffsetYPercent * (0.01f / scale);
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  if (sourceUvX < 0.0f || sourceUvX > 1.0f || sourceUvY < 0.0f || sourceUvY > 1.0f) {
    destination[outOffset] = 0.0f;
    destination[outOffset + 1u] = 0.0f;
    destination[outOffset + 2u] = 0.0f;
    destination[outOffset + 3u] = 1.0f;
    return;
  }
  const float sourcePxX = sourceUvX * safeWidth - 0.5f;
  const float sourcePxY = sourceUvY * safeHeight - 0.5f;
  const int x0 = static_cast<int>(floorf(sourcePxX));
  const int y0 = static_cast<int>(floorf(sourcePxY));
  const float tx = sourcePxX - floorf(sourcePxX);
  const float ty = sourcePxY - floorf(sourcePxY);
  const float4 p00 = sampleFloat4Clamped(source, x0, y0, width, height);
  const float4 p10 = sampleFloat4Clamped(source, x0 + 1, y0, width, height);
  const float4 p01 = sampleFloat4Clamped(source, x0, y0 + 1, width, height);
  const float4 p11 = sampleFloat4Clamped(source, x0 + 1, y0 + 1, width, height);
  const float4 a = make_float4(
    p00.x + (p10.x - p00.x) * tx,
    p00.y + (p10.y - p00.y) * tx,
    p00.z + (p10.z - p00.z) * tx,
    p00.w + (p10.w - p00.w) * tx);
  const float4 b = make_float4(
    p01.x + (p11.x - p01.x) * tx,
    p01.y + (p11.y - p01.y) * tx,
    p01.z + (p11.z - p01.z) * tx,
    p01.w + (p11.w - p01.w) * tx);
  destination[outOffset] = a.x + (b.x - a.x) * ty;
  destination[outOffset + 1u] = a.y + (b.y - a.y) * ty;
  destination[outOffset + 2u] = a.z + (b.z - a.z) * ty;
  destination[outOffset + 3u] = a.w + (b.w - a.w) * ty;
}

__global__ void rawExposureKernel(
  const float *source,
  float *raw,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  const float *inputToSrgb,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float3 rgb = make_float3(source[offset], source[offset + 1u], source[offset + 2u]);
  const float3 rawRgb = filmRawFromRgb(
    rgb,
    *params,
    *spectralInfo,
    *colorInfo,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz,
    inputToSrgb,
    colorDecodeLut,
    colorTransferKind
  );
  raw[offset] = rawRgb.x;
  raw[offset + 1u] = rawRgb.y;
  raw[offset + 2u] = rawRgb.z;
  raw[offset + 3u] = source[offset + 3u];
}

__global__ void autoExposurePreviewKernel(
  const float *source,
  float *luminance,
  int width,
  int height,
  int previewWidth,
  int previewHeight,
  const KernelParams *params,
  const KernelColorInfo *colorInfo,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind,
  float meterR,
  float meterG,
  float meterB
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int previewPixelCount = previewWidth * previewHeight;
  if (index >= previewPixelCount) {
    return;
  }
  const int x = index % previewWidth;
  const int y = index / previewWidth;
  const int sourceX = min(width - 1, static_cast<int>((static_cast<long long>(x) * width) / previewWidth));
  const int sourceY = min(height - 1, static_cast<int>((static_cast<long long>(y) * height) / previewHeight));
  const size_t offset =
    (static_cast<size_t>(sourceY) * static_cast<size_t>(width) + static_cast<size_t>(sourceX)) * 4u;
  const float3 decoded = decodeInputRgb(
    make_float3(source[offset], source[offset + 1u], source[offset + 2u]),
    *params,
    *colorInfo,
    colorDecodeLut,
    colorTransferKind);
  luminance[index] = decoded.x * meterR + decoded.y * meterG + decoded.z * meterB;
}

__global__ void rawToLogRawKernel(const float *raw, float *logRaw, int pixelCount) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  logRaw[offset] = logf(fmaxf(raw[offset], 0.0f) + 1.0e-10f) * invLog10;
  logRaw[offset + 1u] = logf(fmaxf(raw[offset + 1u], 0.0f) + 1.0e-10f) * invLog10;
  logRaw[offset + 2u] = logf(fmaxf(raw[offset + 2u], 0.0f) + 1.0e-10f) * invLog10;
  logRaw[offset + 3u] = raw[offset + 3u];
}

__device__ float sigmoidf(float u) {
  return 1.0f / (1.0f + expf(-u));
}

__device__ float channelValue(float3 value, uint32_t channel) {
  return channel == 0u ? value.x : (channel == 1u ? value.y : value.z);
}

__device__ float developmentActivity(float stops) {
  const float clampedStops = clampf(stops, -2.0f, 2.0f);
  float developmentSeconds = 180.0f;
  if (clampedStops < 0.0f) {
    developmentSeconds = mixf(180.0f, 150.0f, fminf(-clampedStops, 1.0f));
  } else if (clampedStops <= 1.0f) {
    developmentSeconds = mixf(180.0f, 220.0f, clampedStops);
  } else {
    developmentSeconds = mixf(220.0f, 280.0f, clampedStops - 1.0f);
  }
  return logf(developmentSeconds / 180.0f);
}

__device__ float pushPullSpeedGain(float stops) {
  if (stops > 0.0f) {
    const float pushOne = mixf(0.0f, 0.33f, fminf(stops, 1.0f));
    return stops <= 1.0f ? pushOne : mixf(0.33f, 0.5f, fminf(stops - 1.0f, 1.0f));
  }
  if (stops < 0.0f) {
    return mixf(0.0f, -0.2f, fminf(-stops, 1.0f));
  }
  return 0.0f;
}

__device__ float pushPullWarpLogRaw(float logRaw, uint32_t channel, float stops) {
  const float activity = developmentActivity(stops);
  const float signedActivitySquared = activity * fabsf(activity);
  const float3 toeLinear = make_float3(0.0f, 0.0f, 0.0f);
  const float3 midLinear = make_float3(0.25f, 0.28f, 0.31f);
  const float3 shoulderLinear = make_float3(0.32f, 0.38f, 0.45f);
  const float3 toeQuadratic = make_float3(0.0f, 0.0f, 0.0f);
  const float3 midQuadratic = make_float3(0.04f, 0.06f, 0.08f);
  const float3 shoulderQuadratic = make_float3(0.08f, 0.12f, 0.16f);
  const float toeMask = 1.0f - sigmoidf((logRaw + 2.0f) / 0.5f);
  const float shoulderMask = sigmoidf(logRaw / 0.5f);
  const float midMask = fmaxf(1.0f - toeMask - shoulderMask, 0.0f);
  const float toeShift = channelValue(toeLinear, channel) * activity + channelValue(toeQuadratic, channel) * signedActivitySquared;
  const float midShift = channelValue(midLinear, channel) * activity + channelValue(midQuadratic, channel) * signedActivitySquared;
  const float shoulderShift = channelValue(shoulderLinear, channel) * activity + channelValue(shoulderQuadratic, channel) * signedActivitySquared;
  return logRaw + toeShift * toeMask + midShift * midMask + shoulderShift * shoulderMask;
}

__device__ float3 experimentalPushPullLogRaw(float3 logRaw, float stops) {
  constexpr float log2OverLog10 = 0.3010299956639812f;
  float3 shifted = make_float3(
    logRaw.x - (stops - pushPullSpeedGain(stops)) * log2OverLog10,
    logRaw.y - (stops - pushPullSpeedGain(stops)) * log2OverLog10,
    logRaw.z - (stops - pushPullSpeedGain(stops)) * log2OverLog10
  );
  const float activity = developmentActivity(stops);
  const float meanLogRaw = (shifted.x + shifted.y + shifted.z) / 3.0f;
  const float dx = shifted.x - meanLogRaw;
  const float dy = shifted.y - meanLogRaw;
  const float dz = shifted.z - meanLogRaw;
  shifted.x += activity * (-0.015f * dy + 0.015f * dz);
  shifted.y += activity * (0.015f * dx - 0.015f * dz);
  shifted.z += activity * (-0.015f * dx + 0.015f * dy);
  return make_float3(
    pushPullWarpLogRaw(shifted.x, 0u, stops),
    pushPullWarpLogRaw(shifted.y, 1u, stops),
    pushPullWarpLogRaw(shifted.z, 2u, stops)
  );
}

__device__ float experimentalPushPullDensityGain(float logRaw, uint32_t channel, float stops) {
  const float activity = developmentActivity(stops);
  const float signedActivitySquared = activity * fabsf(activity);
  const float3 buildLinear = make_float3(1.05f, 0.95f, 1.00f);
  const float3 buildQuadratic = make_float3(-0.90f, -0.82f, -0.86f);
  const float toeMask = 1.0f - sigmoidf((logRaw + 2.0f) / 0.5f);
  const float shoulderMask = sigmoidf(logRaw / 0.5f);
  const float midMask = fmaxf(1.0f - toeMask - shoulderMask, 0.0f);
  const float regionWeight = 0.12f * toeMask + midMask + shoulderMask;
  const float build = channelValue(buildLinear, channel) * activity + channelValue(buildQuadratic, channel) * signedActivitySquared;
  return clampf(1.0f + build * regionWeight, 0.35f, 2.0f);
}

__device__ float interpDensityCurve(
  float logRaw,
  uint32_t channel,
  float gammaFactor,
  const KernelParams &params,
  const KernelCurveInfo &curveInfo,
  const float *logExposure,
  const float *densityCurves
) {
  const uint32_t count = curveInfo.exposureCount;
  if (count == 0u) {
    return 0.0f;
  }
  const float gamma = fmaxf(gammaFactor, 1.0e-6f);
  const float firstX = logExposure[0] / gamma;
  const float lastX = logExposure[count - 1u] / gamma;
  if (logRaw <= firstX) {
    return densityCurves[channel];
  }
  if (logRaw >= lastX) {
    return densityCurves[(count - 1u) * 3u + channel];
  }
  if (params.densityCurveLookupMode != 0u && count > 1u) {
    const float indexF = clampf((logRaw - firstX) * static_cast<float>(count - 1u) / fmaxf(lastX - firstX, 1.0e-9f), 0.0f, static_cast<float>(count - 1u));
    if (params.densityCurveLookupMode == 2u) {
      const uint32_t idx = static_cast<uint32_t>(clampf(floorf(indexF + 0.5f), 0.0f, static_cast<float>(count - 1u)));
      return densityCurves[idx * 3u + channel];
    }
    const uint32_t lo = static_cast<uint32_t>(floorf(indexF));
    const uint32_t hi = minu32(lo + 1u, count - 1u);
    return mixf(densityCurves[lo * 3u + channel], densityCurves[hi * 3u + channel], indexF - static_cast<float>(lo));
  }
  uint32_t lo = 0u;
  uint32_t hi = count - 1u;
  while (hi - lo > 1u) {
    const uint32_t mid = (lo + hi) >> 1u;
    const float x = logExposure[mid] / gamma;
    if (x <= logRaw) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  const float x0 = logExposure[lo] / gamma;
  const float x1 = logExposure[hi] / gamma;
  const float t = clampf((logRaw - x0) / fmaxf(x1 - x0, 1.0e-9f), 0.0f, 1.0f);
  if (colorAdaptationEnabled(params, kColorAdaptationCurveSmoothing) && count > 2u) {
    const float dx0 = fmaxf(x1 - x0, 1.0e-9f);
    const float d0 = (densityCurves[hi * 3u + channel] - densityCurves[lo * 3u + channel]) / dx0;
    float m0 = d0;
    float m1 = d0;
    if (lo > 0u) {
      const float xPrev = logExposure[lo - 1u] / gamma;
      const float dxPrev = fmaxf(x0 - xPrev, 1.0e-9f);
      const float dPrev = (densityCurves[lo * 3u + channel] - densityCurves[(lo - 1u) * 3u + channel]) / dxPrev;
      m0 = dPrev * d0 > 0.0f ? 0.5f * (dPrev + d0) : 0.0f;
    }
    if (hi + 1u < count) {
      const float xNext = logExposure[hi + 1u] / gamma;
      const float dxNext = fmaxf(xNext - x1, 1.0e-9f);
      const float dNext = (densityCurves[(hi + 1u) * 3u + channel] - densityCurves[hi * 3u + channel]) / dxNext;
      m1 = dNext * d0 > 0.0f ? 0.5f * (dNext + d0) : 0.0f;
    }
    if (fabsf(d0) <= 1.0e-9f) {
      m0 = 0.0f;
      m1 = 0.0f;
    } else {
      const float limit = 3.0f * fabsf(d0);
      m0 = d0 * m0 > 0.0f ? clampf(m0, -limit, limit) : 0.0f;
      m1 = d0 * m1 > 0.0f ? clampf(m1, -limit, limit) : 0.0f;
    }
    const float y0 = densityCurves[lo * 3u + channel];
    const float y1 = densityCurves[hi * 3u + channel];
    const float t2 = t * t;
    const float t3 = t2 * t;
    return (2.0f * t3 - 3.0f * t2 + 1.0f) * y0 +
      (t3 - 2.0f * t2 + t) * dx0 * m0 +
      (-2.0f * t3 + 3.0f * t2) * y1 +
      (t3 - t2) * dx0 * m1;
  }
  return mixf(densityCurves[lo * 3u + channel], densityCurves[hi * 3u + channel], t);
}

__device__ float3 developFilmDensity(
  float3 logRaw,
  const KernelParams &params,
  const KernelCurveInfo &curveInfo,
  const float *logExposure,
  const float *densityCurves
) {
  if (params.filmPushPullMode == 1) {
    const float3 lookupRaw = experimentalPushPullLogRaw(logRaw, params.filmPushPullStops);
    const float3 density = make_float3(
      interpDensityCurve(lookupRaw.x, 0u, params.filmGamma, params, curveInfo, logExposure, densityCurves),
      interpDensityCurve(lookupRaw.y, 1u, params.filmGamma, params, curveInfo, logExposure, densityCurves),
      interpDensityCurve(lookupRaw.z, 2u, params.filmGamma, params, curveInfo, logExposure, densityCurves)
    );
    return make_float3(
      density.x * experimentalPushPullDensityGain(lookupRaw.x, 0u, params.filmPushPullStops),
      density.y * experimentalPushPullDensityGain(lookupRaw.y, 1u, params.filmPushPullStops),
      density.z * experimentalPushPullDensityGain(lookupRaw.z, 2u, params.filmPushPullStops)
    );
  }
  return make_float3(
    interpDensityCurve(logRaw.x, 0u, params.filmGamma, params, curveInfo, logExposure, densityCurves),
    interpDensityCurve(logRaw.y, 1u, params.filmGamma, params, curveInfo, logExposure, densityCurves),
    interpDensityCurve(logRaw.z, 2u, params.filmGamma, params, curveInfo, logExposure, densityCurves)
  );
}

__device__ float3 add3(float3 a, float3 b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 sub3(float3 a, float3 b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 mul3(float3 a, float3 b) {
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float3 mul3s(float3 a, float b) {
  return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 div3(float3 a, float3 b) {
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ float3 max3s(float3 a, float b) {
  return make_float3(fmaxf(a.x, b), fmaxf(a.y, b), fmaxf(a.z, b));
}

__device__ float3 clamp3(float3 a, float lo, float hi) {
  return make_float3(clampf(a.x, lo, hi), clampf(a.y, lo, hi), clampf(a.z, lo, hi));
}

__device__ float3 mix3(float3 a, float3 b, float t) {
  return make_float3(mixf(a.x, b.x, t), mixf(a.y, b.y, t), mixf(a.z, b.z, t));
}

__device__ uint32_t clampIndex(int value, uint32_t count) {
  if (count == 0u) {
    return 0u;
  }
  const int hi = static_cast<int>(count - 1u);
  const int clamped = value < 0 ? 0 : (value > hi ? hi : value);
  return static_cast<uint32_t>(clamped);
}

__device__ float finitef(float value) {
  return isfinite(value) ? value : 0.0f;
}

__device__ float spectralTransmittance(float density, const KernelParams &params) {
  constexpr float log2Ten = 3.3219280948873623f;
  constexpr float lnTen = 2.302585092994046f;
  if (params.spectralTransmittanceMode == 1u) {
    return exp2f(-density * log2Ten);
  }
  if (params.spectralTransmittanceMode == 2u) {
    return expf(-density * lnTen);
  }
  return powf(10.0f, -density);
}

__device__ float3 filmSilverDensity(float3 densityCmy, const KernelSpectralInfo &info) {
  if (info.filmPositive != 0u) {
    return max3s(sub3(
      make_float3(info.filmDensityCurveMaximum0, info.filmDensityCurveMaximum1, info.filmDensityCurveMaximum2),
      densityCmy), 0.0f);
  }
  return max3s(densityCmy, 0.0f);
}

__device__ float retainedSilverImage(float3 densityCmy, bool printStage, const KernelSpectralInfo &info) {
  const float3 silverLayers = printStage ? max3s(densityCmy, 0.0f) : filmSilverDensity(densityCmy, info);
  const float layerShoulder = printStage ? 0.65f : 0.85f;
  const float r = silverLayers.x / (silverLayers.x + layerShoulder);
  const float g = silverLayers.y / (silverLayers.y + layerShoulder);
  const float b = silverLayers.z / (silverLayers.z + layerShoulder);
  return (r + g + b) / 3.0f;
}

__device__ float retainedSilverDensity(float3 densityCmy, float amount, bool printStage, const KernelSpectralInfo &info) {
  const float image = retainedSilverImage(densityCmy, printStage, info);
  const float scale = printStage ? 0.36f : 0.22f;
  return clampf(amount, 0.0f, 1.0f) * scale * image;
}

__device__ float3 bleachBypassDyeDensity(float3 densityCmy, float amount, bool printStage, const KernelSpectralInfo &info) {
  const float image = retainedSilverImage(densityCmy, printStage, info);
  const float blackImageAmount = clampf(clampf(amount, 0.0f, 1.0f) * image, 0.0f, 1.0f);
  const float blackDensity = fmaxf(densityCmy.x, fmaxf(densityCmy.y, densityCmy.z));
  return mix3(densityCmy, make_float3(blackDensity, blackDensity, blackDensity), blackImageAmount);
}

__device__ float negativeLeucoCyanDensityLoss(
  float3 densityCmy,
  float amount,
  const KernelParams &params,
  const KernelSpectralInfo &info
) {
  if (info.filmPositive != 0u) {
    return 0.0f;
  }
  const float cyanMax = fmaxf(info.filmDensityCurveMaximum0, 1.0e-6f);
  const float drive = clampf(fmaxf(densityCmy.x, 0.0f) / cyanMax, 0.0f, 1.0f);
  const float coupling = clampf(params.negativeLeucoCyanCoupling, 0.0f, 2.0f);
  constexpr float maxLoss = 0.30f;
  return fminf(clampf(amount, 0.0f, 1.0f) * coupling * maxLoss * drive, maxLoss * coupling);
}

__device__ float3 negativeBleachBypassDyeDensity(
  float3 densityCmy,
  float amount,
  const KernelParams &params,
  const KernelSpectralInfo &info
) {
  float3 bypassed = bleachBypassDyeDensity(densityCmy, amount, false, info);
  bypassed.x = fmaxf(bypassed.x - negativeLeucoCyanDensityLoss(densityCmy, amount, params, info), 0.0f);
  return bypassed;
}

__device__ float filteredEnlargerIlluminantWithFilters(
  uint32_t wavelength,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  float cFilter,
  float mFilterShift,
  float yFilterShift
) {
  const uint32_t film = clampIndex(params.film, info.filmCount);
  const uint32_t paper = clampIndex(params.paper, info.paperCount);
  const uint32_t neutralOffset = (paper * info.filmCount + film) * 3u;
  const float c = fmaxf(neutralPrintFilters[neutralOffset] + cFilter, 0.0f);
  const float m = fmaxf(neutralPrintFilters[neutralOffset + 1u] + mFilterShift, 0.0f);
  const float y = fmaxf(neutralPrintFilters[neutralOffset + 2u] + yFilterShift, 0.0f);
  const float wheelC = powf(10.0f, -c / 100.0f);
  const float wheelM = powf(10.0f, -m / 100.0f);
  const float wheelY = powf(10.0f, -y / 100.0f);
  const uint32_t filterOffset = wavelength * 3u;
  const float filterC = clampf(customEnlargerFilters[filterOffset], 0.0f, 1.0f);
  const float filterM = clampf(customEnlargerFilters[filterOffset + 1u], 0.0f, 1.0f);
  const float filterY = clampf(customEnlargerFilters[filterOffset + 2u], 0.0f, 1.0f);
  const float dimC = 1.0f - (1.0f - filterC) * (1.0f - wheelC);
  const float dimM = 1.0f - (1.0f - filterM) * (1.0f - wheelM);
  const float dimY = 1.0f - (1.0f - filterY) * (1.0f - wheelY);
  return thKg3Illuminant[wavelength] * dimC * dimM * dimY;
}

__device__ float3 printRawFromFilmDensity(
  float3 filmDensityCmy,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData
) {
  float3 raw = make_float3(0.0f, 0.0f, 0.0f);
  const float3 bypassed = negativeBleachBypassDyeDensity(filmDensityCmy, params.negativeBleachBypassAmount, params, info);
  const float silverDensity = retainedSilverDensity(filmDensityCmy, params.negativeBleachBypassAmount, false, info);
  if (params.printTiming == 1) {
    float3 normalization = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
      const uint32_t channelOffset = wavelength * 3u;
      const float densitySpectral =
        bypassed.x * filmChannelDensity[channelOffset] +
        bypassed.y * filmChannelDensity[channelOffset + 1u] +
        bypassed.z * filmChannelDensity[channelOffset + 2u] +
        filmBaseDensity[wavelength] +
        silverDensity;
      const float transmittance = spectralTransmittance(densitySpectral, params);
      const float3 apd = max3s(make_float3(
        academyPrinterDensityData[channelOffset],
        academyPrinterDensityData[channelOffset + 1u],
        academyPrinterDensityData[channelOffset + 2u]), 0.0f);
      raw = add3(raw, mul3s(apd, finitef(transmittance)));
      normalization = add3(normalization, apd);
    }
    return div3(raw, max3s(normalization, 1.0e-10f));
  }

  for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
    const uint32_t channelOffset = wavelength * 3u;
    const float densitySpectral =
      bypassed.x * filmChannelDensity[channelOffset] +
      bypassed.y * filmChannelDensity[channelOffset + 1u] +
      bypassed.z * filmChannelDensity[channelOffset + 2u] +
      filmBaseDensity[wavelength] +
      silverDensity;
    const float lightRaw = spectralTransmittance(densitySpectral, params) *
      filteredEnlargerIlluminantWithFilters(
        wavelength,
        params,
        info,
        thKg3Illuminant,
        customEnlargerFilters,
        neutralPrintFilters,
        params.filterC,
        params.filterMShift,
        params.filterYShift);
    const float light = finitef(lightRaw);
    raw.x += light * paperLogSensitivity[channelOffset];
    raw.y += light * paperLogSensitivity[channelOffset + 1u];
    raw.z += light * paperLogSensitivity[channelOffset + 2u];
  }
  return raw;
}

__device__ float3 printerLightExposureScale(
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *academyPrinterDensityData
) {
  const float linkedPoint = (params.printerLightsR + params.printerLightsG + params.printerLightsB) / 3.0f;
  float3 points = params.printerLightsGang != 0u
    ? make_float3(linkedPoint, linkedPoint, linkedPoint)
    : make_float3(params.printerLightsR, params.printerLightsG, params.printerLightsB);
  float3 internal = make_float3(0.0f, 0.0f, 0.0f);
  if (params.printTiming == 1 && params.printerLightCalibration != 0u) {
    const uint32_t film = clampIndex(params.film, info.filmCount);
    const uint32_t paper = clampIndex(params.paper, info.paperCount);
    const uint32_t offset = info.filmWavelengthCount * 3u + (paper * info.filmCount + film) * 3u;
    internal = make_float3(
      academyPrinterDensityData[offset],
      academyPrinterDensityData[offset + 1u],
      academyPrinterDensityData[offset + 2u]);
  }
  return make_float3(
    exp2f((internal.x + points.x) / 12.0f),
    exp2f((internal.y + points.y) / 12.0f),
    exp2f((internal.z + points.z) / 12.0f));
}

__device__ float3 apdPrinterTimingExposureScale(
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *academyPrinterDensityData
) {
  if (params.printTiming != 1) {
    return make_float3(1.0f, 1.0f, 1.0f);
  }
  return printerLightExposureScale(params, info, academyPrinterDensityData);
}

__device__ float3 apdNeutralExposureScale(
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *academyPrinterDensityData
) {
  if (params.printTiming != 1 || params.printerLightCalibration == 0u) {
    return make_float3(1.0f, 1.0f, 1.0f);
  }
  const uint32_t film = clampIndex(params.film, info.filmCount);
  const uint32_t paper = clampIndex(params.paper, info.paperCount);
  const uint32_t offset = info.filmWavelengthCount * 3u + (paper * info.filmCount + film) * 3u;
  return make_float3(
    exp2f(academyPrinterDensityData[offset] / 12.0f),
    exp2f(academyPrinterDensityData[offset + 1u] / 12.0f),
    exp2f(academyPrinterDensityData[offset + 2u] / 12.0f));
}

__device__ float3 printRawPreflash(
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData
) {
  if (params.preflashExposure <= 0.0f) {
    return make_float3(0.0f, 0.0f, 0.0f);
  }
  float3 raw = make_float3(0.0f, 0.0f, 0.0f);
  if (params.printTiming == 1) {
    float3 normalization = make_float3(0.0f, 0.0f, 0.0f);
    for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
      const uint32_t channelOffset = wavelength * 3u;
      const float transmittance = spectralTransmittance(filmBaseDensity[wavelength], params);
      const float3 apd = max3s(make_float3(
        academyPrinterDensityData[channelOffset],
        academyPrinterDensityData[channelOffset + 1u],
        academyPrinterDensityData[channelOffset + 2u]), 0.0f);
      raw = add3(raw, mul3s(apd, finitef(transmittance)));
      normalization = add3(normalization, apd);
    }
    return mul3s(
      mul3(div3(raw, max3s(normalization, 1.0e-10f)), apdNeutralExposureScale(params, info, academyPrinterDensityData)),
      fmaxf(params.preflashExposure, 0.0f));
  }
  for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
    const uint32_t channelOffset = wavelength * 3u;
    const float lightRaw = spectralTransmittance(filmBaseDensity[wavelength], params) *
      filteredEnlargerIlluminantWithFilters(
        wavelength,
        params,
        info,
        thKg3Illuminant,
        customEnlargerFilters,
        neutralPrintFilters,
        0.0f,
        params.preflashMFilterShift,
        params.preflashYFilterShift);
    const float light = finitef(lightRaw);
    raw.x += light * paperLogSensitivity[channelOffset];
    raw.y += light * paperLogSensitivity[channelOffset + 1u];
    raw.z += light * paperLogSensitivity[channelOffset + 2u];
  }
  return mul3s(raw, fmaxf(params.preflashExposure, 0.0f));
}

__device__ float3 filmLogRawLinearSrgb(
  float3 linearSrgb,
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const KernelSpectralInfo &info,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz
) {
  constexpr int linearSrgbColorSpace = 15;
  const float3 rgb = max3s(linearSrgb, 0.0f);
  const float3 raw = params.rgbToRawMethod == 1
    ? mallettRaw(rgb, mallettBasisIlluminant)
    : hanatosRaw(mulColorMatrix(rgb, linearSrgbColorSpace, colorInfo, inputToReferenceXyz), params, info, hanatosRawResponse);
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  return make_float3(
    logf(fmaxf(raw.x, 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(raw.y, 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(raw.z, 0.0f) + 1.0e-10f) * invLog10);
}

__device__ float printMidgrayExposureFactor(
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const KernelSpectralInfo &info,
  const KernelCurveInfo &filmCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz
) {
  const float3 midgrayLogRaw = filmLogRawLinearSrgb(
    make_float3(0.184f, 0.184f, 0.184f),
    params,
    colorInfo,
    info,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float3 midgrayDensity = add3(
    developFilmDensity(midgrayLogRaw, params, filmCurveInfo, filmLogExposure, filmDensityCurves),
    make_float3(info.filmDensityCurveMinimum0, info.filmDensityCurveMinimum1, info.filmDensityCurveMinimum2));
  const float3 rawMidgray = max3s(printRawFromFilmDensity(
    midgrayDensity,
    params,
    info,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData), 1.0e-10f);
  const float geomean = expf((logf(rawMidgray.x) + logf(rawMidgray.y) + logf(rawMidgray.z)) / 3.0f);
  return 1.0f / fmaxf(geomean, 1.0e-10f);
}

__device__ float3 developPrintDensity(
  float3 logRaw,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const KernelCurveInfo &paperCurveInfo,
  const float *paperLogExposure,
  const float *paperDensityCurves
) {
  const float3 density = make_float3(
    interpDensityCurve(logRaw.x, 0u, params.printGamma, params, paperCurveInfo, paperLogExposure, paperDensityCurves),
    interpDensityCurve(logRaw.y, 1u, params.printGamma, params, paperCurveInfo, paperLogExposure, paperDensityCurves),
    interpDensityCurve(logRaw.z, 2u, params.printGamma, params, paperCurveInfo, paperLogExposure, paperDensityCurves));
  if (params.printShadowShape == 0.0f && params.printHighlightShape == 0.0f) {
    return density;
  }
  const float3 densityMaximum = max3s(
    make_float3(info.paperDensityCurveMaximum0, info.paperDensityCurveMaximum1, info.paperDensityCurveMaximum2),
    1.0e-6f);
  const float3 normalized = clamp3(div3(density, densityMaximum), 0.0f, 1.0f);
  const float3 oneMinus = sub3(make_float3(1.0f, 1.0f, 1.0f), normalized);
  const float3 shadowBasis = mul3(mul3(normalized, normalized), oneMinus);
  const float3 highlightBasis = mul3(mul3(normalized, oneMinus), oneMinus);
  constexpr float strength = 0.5f;
  const float shadow = strength * clampf(params.printShadowShape, -1.0f, 1.0f);
  const float highlight = strength * clampf(params.printHighlightShape, -1.0f, 1.0f);
  const float3 shaped = clamp3(sub3(sub3(normalized, mul3s(shadowBasis, shadow)), mul3s(highlightBasis, highlight)), 0.0f, 1.0f);
  return mul3(shaped, densityMaximum);
}

struct CudaScanResult {
  float3 rgb;
  float y;
};

__device__ uint32_t finalOutputColorSpace(const KernelParams &params, const KernelColorInfo &colorInfo) {
  constexpr int linearRec2020ColorSpace = 14;
  return colorSpaceIndex(params.outputRole == 1 ? linearRec2020ColorSpace : params.outputColorSpace, colorInfo);
}

__device__ float3 mulScanMatrix(
  float3 rgb,
  int colorSpace,
  const KernelColorInfo &colorInfo,
  const float *scanToOutputRgbData,
  uint32_t matrixBaseOffset
) {
  const uint32_t offset = matrixBaseOffset + colorSpaceIndex(colorSpace, colorInfo) * 9u;
  return make_float3(
    scanToOutputRgbData[offset] * rgb.x + scanToOutputRgbData[offset + 1u] * rgb.y + scanToOutputRgbData[offset + 2u] * rgb.z,
    scanToOutputRgbData[offset + 3u] * rgb.x + scanToOutputRgbData[offset + 4u] * rgb.y + scanToOutputRgbData[offset + 5u] * rgb.z,
    scanToOutputRgbData[offset + 6u] * rgb.x + scanToOutputRgbData[offset + 7u] * rgb.y + scanToOutputRgbData[offset + 8u] * rgb.z);
}

__device__ float sampleEncodeLutRange(
  float value,
  uint32_t colorSpace,
  float minimum,
  float maximum,
  const KernelColorInfo &colorInfo,
  const float *colorEncodeLut
) {
  const uint32_t lutSize = colorInfo.transferLutSize;
  if (!colorEncodeLut || lutSize <= 1u) {
    return value;
  }
  const uint32_t offset = colorSpace * lutSize;
  const float range = fmaxf(maximum - minimum, 1.0e-6f);
  const float step = range / static_cast<float>(lutSize - 1u);
  if (value <= minimum) {
    const float y0 = colorEncodeLut[offset];
    const float y1 = colorEncodeLut[offset + 1u];
    return y0 + (value - minimum) * ((y1 - y0) / fmaxf(step, 1.0e-12f));
  }
  if (value >= maximum) {
    const float y0 = colorEncodeLut[offset + lutSize - 2u];
    const float y1 = colorEncodeLut[offset + lutSize - 1u];
    return y1 + (value - maximum) * ((y1 - y0) / fmaxf(step, 1.0e-12f));
  }
  const float t = (value - minimum) / range;
  const float position = t * static_cast<float>(lutSize - 1u);
  const uint32_t lo = static_cast<uint32_t>(floorf(position));
  const uint32_t hi = minu32(lo + 1u, lutSize - 1u);
  const float f = position - static_cast<float>(lo);
  return mixf(colorEncodeLut[offset + lo], colorEncodeLut[offset + hi], f);
}

__device__ float3 encodeOutputRgb(
  float3 rgb,
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind
) {
  const uint32_t colorSpace = colorSpaceIndex(params.outputColorSpace, colorInfo);
  if (!colorTransferKind || colorTransferKind[colorSpace] == 0u) {
    return rgb;
  }
  return make_float3(
    sampleEncodeLutRange(rgb.x, colorSpace, colorInfo.encodeMin, colorInfo.encodeMax, colorInfo, colorEncodeLut),
    sampleEncodeLutRange(rgb.y, colorSpace, colorInfo.encodeMin, colorInfo.encodeMax, colorInfo, colorEncodeLut),
    sampleEncodeLutRange(rgb.z, colorSpace, colorInfo.encodeMin, colorInfo.encodeMax, colorInfo, colorEncodeLut));
}

__device__ float rec2020Luminance(float3 rgb) {
  return rgb.x * 0.2627f + rgb.y * 0.6780f + rgb.z * 0.0593f;
}

__device__ float3 hdrMapToNits(float3 rgb, const KernelParams &params) {
  const float referenceWhite = fmaxf(params.hdrReferenceWhiteNits, 1.0f);
  const float peak = fmaxf(params.hdrPeakNits, referenceWhite + 1.0f);
  float3 nits = mul3s(max3s(rgb, 0.0f), referenceWhite * exp2f(params.hdrExposureEv));
  const float sourceY = fmaxf(rec2020Luminance(nits), 1.0e-6f);
  float mappedY = sourceY;
  if (params.hdrToneMapping == 1) {
    mappedY = fminf(sourceY, peak);
  } else if (sourceY > referenceWhite) {
    const float shoulder = fmaxf(peak - referenceWhite, 1.0f);
    mappedY = referenceWhite + shoulder * (1.0f - expf(-(sourceY - referenceWhite) / shoulder));
  }
  return mul3s(nits, mappedY / sourceY);
}

__device__ float encodePq(float nits) {
  constexpr float m1 = 2610.0f / 16384.0f;
  constexpr float m2 = 2523.0f / 32.0f;
  constexpr float c1 = 3424.0f / 4096.0f;
  constexpr float c2 = 2413.0f / 128.0f;
  constexpr float c3 = 2392.0f / 128.0f;
  const float y = powf(fmaxf(nits, 0.0f) / 10000.0f, m1);
  return powf((c1 + c2 * y) / (1.0f + c3 * y), m2);
}

__device__ float encodeHlg(float sceneLinear) {
  constexpr float a = 0.17883277f;
  constexpr float b = 1.0f - 4.0f * a;
  constexpr float c = 0.55991073f;
  const float e = fmaxf(sceneLinear, 0.0f);
  return e <= (1.0f / 12.0f) ? sqrtf(3.0f * e) : a * logf(12.0f * e - b) + c;
}

__device__ float3 hlgDisplayNitsToSignal(float3 nits, float peakNits) {
  const float gamma = fmaxf(1.0f + 0.42f * (logf(fmaxf(peakNits, 1.0f) / 1000.0f) / logf(10.0f)), 1.0e-6f);
  return make_float3(
    encodeHlg(powf(fmaxf(nits.x, 0.0f) / peakNits, 1.0f / gamma)),
    encodeHlg(powf(fmaxf(nits.y, 0.0f) / peakNits, 1.0f / gamma)),
    encodeHlg(powf(fmaxf(nits.z, 0.0f) / peakNits, 1.0f / gamma)));
}

__device__ float reinhardKneeCuda(float value, float threshold, float limit, float power) {
  if (!isfinite(value) || value <= threshold) {
    return value;
  }
  const float scale = fmaxf(limit - threshold, 1.0e-12f);
  const float x = (value - threshold) / scale;
  const float y = x / powf(1.0f + powf(x, power), 1.0f / power);
  return threshold + scale * y;
}

__device__ float signedCuberootCuda(float value) {
  return value < 0.0f ? -powf(-value, 1.0f / 3.0f) : powf(value, 1.0f / 3.0f);
}

__device__ float3 multiplyPackedColorMatrixCuda(const float *data, uint32_t offset, float3 value) {
  return make_float3(
    data[offset] * value.x + data[offset + 1u] * value.y + data[offset + 2u] * value.z,
    data[offset + 3u] * value.x + data[offset + 4u] * value.y + data[offset + 5u] * value.z,
    data[offset + 6u] * value.x + data[offset + 7u] * value.y + data[offset + 8u] * value.z);
}

__device__ float3 oklabFromOutputRgbCuda(float3 rgb, const float *outputGamutCompressionData, uint32_t dataOffset) {
  const float3 lms = multiplyPackedColorMatrixCuda(outputGamutCompressionData, dataOffset, rgb);
  const float3 lmsPrime = make_float3(
    signedCuberootCuda(lms.x),
    signedCuberootCuda(lms.y),
    signedCuberootCuda(lms.z));
  return make_float3(
    0.2104542553f * lmsPrime.x + 0.7936177850f * lmsPrime.y - 0.0040720468f * lmsPrime.z,
    1.9779984951f * lmsPrime.x - 2.4285922050f * lmsPrime.y + 0.4505937099f * lmsPrime.z,
    0.0259040371f * lmsPrime.x + 0.7827717662f * lmsPrime.y - 0.8086757660f * lmsPrime.z);
}

__device__ float3 outputRgbFromOklabCuda(float3 lab, const float *outputGamutCompressionData, uint32_t dataOffset) {
  const float l = lab.x + 0.3963377774f * lab.y + 0.2158037573f * lab.z;
  const float m = lab.x - 0.1055613458f * lab.y - 0.0638541728f * lab.z;
  const float s = lab.x - 0.0894841775f * lab.y - 1.2914855480f * lab.z;
  const float3 lms = make_float3(l * l * l, m * m * m, s * s * s);
  return multiplyPackedColorMatrixCuda(outputGamutCompressionData, dataOffset + 9u, lms);
}

__device__ bool rgbInBoundsCuda(float3 rgb, float lowerBound, float upperBound) {
  constexpr float epsilon = 1.0e-6f;
  return isfinite(rgb.x) && isfinite(rgb.y) && isfinite(rgb.z) &&
    rgb.x >= lowerBound - epsilon && rgb.y >= lowerBound - epsilon && rgb.z >= lowerBound - epsilon &&
    rgb.x <= upperBound + epsilon && rgb.y <= upperBound + epsilon && rgb.z <= upperBound + epsilon;
}

__device__ __noinline__ float solveOklchCmaxCuda(
  float3 lab,
  float chroma,
  float hueX,
  float hueY,
  const float *outputGamutCompressionData,
  uint32_t dataOffset,
  float lowerBound,
  float upperBound
) {
  float lo = 0.0f;
  float hi = fmaxf(chroma, 1.0e-6f);
  const float maxHi = 4.0f * fmaxf(upperBound, 1.0f);
  for (uint32_t expansion = 0u; expansion < 12u; ++expansion) {
    const float3 candidate = outputRgbFromOklabCuda(
      make_float3(lab.x, hueX * hi, hueY * hi),
      outputGamutCompressionData,
      dataOffset);
    if (!rgbInBoundsCuda(candidate, lowerBound, upperBound)) {
      break;
    }
    lo = hi;
    hi = fminf(hi * 2.0f, maxHi);
    if (hi >= maxHi) {
      break;
    }
  }
  for (uint32_t iteration = 0u; iteration < 16u; ++iteration) {
    const float mid = 0.5f * (lo + hi);
    const float3 candidate = outputRgbFromOklabCuda(
      make_float3(lab.x, hueX * mid, hueY * mid),
      outputGamutCompressionData,
      dataOffset);
    if (rgbInBoundsCuda(candidate, lowerBound, upperBound)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

__device__ __noinline__ float3 outputGamutCompressOklchCuda(
  float3 rgb,
  uint32_t colorSpace,
  const KernelColorInfo &colorInfo,
  const float *colorEncodeLut,
  float lowerBound,
  float upperBound,
  bool softenInGamut,
  bool compressLightness,
  bool compressChroma
) {
  if (!isfinite(rgb.x) || !isfinite(rgb.y) || !isfinite(rgb.z)) {
    return make_float3(lowerBound, lowerBound, lowerBound);
  }
  const bool inBounds = rgbInBoundsCuda(rgb, lowerBound, upperBound);
  if (inBounds && !softenInGamut) {
    return rgb;
  }
  const float *outputGamutCompressionData =
    colorEncodeLut + colorInfo.colorSpaceCount * colorInfo.transferLutSize;
  const uint32_t dataOffset = colorSpace * kOutputGamutCompressionStride;
  float3 lab = oklabFromOutputRgbCuda(rgb, outputGamutCompressionData, dataOffset);
  if (compressLightness) {
    lab.x = softenInGamut
      ? reinhardKneeCuda(fmaxf(lab.x, 0.0f), 0.7f, 1.0f, 2.2f)
      : clampf(lab.x, 0.0f, powf(fmaxf(upperBound, 0.0f), 1.0f / 3.0f));
  }
  const float chroma = sqrtf(lab.y * lab.y + lab.z * lab.z);
  if (!(chroma > 1.0e-10f) || !isfinite(chroma)) {
    if (inBounds) {
      return rgb;
    }
    const float3 neutral = outputRgbFromOklabCuda(
      make_float3(lab.x, 0.0f, 0.0f),
      outputGamutCompressionData,
      dataOffset);
    return make_float3(
      clampf(neutral.x, lowerBound, upperBound),
      clampf(neutral.y, lowerBound, upperBound),
      clampf(neutral.z, lowerBound, upperBound));
  }
  if (!compressChroma) {
    const float3 lightnessCompressed = outputRgbFromOklabCuda(lab, outputGamutCompressionData, dataOffset);
    return rgbInBoundsCuda(lightnessCompressed, lowerBound, upperBound)
      ? lightnessCompressed
      : make_float3(
          clampf(lightnessCompressed.x, lowerBound, upperBound),
          clampf(lightnessCompressed.y, lowerBound, upperBound),
          clampf(lightnessCompressed.z, lowerBound, upperBound));
  }
  const float hueX = lab.y / chroma;
  const float hueY = lab.z / chroma;
  const float cmax = fmaxf(
    solveOklchCmaxCuda(lab, chroma, hueX, hueY, outputGamutCompressionData, dataOffset, lowerBound, upperBound),
    1.0e-9f);
  const float normalizedChroma = chroma / cmax;
  const float compressedNormalized = softenInGamut
    ? reinhardKneeCuda(normalizedChroma, 0.0f, 1.0f, 6.0f)
    : (normalizedChroma <= 1.0f ? normalizedChroma : reinhardKneeCuda(normalizedChroma, 0.85f, 1.0f, 4.0f));
  const float compressedChroma = fminf(compressedNormalized * cmax, cmax);
  const float3 compressed = outputRgbFromOklabCuda(
    make_float3(lab.x, hueX * compressedChroma, hueY * compressedChroma),
    outputGamutCompressionData,
    dataOffset);
  return rgbInBoundsCuda(compressed, lowerBound, upperBound)
    ? compressed
    : make_float3(
        clampf(compressed.x, lowerBound, upperBound),
        clampf(compressed.y, lowerBound, upperBound),
        clampf(compressed.z, lowerBound, upperBound));
}

__device__ float inverseRcmOotfChannel(float value) {
  const float x = fabsf(value);
  const float signal = powf(x, 1.0f / 2.4f);
  const float sceneLinear = signal < 0.081f
    ? signal / 4.5f
    : powf((signal + 0.099f) / 1.099f, 1.0f / 0.45f);
  return value < 0.0f ? -sceneLinear : sceneLinear;
}

__device__ float3 applyInverseRcmOotf(float3 rgb) {
  return make_float3(
    inverseRcmOotfChannel(rgb.x),
    inverseRcmOotfChannel(rgb.y),
    inverseRcmOotfChannel(rgb.z));
}

__device__ float3 finalizeOutputRgb(
  float3 rgb,
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind
) {
  // Common path: avoid the heavier OKLch helpers unless color adaptation is enabled.
  if (params.colorAdaptationFlags == 0u) {
    if (params.outputRole == 1) {
      const float peak = fmaxf(params.hdrPeakNits, fmaxf(params.hdrReferenceWhiteNits, 1.0f) + 1.0f);
      const float3 nits = hdrMapToNits(rgb, params);
      return params.hdrTransfer == 1
        ? hlgDisplayNitsToSignal(nits, peak)
        : make_float3(encodePq(nits.x), encodePq(nits.y), encodePq(nits.z));
    }
    if (params.outputRole == 2) {
      const int colorSpace = params.outputColorSpace;
      const bool inverseOotf =
        colorSpace == 2 || colorSpace == 3 ||
        (colorSpace >= 10 && colorSpace <= 18) ||
        (colorSpace >= 21 && colorSpace <= 25);
      if (inverseOotf) {
        rgb = applyInverseRcmOotf(rgb);
      }
    }
    return encodeOutputRgb(rgb, params, colorInfo, colorEncodeLut, colorTransferKind);
  }
  if (params.outputRole == 1) {
    const float peak = fmaxf(params.hdrPeakNits, fmaxf(params.hdrReferenceWhiteNits, 1.0f) + 1.0f);
    const bool compressLightness = colorAdaptationEnabled(params, kColorAdaptationOutputLightnessCompression);
    const bool compressChroma = colorAdaptationEnabled(params, kColorAdaptationOutputChromaCompression);
    const bool compressOutputGamut = compressLightness || compressChroma;
    float3 nits = hdrMapToNits(rgb, params);
    if (compressOutputGamut) {
      nits = outputGamutCompressOklchCuda(
        make_float3(nits.x / peak, nits.y / peak, nits.z / peak),
        finalOutputColorSpace(params, colorInfo),
        colorInfo,
        colorEncodeLut,
        0.0f,
        1.0f,
        false,
        compressLightness,
        compressChroma);
      nits = make_float3(
        clampf(nits.x * peak, 0.0f, peak),
        clampf(nits.y * peak, 0.0f, peak),
        clampf(nits.z * peak, 0.0f, peak));
    }
    return params.hdrTransfer == 1
      ? hlgDisplayNitsToSignal(nits, peak)
      : make_float3(encodePq(nits.x), encodePq(nits.y), encodePq(nits.z));
  }
  if (params.outputRole == 2) {
    const int colorSpace = params.outputColorSpace;
    const bool inverseOotf =
      colorSpace == 2 || colorSpace == 3 ||
      (colorSpace >= 10 && colorSpace <= 18) ||
      (colorSpace >= 21 && colorSpace <= 25);
    if (inverseOotf) {
      rgb = applyInverseRcmOotf(rgb);
    }
  } else {
    const bool compressLightness = colorAdaptationEnabled(params, kColorAdaptationOutputLightnessCompression);
    const bool compressChroma = colorAdaptationEnabled(params, kColorAdaptationOutputChromaCompression);
    if (compressLightness || compressChroma) {
      rgb = outputGamutCompressOklchCuda(
        rgb,
        finalOutputColorSpace(params, colorInfo),
        colorInfo,
        colorEncodeLut,
        0.0f,
        1.0f,
        true,
        compressLightness,
        compressChroma);
    }
  }
  return encodeOutputRgb(rgb, params, colorInfo, colorEncodeLut, colorTransferKind);
}

__device__ CudaScanResult scanDensityToOutputRgbLinearY(
  float3 densityCmy,
  float retainedSilver,
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const KernelSpectralInfo &info,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  bool paperScan
) {
  const uint32_t baseDensityOffset = paperScan ? info.filmWavelengthCount * 3u : 0u;
  const uint32_t scanIlluminantOffset = paperScan ? info.filmWavelengthCount : 0u;
  const uint32_t standardObserverOffset = info.filmWavelengthCount * 2u;
  const uint32_t scanMatrixOffset = paperScan ? colorInfo.colorSpaceCount * 9u : 0u;
  float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
  float normalization = 0.0f;
  for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
    const uint32_t channelOffset = wavelength * 3u;
    const float cDensity = paperScan ? paperScanDensityData[channelOffset] : filmChannelDensity[channelOffset];
    const float mDensity = paperScan ? paperScanDensityData[channelOffset + 1u] : filmChannelDensity[channelOffset + 1u];
    const float yDensity = paperScan ? paperScanDensityData[channelOffset + 2u] : filmChannelDensity[channelOffset + 2u];
    const float baseDensity = paperScan ? paperScanDensityData[baseDensityOffset + wavelength] : filmBaseDensity[wavelength];
    const float scanIlluminant = scanIlluminantsAndCmfs[scanIlluminantOffset + wavelength];
    const float densitySpectral =
      densityCmy.x * cDensity +
      densityCmy.y * mDensity +
      densityCmy.z * yDensity +
      baseDensity +
      retainedSilver;
    const float light = finitef(spectralTransmittance(densitySpectral, params) * scanIlluminant);
    const float cmfX = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset];
    const float cmfY = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset + 1u];
    const float cmfZ = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset + 2u];
    xyz.x += light * cmfX;
    xyz.y += light * cmfY;
    xyz.z += light * cmfZ;
    normalization += scanIlluminant * cmfY;
  }
  xyz = mul3s(xyz, 1.0f / fmaxf(normalization, 1.0e-10f));
  CudaScanResult result;
  result.rgb = mulScanMatrix(
    xyz,
    static_cast<int>(finalOutputColorSpace(params, colorInfo)),
    colorInfo,
    scanToOutputRgbData,
    scanMatrixOffset);
  result.y = xyz.y;
  return result;
}

__device__ float scanDensityToY(
  float3 densityCmy,
  float retainedSilver,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  bool paperScan
) {
  const uint32_t baseDensityOffset = paperScan ? info.filmWavelengthCount * 3u : 0u;
  const uint32_t scanIlluminantOffset = paperScan ? info.filmWavelengthCount : 0u;
  const uint32_t standardObserverOffset = info.filmWavelengthCount * 2u;
  float y = 0.0f;
  float normalization = 0.0f;
  for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
    const uint32_t channelOffset = wavelength * 3u;
    const float cDensity = paperScan ? paperScanDensityData[channelOffset] : filmChannelDensity[channelOffset];
    const float mDensity = paperScan ? paperScanDensityData[channelOffset + 1u] : filmChannelDensity[channelOffset + 1u];
    const float yDensity = paperScan ? paperScanDensityData[channelOffset + 2u] : filmChannelDensity[channelOffset + 2u];
    const float baseDensity = paperScan ? paperScanDensityData[baseDensityOffset + wavelength] : filmBaseDensity[wavelength];
    const float scanIlluminant = scanIlluminantsAndCmfs[scanIlluminantOffset + wavelength];
    const float densitySpectral =
      densityCmy.x * cDensity +
      densityCmy.y * mDensity +
      densityCmy.z * yDensity +
      baseDensity +
      retainedSilver;
    const float light = finitef(spectralTransmittance(densitySpectral, params) * scanIlluminant);
    const float cmfY = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset + 1u];
    y += light * cmfY;
    normalization += scanIlluminant * cmfY;
  }
  return y / fmaxf(normalization, 1.0e-10f);
}

__device__ float densityCurveMax(uint32_t channel, bool paper, const KernelSpectralInfo &info) {
  if (channel == 0u) {
    return paper ? info.paperDensityCurveMaximum0 : info.filmDensityCurveMaximum0;
  }
  if (channel == 1u) {
    return paper ? info.paperDensityCurveMaximum1 : info.filmDensityCurveMaximum1;
  }
  return paper ? info.paperDensityCurveMaximum2 : info.filmDensityCurveMaximum2;
}

__device__ float3 densityCurveMaxCmy(bool paper, const KernelSpectralInfo &info) {
  return make_float3(
    fmaxf(densityCurveMax(0u, paper, info), 1.0e-6f),
    fmaxf(densityCurveMax(1u, paper, info), 1.0e-6f),
    fmaxf(densityCurveMax(2u, paper, info), 1.0e-6f));
}

__device__ float printReferenceY(
  bool blackReference,
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const KernelSpectralInfo &info,
  const KernelCurveInfo &filmCurveInfo,
  const KernelCurveInfo &paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz
) {
  const float3 filmBlack =
    make_float3(-params.grainDensityMinR, -params.grainDensityMinG, -params.grainDensityMinB);
  const float3 filmWhite = densityCurveMaxCmy(false, info);
  const float3 filmDensity = blackReference ? filmBlack : filmWhite;
  const float exposureFactor = printMidgrayExposureFactor(
    params,
    colorInfo,
    info,
    filmCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float3 preflash = printRawPreflash(
    params,
    info,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData);
  float3 raw = printRawFromFilmDensity(
    filmDensity,
    params,
    info,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData);
  raw = add3(mul3(mul3s(raw, exposureFactor), apdPrinterTimingExposureScale(params, info, academyPrinterDensityData)), preflash);
  raw = max3s(mul3s(raw, exp2f(params.printExposureEv)), 0.0f);
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  const float3 printLogRaw = make_float3(
    logf(raw.x + 1.0e-10f) * invLog10,
    logf(raw.y + 1.0e-10f) * invLog10,
    logf(raw.z + 1.0e-10f) * invLog10);
  const float3 printDensity =
    developPrintDensity(printLogRaw, params, info, paperCurveInfo, paperLogExposure, paperDensityCurves);
  return scanDensityToY(
    printDensity,
    0.0f,
    params,
    info,
    filmChannelDensity,
    filmBaseDensity,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    true);
}

__device__ float decodeSrgbScalar(float value) {
  value = clampf(value, 0.0f, 1.0f);
  return value <= 0.04045f ? value / 12.92f : powf((value + 0.055f) / 1.055f, 2.4f);
}

__device__ float scannerTargetLevel(bool correctionEnabled, float level, float referenceY) {
  return correctionEnabled ? decodeSrgbScalar(level) : referenceY;
}

__device__ float3 applyScannerBlackWhiteCorrection(
  float3 rgb,
  float sourceY,
  float referenceBlackY,
  float referenceWhiteY,
  const KernelParams &params
) {
  if (params.scannerEnabled == 0u || (params.scannerBlackCorrection == 0u && params.scannerWhiteCorrection == 0u)) {
    return rgb;
  }
  const float blackLevel = scannerTargetLevel(params.scannerBlackCorrection != 0u, params.scannerBlackLevel, referenceBlackY);
  const float whiteLevel = scannerTargetLevel(params.scannerWhiteCorrection != 0u, params.scannerWhiteLevel, referenceWhiteY);
  const float m = (whiteLevel - blackLevel) / fmaxf(referenceWhiteY - referenceBlackY, 1.0e-10f);
  const float q = blackLevel - m * referenceBlackY;
  const float correctedY = clampf(m * sourceY + q, 0.0f, 1.0f);
  return mul3s(rgb, correctedY / fmaxf(sourceY, 1.0e-10f));
}

__device__ float3 scanIlluminantToOutputRgb(
  const KernelParams &params,
  const KernelColorInfo &colorInfo,
  const KernelSpectralInfo &info,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData
) {
  const uint32_t paperScanIlluminantOffset = info.filmWavelengthCount;
  const uint32_t standardObserverOffset = info.filmWavelengthCount * 2u;
  const uint32_t paperScanMatrixOffset = colorInfo.colorSpaceCount * 9u;
  float3 xyz = make_float3(0.0f, 0.0f, 0.0f);
  float normalization = 0.0f;
  for (uint32_t wavelength = 0u; wavelength < info.filmWavelengthCount; ++wavelength) {
    const uint32_t channelOffset = wavelength * 3u;
    const float scanIlluminant = scanIlluminantsAndCmfs[paperScanIlluminantOffset + wavelength];
    const float cmfX = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset];
    const float cmfY = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset + 1u];
    const float cmfZ = scanIlluminantsAndCmfs[standardObserverOffset + channelOffset + 2u];
    xyz.x += scanIlluminant * cmfX;
    xyz.y += scanIlluminant * cmfY;
    xyz.z += scanIlluminant * cmfZ;
    normalization += scanIlluminant * cmfY;
  }
  xyz = mul3s(xyz, 1.0f / fmaxf(normalization, 1.0e-10f));
  return mulScanMatrix(
    xyz,
    static_cast<int>(finalOutputColorSpace(params, colorInfo)),
    colorInfo,
    scanToOutputRgbData,
    paperScanMatrixOffset);
}

// film development and DIR density correction
// Film curve development and retained-silver density work.
__global__ void developFromRawKernel(
  const float *raw,
  float *density,
  int pixelCount,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *logExposure,
  const float *densityCurves
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  const float3 logRaw = make_float3(
    logf(fmaxf(raw[offset], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(raw[offset + 1u], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(raw[offset + 2u], 0.0f) + 1.0e-10f) * invLog10
  );
  const float3 d = developFilmDensity(logRaw, *params, *curveInfo, logExposure, densityCurves);
  density[offset] = d.x;
  density[offset + 1u] = d.y;
  density[offset + 2u] = d.z;
  density[offset + 3u] = raw[offset + 3u];
}

__device__ float4 add4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float4 mul4s(float4 a, float b) {
  return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}

__device__ KernelGaussianBlurInfo gaussianBlurInfoDevice(float sigma, uint32_t radiusLimit) {
  KernelGaussianBlurInfo info{};
  info.invWeightSum = 1.0f;
  if (sigma <= 1.0e-4f) {
    return info;
  }
  info.active = 1u;
  info.radius = min(static_cast<uint32_t>(ceilf(3.0f * fmaxf(sigma, 1.0e-6f))), radiusLimit);
  const float invSigma2 = 1.0f / fmaxf(sigma * sigma, 1.0e-8f);
  info.firstWeight = expf(-0.5f * invSigma2);
  info.firstRatio = expf(-1.5f * invSigma2);
  info.ratioStep = expf(-invSigma2);
  float weight = info.firstWeight;
  float ratio = info.firstRatio;
  float weightSum = 1.0f;
  for (uint32_t offset = 1u; offset <= info.radius; ++offset) {
    weightSum += 2.0f * weight;
    weight *= ratio;
    ratio *= info.ratioStep;
  }
  info.invWeightSum = 1.0f / fmaxf(weightSum, 1.0e-8f);
  return info;
}

__device__ float4 dirGaussianSampleX(
  const float *source,
  int width,
  int height,
  int x,
  int y,
  const KernelGaussianBlurInfo &info
) {
  const int index = y * width + x;
  const float4 center = sampleFloat4Clamped(source, x, y, width, height);
  if (info.active == 0u || info.radius == 0u) {
    return center;
  }
  const bool interior = x >= static_cast<int>(info.radius) &&
    x + static_cast<int>(info.radius) < width;
  float4 value = center;
  float weight = info.firstWeight;
  float ratio = info.firstRatio;
  for (uint32_t offset = 1u; offset <= info.radius; ++offset) {
    const float4 pair = interior
      ? add4(
          make_float4(
            source[(static_cast<size_t>(index) - offset) * 4u],
            source[(static_cast<size_t>(index) - offset) * 4u + 1u],
            source[(static_cast<size_t>(index) - offset) * 4u + 2u],
            source[(static_cast<size_t>(index) - offset) * 4u + 3u]),
          make_float4(
            source[(static_cast<size_t>(index) + offset) * 4u],
            source[(static_cast<size_t>(index) + offset) * 4u + 1u],
            source[(static_cast<size_t>(index) + offset) * 4u + 2u],
            source[(static_cast<size_t>(index) + offset) * 4u + 3u]))
      : add4(
          sampleFloat4Clamped(source, x - static_cast<int>(offset), y, width, height),
          sampleFloat4Clamped(source, x + static_cast<int>(offset), y, width, height));
    value = add4(value, mul4s(pair, weight));
    weight *= ratio;
    ratio *= info.ratioStep;
  }
  return mul4s(value, info.invWeightSum);
}

__device__ float4 dirGaussianSampleY(
  const float *source,
  int width,
  int height,
  int x,
  int y,
  const KernelGaussianBlurInfo &info
) {
  const int index = y * width + x;
  const float4 center = sampleFloat4Clamped(source, x, y, width, height);
  if (info.active == 0u || info.radius == 0u) {
    return center;
  }
  const bool interior = y >= static_cast<int>(info.radius) &&
    y + static_cast<int>(info.radius) < height;
  float4 value = center;
  float weight = info.firstWeight;
  float ratio = info.firstRatio;
  for (uint32_t offset = 1u; offset <= info.radius; ++offset) {
    const float4 pair = interior
      ? add4(
          make_float4(
            source[(static_cast<size_t>(index) - static_cast<size_t>(offset) * width) * 4u],
            source[(static_cast<size_t>(index) - static_cast<size_t>(offset) * width) * 4u + 1u],
            source[(static_cast<size_t>(index) - static_cast<size_t>(offset) * width) * 4u + 2u],
            source[(static_cast<size_t>(index) - static_cast<size_t>(offset) * width) * 4u + 3u]),
          make_float4(
            source[(static_cast<size_t>(index) + static_cast<size_t>(offset) * width) * 4u],
            source[(static_cast<size_t>(index) + static_cast<size_t>(offset) * width) * 4u + 1u],
            source[(static_cast<size_t>(index) + static_cast<size_t>(offset) * width) * 4u + 2u],
            source[(static_cast<size_t>(index) + static_cast<size_t>(offset) * width) * 4u + 3u]))
      : add4(
          sampleFloat4Clamped(source, x, y - static_cast<int>(offset), width, height),
          sampleFloat4Clamped(source, x, y + static_cast<int>(offset), width, height));
    value = add4(value, mul4s(pair, weight));
    weight *= ratio;
    ratio *= info.ratioStep;
  }
  return mul4s(value, info.invWeightSum);
}

// DIR coupler correction and its blur tails.
__global__ void dirCorrectionFromDensityKernel(
  const float *density,
  float *correction,
  int pixelCount,
  const KernelSpectralInfo *spectralInfo,
  const KernelDirInfo *dirInfo
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float3 silver = filmSilverDensity(
    make_float3(density[offset], density[offset + 1u], density[offset + 2u]),
    *spectralInfo);
  const KernelDirInfo info = *dirInfo;
  correction[offset] = silver.x * info.matrix00 + silver.y * info.matrix10 + silver.z * info.matrix20;
  correction[offset + 1u] = silver.x * info.matrix01 + silver.y * info.matrix11 + silver.z * info.matrix21;
  correction[offset + 2u] = silver.x * info.matrix02 + silver.y * info.matrix12 + silver.z * info.matrix22;
  correction[offset + 3u] = density[offset + 3u];
}

__global__ void dirBlurXKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 v = dirGaussianSampleX(source, width, height, x, y, *blurInfo);
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void dirBlurYKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 v = dirGaussianSampleY(source, width, height, x, y, *blurInfo);
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void dirTailBlurXKernel(
  const float *source,
  float *tailPlanes,
  int width,
  int height,
  const KernelGaussianBlurInfo *tailBlurInfos
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  for (uint32_t component = 0u; component < 3u; ++component) {
    const float4 v = dirGaussianSampleX(source, width, height, x, y, tailBlurInfos[component]);
    const size_t offset = (static_cast<size_t>(component) * pixelCount + static_cast<size_t>(index)) * 4u;
    tailPlanes[offset] = v.x;
    tailPlanes[offset + 1u] = v.y;
    tailPlanes[offset + 2u] = v.z;
    tailPlanes[offset + 3u] = v.w;
  }
}

__device__ float4 tailPlaneSampleClamped(
  const float *tailPlanes,
  int plane,
  int width,
  int height,
  int x,
  int y
) {
  const int sx = safePixelCoord(x, width);
  const int sy = safePixelCoord(y, height);
  const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t offset = (static_cast<size_t>(plane) * pixelCount + static_cast<size_t>(sy) * width + sx) * 4u;
  return make_float4(tailPlanes[offset], tailPlanes[offset + 1u], tailPlanes[offset + 2u], tailPlanes[offset + 3u]);
}

__device__ float4 tailGaussianSampleY(
  const float *tailPlanes,
  int plane,
  int width,
  int height,
  int x,
  int y,
  const KernelGaussianBlurInfo &info
) {
  const int index = y * width + x;
  const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
  const size_t planeBase = static_cast<size_t>(plane) * pixelCount;
  const size_t centerOffset = (planeBase + static_cast<size_t>(index)) * 4u;
  const float4 center = make_float4(
    tailPlanes[centerOffset],
    tailPlanes[centerOffset + 1u],
    tailPlanes[centerOffset + 2u],
    tailPlanes[centerOffset + 3u]);
  if (info.active == 0u || info.radius == 0u) {
    return center;
  }
  const bool interior = y >= static_cast<int>(info.radius) &&
    y + static_cast<int>(info.radius) < height;
  float4 value = center;
  float weight = info.firstWeight;
  float ratio = info.firstRatio;
  for (uint32_t offset = 1u; offset <= info.radius; ++offset) {
    const float4 pair = interior
      ? add4(
          tailPlaneSampleClamped(tailPlanes, plane, width, height, x, y - static_cast<int>(offset)),
          tailPlaneSampleClamped(tailPlanes, plane, width, height, x, y + static_cast<int>(offset)))
      : add4(
          tailPlaneSampleClamped(tailPlanes, plane, width, height, x, y - static_cast<int>(offset)),
          tailPlaneSampleClamped(tailPlanes, plane, width, height, x, y + static_cast<int>(offset)));
    value = add4(value, mul4s(pair, weight));
    weight *= ratio;
    ratio *= info.ratioStep;
  }
  return mul4s(value, info.invWeightSum);
}

__global__ void dirTailBlurYAccumulateKernel(
  const float *tailPlanes,
  float *correctionInOut,
  int width,
  int height,
  const KernelParams *params,
  const KernelGaussianBlurInfo *tailBlurInfos
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 b0 = tailGaussianSampleY(tailPlanes, 0, width, height, x, y, tailBlurInfos[0]);
  const float4 b1 = tailGaussianSampleY(tailPlanes, 1, width, height, x, y, tailBlurInfos[1]);
  const float4 b2 = tailGaussianSampleY(tailPlanes, 2, width, height, x, y, tailBlurInfos[2]);
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float tailWeight = clampf(params->dirCouplersDiffusionTailWeight, 0.0f, 1.0f);
  const float3 base = make_float3(correctionInOut[offset], correctionInOut[offset + 1u], correctionInOut[offset + 2u]);
  const float3 tail = make_float3(
    0.1633f * b0.x + 0.6496f * b1.x + 0.1870f * b2.x,
    0.1633f * b0.y + 0.6496f * b1.y + 0.1870f * b2.y,
    0.1633f * b0.z + 0.6496f * b1.z + 0.1870f * b2.z);
  correctionInOut[offset] = base.x * (1.0f - tailWeight) + tail.x * tailWeight;
  correctionInOut[offset + 1u] = base.y * (1.0f - tailWeight) + tail.y * tailWeight;
  correctionInOut[offset + 2u] = base.z * (1.0f - tailWeight) + tail.z * tailWeight;
}

__global__ void dirRedevelopKernel(
  const float *logRaw,
  const float *correction,
  float *density,
  int pixelCount,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *logExposure,
  const float *correctedDensityCurves
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float3 correctedLogRaw = make_float3(
    logRaw[offset] - correction[offset],
    logRaw[offset + 1u] - correction[offset + 1u],
    logRaw[offset + 2u] - correction[offset + 2u]);
  const float3 d = developFilmDensity(correctedLogRaw, *params, *curveInfo, logExposure, correctedDensityCurves);
  density[offset] = d.x;
  density[offset + 1u] = d.y;
  density[offset + 2u] = d.z;
  density[offset + 3u] = logRaw[offset + 3u];
}

// diffusion is a weighted set of separable Gaussian lobes, used by camera and print stages
// Diffusion pass family: blur components, accumulate, resolve.
__global__ void clearFrameKernel(float *destination, int pixelCount) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = 0.0f;
  destination[offset + 1u] = 0.0f;
  destination[offset + 2u] = 0.0f;
  destination[offset + 3u] = 0.0f;
}

__global__ void diffusionBlurXKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentIndex
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const KernelGaussianBlurInfo info = gaussianBlurInfoDevice(fmaxf(components[componentIndex].sigmaPx, 1.0e-6f), 256u);
  const float4 value = dirGaussianSampleX(source, width, height, x, y, info);
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  destination[outOffset] = value.x;
  destination[outOffset + 1u] = value.y;
  destination[outOffset + 2u] = value.z;
  destination[outOffset + 3u] = value.w;
}

__global__ void diffusionBlurYAccumulateKernel(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentIndex
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const KernelDiffusionComponent component = components[componentIndex];
  const KernelGaussianBlurInfo info = gaussianBlurInfoDevice(fmaxf(component.sigmaPx, 1.0e-6f), 256u);
  const float4 value = dirGaussianSampleY(source, width, height, x, y, info);
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  accum[outOffset] += value.x * component.weightR;
  accum[outOffset + 1u] += value.y * component.weightG;
  accum[outOffset + 2u] += value.z * component.weightB;
  accum[outOffset + 3u] = value.w;
}

__global__ void diffusionGroupBlurXKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const uint32_t groupCount = min(componentCount, 4u);
  for (uint32_t slot = 0u; slot < groupCount; ++slot) {
    const KernelDiffusionComponent component = components[componentStart + slot];
    const KernelGaussianBlurInfo info = gaussianBlurInfoDevice(fmaxf(component.sigmaPx, 1.0e-6f), 256u);
    const float4 value = dirGaussianSampleX(source, width, height, x, y, info);
    const size_t outOffset =
      (static_cast<size_t>(slot) * static_cast<size_t>(pixelCount) + static_cast<size_t>(index)) * 4u;
    destination[outOffset] = value.x;
    destination[outOffset + 1u] = value.y;
    destination[outOffset + 2u] = value.z;
    destination[outOffset + 3u] = value.w;
  }
}

__global__ void diffusionGroupBlurYAccumulateKernel(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const uint32_t groupCount = min(componentCount, 4u);
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  float4 accumValue = make_float4(accum[outOffset], accum[outOffset + 1u], accum[outOffset + 2u], accum[outOffset + 3u]);
  for (uint32_t slot = 0u; slot < groupCount; ++slot) {
    const KernelDiffusionComponent component = components[componentStart + slot];
    const float *plane = source + static_cast<size_t>(slot) * static_cast<size_t>(pixelCount) * 4u;
    const KernelGaussianBlurInfo info = gaussianBlurInfoDevice(fmaxf(component.sigmaPx, 1.0e-6f), 256u);
    const float4 value = dirGaussianSampleY(plane, width, height, x, y, info);
    accumValue.x += value.x * component.weightR;
    accumValue.y += value.y * component.weightG;
    accumValue.z += value.z * component.weightB;
    accumValue.w = value.w;
  }
  accum[outOffset] = accumValue.x;
  accum[outOffset + 1u] = accumValue.y;
  accum[outOffset + 2u] = accumValue.z;
  accum[outOffset + 3u] = accumValue.w;
}

__global__ void diffusionDownsampleKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  uint32_t scale
) {
  const uint32_t safeScale = max(scale, 1u);
  const int reducedWidth = (width + static_cast<int>(safeScale) - 1) / static_cast<int>(safeScale);
  const int reducedHeight = (height + static_cast<int>(safeScale) - 1) / static_cast<int>(safeScale);
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int reducedPixelCount = reducedWidth * reducedHeight;
  if (index >= reducedPixelCount) {
    return;
  }
  const int rx = index % reducedWidth;
  const int ry = index / reducedWidth;
  const int baseX = rx * static_cast<int>(safeScale);
  const int baseY = ry * static_cast<int>(safeScale);
  float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  for (uint32_t oy = 0u; oy < safeScale; ++oy) {
    for (uint32_t ox = 0u; ox < safeScale; ++ox) {
      const float4 sample = sampleFloat4Clamped(
        source,
        baseX + static_cast<int>(ox),
        baseY + static_cast<int>(oy),
        width,
        height);
      value = add4(value, sample);
      weightSum += 1.0f;
    }
  }
  const float invWeight = 1.0f / fmaxf(weightSum, 1.0e-8f);
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  destination[outOffset] = value.x * invWeight;
  destination[outOffset + 1u] = value.y * invWeight;
  destination[outOffset + 2u] = value.z * invWeight;
  destination[outOffset + 3u] = value.w * invWeight;
}

__global__ void diffusionReducedGroupBlurXKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float sigmaScale = 1.0f / static_cast<float>(max(scale, 1u));
  const uint32_t groupCount = min(componentCount, 4u);
  for (uint32_t slot = 0u; slot < groupCount; ++slot) {
    const KernelDiffusionComponent component = components[componentStart + slot];
    const KernelGaussianBlurInfo info =
      gaussianBlurInfoDevice(fmaxf(component.sigmaPx * sigmaScale, 1.0e-6f), 256u);
    const float4 value = dirGaussianSampleX(source, width, height, x, y, info);
    const size_t outOffset =
      (static_cast<size_t>(slot) * static_cast<size_t>(pixelCount) + static_cast<size_t>(index)) * 4u;
    destination[outOffset] = value.x;
    destination[outOffset + 1u] = value.y;
    destination[outOffset + 2u] = value.z;
    destination[outOffset + 3u] = value.w;
  }
}

__global__ void diffusionReducedGroupBlurYKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount
) {
  (void)scale;
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const uint32_t groupCount = min(componentCount, 4u);
  for (uint32_t slot = 0u; slot < groupCount; ++slot) {
    const KernelDiffusionComponent component = components[componentStart + slot];
    const float sigmaScale = 1.0f / static_cast<float>(max(scale, 1u));
    const KernelGaussianBlurInfo info =
      gaussianBlurInfoDevice(fmaxf(component.sigmaPx * sigmaScale, 1.0e-6f), 256u);
    const float *plane = source + static_cast<size_t>(slot) * static_cast<size_t>(pixelCount) * 4u;
    const float4 value = dirGaussianSampleY(plane, width, height, x, y, info);
    const size_t outOffset =
      (static_cast<size_t>(slot) * static_cast<size_t>(pixelCount) + static_cast<size_t>(index)) * 4u;
    destination[outOffset] = value.x;
    destination[outOffset + 1u] = value.y;
    destination[outOffset + 2u] = value.z;
    destination[outOffset + 3u] = value.w;
  }
}

__device__ float4 reducedBilinearSample(const float *source, int width, int height, float x, float y) {
  const float maxX = fmaxf(static_cast<float>(width) - 1.0f, 0.0f);
  const float maxY = fmaxf(static_cast<float>(height) - 1.0f, 0.0f);
  const float cx = clampf(x, 0.0f, maxX);
  const float cy = clampf(y, 0.0f, maxY);
  const int x0 = static_cast<int>(floorf(cx));
  const int y0 = static_cast<int>(floorf(cy));
  const int x1 = min(x0 + 1, max(width, 1) - 1);
  const int y1 = min(y0 + 1, max(height, 1) - 1);
  const float tx = cx - static_cast<float>(x0);
  const float ty = cy - static_cast<float>(y0);
  const float4 p00 = sampleFloat4Clamped(source, x0, y0, width, height);
  const float4 p10 = sampleFloat4Clamped(source, x1, y0, width, height);
  const float4 p01 = sampleFloat4Clamped(source, x0, y1, width, height);
  const float4 p11 = sampleFloat4Clamped(source, x1, y1, width, height);
  const float4 a = make_float4(
    mixf(p00.x, p10.x, tx),
    mixf(p00.y, p10.y, tx),
    mixf(p00.z, p10.z, tx),
    mixf(p00.w, p10.w, tx));
  const float4 b = make_float4(
    mixf(p01.x, p11.x, tx),
    mixf(p01.y, p11.y, tx),
    mixf(p01.z, p11.z, tx),
    mixf(p01.w, p11.w, tx));
  return make_float4(
    mixf(a.x, b.x, ty),
    mixf(a.y, b.y, ty),
    mixf(a.z, b.z, ty),
    mixf(a.w, b.w, ty));
}

__global__ void diffusionReducedGroupUpsampleAccumulateKernel(
  const float *source,
  float *accum,
  int width,
  int height,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const uint32_t safeScale = max(scale, 1u);
  const int reducedWidth = (width + static_cast<int>(safeScale) - 1) / static_cast<int>(safeScale);
  const int reducedHeight = (height + static_cast<int>(safeScale) - 1) / static_cast<int>(safeScale);
  const int x = index % width;
  const int y = index / width;
  const float sampleX = (static_cast<float>(x) + 0.5f) / static_cast<float>(safeScale) - 0.5f;
  const float sampleY = (static_cast<float>(y) + 0.5f) / static_cast<float>(safeScale) - 0.5f;
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  float4 accumValue = make_float4(accum[outOffset], accum[outOffset + 1u], accum[outOffset + 2u], accum[outOffset + 3u]);
  const uint32_t groupCount = min(componentCount, 4u);
  const size_t reducedPixelCount = static_cast<size_t>(reducedWidth) * static_cast<size_t>(reducedHeight);
  for (uint32_t slot = 0u; slot < groupCount; ++slot) {
    const KernelDiffusionComponent component = components[componentStart + slot];
    const float *plane = source + static_cast<size_t>(slot) * reducedPixelCount * 4u;
    const float4 value = reducedBilinearSample(plane, reducedWidth, reducedHeight, sampleX, sampleY);
    accumValue.x += value.x * component.weightR;
    accumValue.y += value.y * component.weightG;
    accumValue.z += value.z * component.weightB;
    accumValue.w = value.w;
  }
  accum[outOffset] = accumValue.x;
  accum[outOffset + 1u] = accumValue.y;
  accum[outOffset + 2u] = accumValue.z;
  accum[outOffset + 3u] = accumValue.w;
}

__global__ void diffusionResolveKernel(
  const float *source,
  const float *accum,
  float *destination,
  int pixelCount,
  const KernelDiffusionInfo *info
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float scatter = clampf(info->scatterFraction, 0.0f, 0.99f);
  destination[offset] = (1.0f - scatter) * source[offset] + scatter * accum[offset];
  destination[offset + 1u] = (1.0f - scatter) * source[offset + 1u] + scatter * accum[offset + 1u];
  destination[offset + 2u] = (1.0f - scatter) * source[offset + 2u] + scatter * accum[offset + 2u];
  destination[offset + 3u] = source[offset + 3u];
}

__device__ float maxComponent(float3 value) {
  return fmaxf(value.x, fmaxf(value.y, value.z));
}

__device__ float3 halationScatterCoreSigma(const KernelParams &params) {
  const float scale = fmaxf(params.scatterScale, 0.0f) / fmaxf(params.filmPixelSizeUm, 1.0e-6f);
  return make_float3(2.2f * scale, 2.0f * scale, 1.6f * scale);
}

__device__ float3 halationScatterTailSigma(const KernelParams &params, uint32_t component) {
  const float ratios[3] = {0.5360f, 1.5236f, 2.7684f};
  const float ratio = ratios[component > 2u ? 2u : component];
  const float scale = ratio * fmaxf(params.scatterScale, 0.0f) / fmaxf(params.filmPixelSizeUm, 1.0e-6f);
  return make_float3(9.3f * scale, 9.7f * scale, 9.1f * scale);
}

__device__ float3 halationFirstSigma(const KernelParams &params, uint32_t bounce) {
  const float scale = fmaxf(params.halationScale, 0.0f) *
    sqrtf(static_cast<float>(bounce) + 1.0f) /
    fmaxf(params.filmPixelSizeUm, 1.0e-6f);
  return make_float3(
    params.halationFirstSigmaUmR * scale,
    params.halationFirstSigmaUmG * scale,
    params.halationFirstSigmaUmB * scale);
}

__device__ float3 halationSigmaForMode(const KernelParams &params, uint32_t mode, uint32_t component) {
  if (mode == 0u) {
    return halationScatterCoreSigma(params);
  }
  if (mode == 1u) {
    return halationScatterTailSigma(params, component);
  }
  return halationFirstSigma(params, component);
}

__device__ void halationWeights(float sigma, float &weight, float &ratio, float &ratioStep) {
  if (sigma <= 1.0e-4f) {
    weight = 0.0f;
    ratio = 0.0f;
    ratioStep = 0.0f;
    return;
  }
  const float invSigma2 = 1.0f / fmaxf(sigma * sigma, 1.0e-8f);
  weight = expf(-0.5f * invSigma2);
  ratio = expf(-1.5f * invSigma2);
  ratioStep = expf(-invSigma2);
}

__device__ float4 halationChannelGaussianSampleX(
  const float *source,
  int width,
  int height,
  int x,
  int y,
  float3 sigma
) {
  const float4 center = sampleFloat4Clamped(source, x, y, width, height);
  const int radius = min(static_cast<int>(ceilf(3.0f * maxComponent(sigma))), 256);
  if (radius <= 0) {
    return center;
  }
  float3 value = make_float3(center.x, center.y, center.z);
  float3 weightSum = make_float3(1.0f, 1.0f, 1.0f);
  float3 weight;
  float3 ratio;
  float3 ratioStep;
  halationWeights(sigma.x, weight.x, ratio.x, ratioStep.x);
  halationWeights(sigma.y, weight.y, ratio.y, ratioStep.y);
  halationWeights(sigma.z, weight.z, ratio.z, ratioStep.z);
  float alpha = center.w;
  float alphaCount = 1.0f;
  for (int offset = 1; offset <= radius; ++offset) {
    const float4 left = sampleFloat4Clamped(source, x - offset, y, width, height);
    const float4 right = sampleFloat4Clamped(source, x + offset, y, width, height);
    value.x += (left.x + right.x) * weight.x;
    value.y += (left.y + right.y) * weight.y;
    value.z += (left.z + right.z) * weight.z;
    weightSum.x += 2.0f * weight.x;
    weightSum.y += 2.0f * weight.y;
    weightSum.z += 2.0f * weight.z;
    alpha += left.w + right.w;
    alphaCount += 2.0f;
    weight.x *= ratio.x;
    weight.y *= ratio.y;
    weight.z *= ratio.z;
    ratio.x *= ratioStep.x;
    ratio.y *= ratioStep.y;
    ratio.z *= ratioStep.z;
  }
  return make_float4(
    value.x / fmaxf(weightSum.x, 1.0e-8f),
    value.y / fmaxf(weightSum.y, 1.0e-8f),
    value.z / fmaxf(weightSum.z, 1.0e-8f),
    alpha / fmaxf(alphaCount, 1.0f));
}

__device__ float4 halationChannelGaussianSampleY(
  const float *source,
  int width,
  int height,
  int x,
  int y,
  float3 sigma
) {
  const float4 center = sampleFloat4Clamped(source, x, y, width, height);
  const int radius = min(static_cast<int>(ceilf(3.0f * maxComponent(sigma))), 256);
  if (radius <= 0) {
    return center;
  }
  float3 value = make_float3(center.x, center.y, center.z);
  float3 weightSum = make_float3(1.0f, 1.0f, 1.0f);
  float3 weight;
  float3 ratio;
  float3 ratioStep;
  halationWeights(sigma.x, weight.x, ratio.x, ratioStep.x);
  halationWeights(sigma.y, weight.y, ratio.y, ratioStep.y);
  halationWeights(sigma.z, weight.z, ratio.z, ratioStep.z);
  float alpha = center.w;
  float alphaCount = 1.0f;
  for (int offset = 1; offset <= radius; ++offset) {
    const float4 top = sampleFloat4Clamped(source, x, y - offset, width, height);
    const float4 bottom = sampleFloat4Clamped(source, x, y + offset, width, height);
    value.x += (top.x + bottom.x) * weight.x;
    value.y += (top.y + bottom.y) * weight.y;
    value.z += (top.z + bottom.z) * weight.z;
    weightSum.x += 2.0f * weight.x;
    weightSum.y += 2.0f * weight.y;
    weightSum.z += 2.0f * weight.z;
    alpha += top.w + bottom.w;
    alphaCount += 2.0f;
    weight.x *= ratio.x;
    weight.y *= ratio.y;
    weight.z *= ratio.z;
    ratio.x *= ratioStep.x;
    ratio.y *= ratioStep.y;
    ratio.z *= ratioStep.z;
  }
  return make_float4(
    value.x / fmaxf(weightSum.x, 1.0e-8f),
    value.y / fmaxf(weightSum.y, 1.0e-8f),
    value.z / fmaxf(weightSum.z, 1.0e-8f),
    alpha / fmaxf(alphaCount, 1.0f));
}

// halation: highlight boost, scatter tail, then emulsion bounce
// Halation pass family: boost, scatter, bounce, resolve back to raw.
__global__ void halationBoostInfoKernel(
  const float *raw,
  float *boostInfo,
  int pixelCount,
  const KernelParams *params
) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  float frameMax = 0.0f;
  for (int index = 0; index < pixelCount; ++index) {
    const size_t offset = static_cast<size_t>(index) * 4u;
    frameMax = fmaxf(frameMax, fmaxf(raw[offset], fmaxf(raw[offset + 1u], raw[offset + 2u])));
  }
  const KernelParams p = *params;
  const float rawX0 = clampf(0.184f * exp2f(p.halationProtectEv), 0.0f, frameMax);
  const float boostRange = clampf(p.halationBoostRange, 0.0f, 1.0f);
  const float a = powf(28.0f, 1.0f - boostRange);
  const float x0 = frameMax > 0.0f ? rawX0 / frameMax : 1.0f;
  const float dx = 1.0f - x0;
  const float denom = expf(a * dx) - a * dx - 1.0f;
  const float k = (frameMax > 0.0f && rawX0 < frameMax && denom > 1.0e-10f)
    ? (exp2f(fmaxf(p.halationBoostEv, 0.0f)) - 1.0f) / denom
    : 0.0f;
  boostInfo[0] = frameMax;
  boostInfo[1] = rawX0;
  boostInfo[2] = a;
  boostInfo[3] = k;
}

__device__ float halationBoostChannel(float value, const float *boostInfo) {
  const float frameMax = boostInfo[0];
  const float rawX0 = boostInfo[1];
  const float a = boostInfo[2];
  const float k = boostInfo[3];
  if (frameMax <= 0.0f || value <= rawX0 || k <= 0.0f) {
    return value;
  }
  const float dx = (value - rawX0) / frameMax;
  return value + k * frameMax * (expf(a * dx) - a * dx - 1.0f);
}

__global__ void halationBoostApplyKernel(
  const float *raw,
  const float *boostInfo,
  float *destination,
  int pixelCount
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = halationBoostChannel(raw[offset], boostInfo);
  destination[offset + 1u] = halationBoostChannel(raw[offset + 1u], boostInfo);
  destination[offset + 2u] = halationBoostChannel(raw[offset + 2u], boostInfo);
  destination[offset + 3u] = raw[offset + 3u];
}

__global__ void halationChannelBlurXKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  uint32_t mode,
  uint32_t component
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 v = halationChannelGaussianSampleX(source, width, height, x, y, halationSigmaForMode(*params, mode, component));
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void halationChannelBlurYKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  uint32_t mode,
  uint32_t component
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 v = halationChannelGaussianSampleY(source, width, height, x, y, halationSigmaForMode(*params, mode, component));
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void halationScatterTailBlurYAccumulateKernel(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelParams *params,
  uint32_t component
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float tailWeights[3] = {0.1633f, 0.6496f, 0.1870f};
  const float weight = tailWeights[component > 2u ? 2u : component];
  const float4 v = halationChannelGaussianSampleY(source, width, height, x, y, halationScatterTailSigma(*params, component));
  const size_t offset = static_cast<size_t>(index) * 4u;
  accum[offset] += v.x * weight;
  accum[offset + 1u] += v.y * weight;
  accum[offset + 2u] += v.z * weight;
  accum[offset + 3u] = v.w;
}

__global__ void halationScatterResolveKernel(
  const float *raw,
  const float *core,
  const float *tail,
  float *destination,
  int pixelCount,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float amount = clampf(params->scatterAmount, 0.0f, 1.0f);
  const float3 tailWeight = make_float3(0.78f, 0.65f, 0.67f);
  const float3 scattered = make_float3(
    mixf(core[offset], tail[offset], tailWeight.x),
    mixf(core[offset + 1u], tail[offset + 1u], tailWeight.y),
    mixf(core[offset + 2u], tail[offset + 2u], tailWeight.z));
  destination[offset] = mixf(raw[offset], scattered.x, amount);
  destination[offset + 1u] = mixf(raw[offset + 1u], scattered.y, amount);
  destination[offset + 2u] = mixf(raw[offset + 2u], scattered.z, amount);
  destination[offset + 3u] = raw[offset + 3u];
}

__global__ void halationBounceBlurYAccumulateKernel(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelParams *params,
  uint32_t bounce
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float4 v = halationChannelGaussianSampleY(source, width, height, x, y, halationFirstSigma(*params, bounce));
  const float weight = powf(0.5f, static_cast<float>(bounce)) / 1.75f;
  const size_t offset = static_cast<size_t>(index) * 4u;
  accum[offset] += v.x * weight;
  accum[offset + 1u] += v.y * weight;
  accum[offset + 2u] += v.z * weight;
  accum[offset + 3u] = v.w;
}

__global__ void halationResolveRawKernel(
  const float *raw,
  const float *halation,
  float *destination,
  int pixelCount,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const KernelParams p = *params;
  const float3 amount = make_float3(
    fmaxf(p.halationStrengthR, 0.0f) * fmaxf(p.halationAmount, 0.0f),
    fmaxf(p.halationStrengthG, 0.0f) * fmaxf(p.halationAmount, 0.0f),
    fmaxf(p.halationStrengthB, 0.0f) * fmaxf(p.halationAmount, 0.0f));
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = (raw[offset] + amount.x * halation[offset]) / (1.0f + amount.x);
  destination[offset + 1u] = (raw[offset + 1u] + amount.y * halation[offset + 1u]) / (1.0f + amount.y);
  destination[offset + 2u] = (raw[offset + 2u] + amount.z * halation[offset + 2u]) / (1.0f + amount.z);
  destination[offset + 3u] = raw[offset + 3u];
}

__device__ uint32_t grainHash(uint32_t x) {
  x ^= x >> 16u;
  x *= 0x7feb352du;
  x ^= x >> 15u;
  x *= 0x846ca68bu;
  x ^= x >> 16u;
  return x;
}

__device__ float grainRand01(uint32_t seed) {
  return (static_cast<float>(grainHash(seed) & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

__device__ float grainGaussian(uint32_t seed) {
  const float u1 = fmaxf(grainRand01(seed), 1.0e-6f);
  const float u2 = grainRand01(seed ^ 0x9e3779b9u);
  constexpr float twoPi = 6.28318530718f;
  return sqrtf(-2.0f * logf(u1)) * cosf(twoPi * u2);
}

__device__ float grainFilmFormatMm(int format) {
  switch (format) {
    case 0: return 4.8f;
    case 1: return 5.79f;
    case 2: return 10.26f;
    case 3: return 12.52f;
    case 4: return 16.0f;
    case 5: return 21.95f;
    case 6: return 35.0f;
    case 7: return 54.78f;
    case 8: return 65.0f;
    case 9: return 70.0f;
    default: return 35.0f;
  }
}

__device__ float grainChannelDensityMin(uint32_t channel, const KernelParams &params) {
  const float value = channel == 0u ? params.grainDensityMinR :
    (channel == 1u ? params.grainDensityMinG : params.grainDensityMinB);
  return fmaxf(value, 0.0f);
}

__device__ float grainChannelParticleScale(uint32_t channel, const KernelParams &params) {
  const float value = channel == 0u ? params.grainParticleScaleR :
    (channel == 1u ? params.grainParticleScaleG : params.grainParticleScaleB);
  return fmaxf(value, 1.0e-3f);
}

__device__ float grainChannelUniformity(uint32_t channel, const KernelParams &params) {
  const float value = channel == 0u ? params.grainUniformityR :
    (channel == 1u ? params.grainUniformityG : params.grainUniformityB);
  return clampf(value, 0.0f, 1.0f);
}

__device__ float grainDensityCurveMax(uint32_t channel, const KernelCurveInfo &curveInfo, const float *densityCurves) {
  float maximum = 0.0f;
  for (uint32_t i = 0u; i < curveInfo.exposureCount; ++i) {
    maximum = fmaxf(maximum, densityCurves[i * 3u + channel]);
  }
  return fmaxf(maximum, 1.0e-6f);
}

__device__ uint32_t grainFilmCellSeed(float filmUmX, float filmUmY, float cellSizeUm, uint32_t seed) {
  const float safeCellSize = fmaxf(cellSizeUm, 1.0e-4f);
  const int32_t cellX = static_cast<int32_t>(floorf(filmUmX / safeCellSize));
  const int32_t cellY = static_cast<int32_t>(floorf(filmUmY / safeCellSize));
  return grainHash(seed ^ (static_cast<uint32_t>(cellX) * 0x1f123bb5u) ^ (static_cast<uint32_t>(cellY) * 0x5f356495u));
}

__device__ uint32_t grainMixSeed(uint32_t seed, uint32_t value) {
  return grainHash(seed ^ (value + 0x9e3779b9u + (seed << 6u) + (seed >> 2u)));
}

__device__ uint32_t grainCellSeed(int cellX, int cellY, uint32_t channel, uint32_t layer, const KernelParams &params) {
  uint32_t seed = grainMixSeed(params.grainSeed, static_cast<uint32_t>(cellX));
  seed = grainMixSeed(seed, static_cast<uint32_t>(cellY));
  seed = grainMixSeed(seed, channel * 0x85ebca6bu);
  seed = grainMixSeed(seed, layer * 0xc2b2ae35u);
  const uint32_t frameSeed = params.grainAnimate != 0u ? static_cast<uint32_t>(floorf(params.time * 24.0f + 0.5f)) : 0u;
  return grainMixSeed(seed, frameSeed);
}

__device__ uint32_t grainPoissonSample(float lambda, uint32_t seed, uint32_t cap) {
  if (lambda <= 0.0f || cap == 0u) {
    return 0u;
  }
  if (lambda < 1.0e-5f) {
    return grainRand01(seed ^ 0x4cf5ad43u) < lambda ? 1u : 0u;
  }
  if (lambda < 8.0f) {
    const float threshold = expf(-lambda);
    float product = 1.0f;
    uint32_t k = 0u;
    while (k < cap) {
      product *= grainRand01(seed ^ (k * 0x27d4eb2du));
      if (product <= threshold) {
        break;
      }
      ++k;
    }
    return min(k, cap);
  }
  const float poissonDraw = floorf(lambda + sqrtf(lambda) * grainGaussian(seed ^ 0x165667b1u) + 0.5f);
  return static_cast<uint32_t>(clampf(poissonDraw, 0.0f, static_cast<float>(cap)));
}

__device__ float grainNormalQuantile(float p) {
  p = clampf(p, 1.0e-6f, 1.0f - 1.0e-6f);
  constexpr float a1 = -3.969683028665376e+01f;
  constexpr float a2 = 2.209460984245205e+02f;
  constexpr float a3 = -2.759285104469687e+02f;
  constexpr float a4 = 1.383577518672690e+02f;
  constexpr float a5 = -3.066479806614716e+01f;
  constexpr float a6 = 2.506628277459239e+00f;
  constexpr float b1 = -5.447609879822406e+01f;
  constexpr float b2 = 1.615858368580409e+02f;
  constexpr float b3 = -1.556989798598866e+02f;
  constexpr float b4 = 6.680131188771972e+01f;
  constexpr float b5 = -1.328068155288572e+01f;
  constexpr float c1 = -7.784894002430293e-03f;
  constexpr float c2 = -3.223964580411365e-01f;
  constexpr float c3 = -2.400758277161838e+00f;
  constexpr float c4 = -2.549732539343734e+00f;
  constexpr float c5 = 4.374664141464968e+00f;
  constexpr float c6 = 2.938163982698783e+00f;
  constexpr float d1 = 7.784695709041462e-03f;
  constexpr float d2 = 3.224671290700398e-01f;
  constexpr float d3 = 2.445134137142996e+00f;
  constexpr float d4 = 3.754408661907416e+00f;
  constexpr float pLow = 0.02425f;
  constexpr float pHigh = 1.0f - pLow;
  if (p < pLow) {
    const float q = sqrtf(-2.0f * logf(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
  }
  if (p > pHigh) {
    const float q = sqrtf(-2.0f * logf(1.0f - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0f);
  }
  const float q = p - 0.5f;
  const float r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
    (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0f);
}

__device__ float grainParticleDevelopedDensity(
  float density,
  float densityMax,
  float particles,
  float uniformity,
  float blurDamping,
  uint32_t seed
) {
  const float safeDensityMax = fmaxf(densityMax, 1.0e-6f);
  const float safeParticles = fmaxf(particles, 1.0e-3f);
  const float probability = clampf(density / safeDensityMax, 1.0e-6f, 1.0f - 1.0e-6f);
  const float saturation = fmaxf(1.0f - probability * uniformity * (1.0f - 1.0e-6f), 1.0e-6f);
  const float expectedSeeds = safeParticles / saturation;
  const float expectedDeveloped = expectedSeeds * probability;
  const float variance = fmaxf(expectedSeeds * probability, 1.0e-6f);
  const float developed = clampf(
    expectedDeveloped + sqrtf(variance) * grainGaussian(seed) * blurDamping,
    0.0f,
    expectedSeeds);
  return developed * (safeDensityMax / safeParticles) * saturation;
}

__device__ float3 applyGrainControls(float3 baseDensity, float3 grainedDensity, const KernelParams &params) {
  const float amount = fmaxf(params.grainAmount, 0.0f);
  const float saturation = clampf(params.grainSaturation, 0.0f, 1.0f);
  if (amount == 1.0f && saturation == 1.0f) {
    return grainedDensity;
  }
  float3 delta = mul3s(sub3(grainedDensity, baseDensity), amount);
  const float neutral = (delta.x + delta.y + delta.z) / 3.0f;
  delta = mix3(make_float3(neutral, neutral, neutral), delta, saturation);
  return max3s(add3(baseDensity, delta), 0.0f);
}

__device__ float grainLayerParticleScale(uint32_t layer, const KernelParams &params) {
  const float value = layer == 0u ? params.grainParticleScaleLayer0 :
    (layer == 1u ? params.grainParticleScaleLayer1 : params.grainParticleScaleLayer2);
  return fmaxf(value, 1.0e-3f);
}

__device__ float interpDensityLayer(
  float density,
  uint32_t channel,
  uint32_t layer,
  const KernelCurveInfo &curveInfo,
  const KernelSpectralInfo &info,
  const float *densityCurves,
  const float *paperScanDensityData
) {
  const uint32_t count = curveInfo.exposureCount;
  if (count == 0u) {
    return 0.0f;
  }
  const float target = info.filmPositive != 0u ? -density : density;
  const float firstX = info.filmPositive != 0u ? -densityCurves[channel] : densityCurves[channel];
  const float lastX = info.filmPositive != 0u
    ? -densityCurves[(count - 1u) * 3u + channel]
    : densityCurves[(count - 1u) * 3u + channel];
  const bool ascending = lastX >= firstX;
  const uint32_t layerOffset = info.filmWavelengthCount * 4u;
  if ((ascending && target <= firstX) || (!ascending && target >= firstX)) {
    return fmaxf(paperScanDensityData[layerOffset + layer * 3u + channel], 0.0f);
  }
  if ((ascending && target >= lastX) || (!ascending && target <= lastX)) {
    return fmaxf(paperScanDensityData[layerOffset + (count - 1u) * 9u + layer * 3u + channel], 0.0f);
  }

  uint32_t lo = 0u;
  uint32_t hi = count - 1u;
  while (hi - lo > 1u) {
    const uint32_t mid = (lo + hi) >> 1u;
    const float x = info.filmPositive != 0u ? -densityCurves[mid * 3u + channel] : densityCurves[mid * 3u + channel];
    if ((ascending && x <= target) || (!ascending && x >= target)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const float x0 = info.filmPositive != 0u ? -densityCurves[lo * 3u + channel] : densityCurves[lo * 3u + channel];
  const float x1 = info.filmPositive != 0u ? -densityCurves[hi * 3u + channel] : densityCurves[hi * 3u + channel];
  const float y0 = paperScanDensityData[layerOffset + lo * 9u + layer * 3u + channel];
  const float y1 = paperScanDensityData[layerOffset + hi * 9u + layer * 3u + channel];
  const float t = clampf((target - x0) / fmaxf(x1 - x0, 1.0e-9f), 0.0f, 1.0f);
  return fmaxf(mixf(y0, y1, t), 0.0f);
}

__device__ float densityCurveLayerMaximum(
  uint32_t layer,
  uint32_t channel,
  const KernelCurveInfo &curveInfo,
  const KernelSpectralInfo &info,
  const float *paperScanDensityData
) {
  const uint32_t maximaOffset = info.filmWavelengthCount * 4u + curveInfo.exposureCount * 9u;
  return paperScanDensityData[maximaOffset + layer * 3u + channel];
}

__device__ float layerDensityMaxTotal(
  uint32_t channel,
  const KernelCurveInfo &curveInfo,
  const KernelSpectralInfo &info,
  const float *paperScanDensityData
) {
  float total = 0.0f;
  for (uint32_t layer = 0u; layer < 3u; ++layer) {
    total += fmaxf(densityCurveLayerMaximum(layer, channel, curveInfo, info, paperScanDensityData), 0.0f);
  }
  return fmaxf(total, 1.0e-6f);
}

__device__ float channelComponent(float3 value, uint32_t channel) {
  return channel == 0u ? value.x : (channel == 1u ? value.y : value.z);
}

__device__ void outputPixelFilmUm(
  int x,
  int y,
  int width,
  int height,
  const KernelParams &params,
  float &filmUmX,
  float &filmUmY
) {
  const float scale = fmaxf(params.enlargerScale, 1.0f);
  const float safeWidth = static_cast<float>(width > 0 ? width : 1);
  const float safeHeight = static_cast<float>(height > 0 ? height : 1);
  const float outputUvX = (static_cast<float>(x) + 0.5f) / safeWidth;
  const float outputUvY = (static_cast<float>(y) + 0.5f) / safeHeight;
  const float sourceUvX = 0.5f + (outputUvX - 0.5f) / scale + params.enlargerOffsetXPercent * (0.01f / scale);
  const float sourceUvY = 0.5f + (outputUvY - 0.5f) / scale + params.enlargerOffsetYPercent * (0.01f / scale);
  const float framePixelSizeUm = fmaxf(params.filmPixelSizeUm * scale, 1.0e-6f);
  filmUmX = sourceUvX * safeWidth * framePixelSizeUm;
  filmUmY = sourceUvY * safeHeight * framePixelSizeUm;
}

__device__ void filmUmToOutputPixel(
  float filmUmX,
  float filmUmY,
  int width,
  int height,
  const KernelParams &params,
  int &outX,
  int &outY
) {
  const float scale = fmaxf(params.enlargerScale, 1.0f);
  const float safeWidth = static_cast<float>(width > 0 ? width : 1);
  const float safeHeight = static_cast<float>(height > 0 ? height : 1);
  const float framePixelSizeUm = fmaxf(params.filmPixelSizeUm * scale, 1.0e-6f);
  const float sourceUvX = filmUmX / (safeWidth * framePixelSizeUm);
  const float sourceUvY = filmUmY / (safeHeight * framePixelSizeUm);
  const float outputUvX = 0.5f + (sourceUvX - 0.5f) * scale - params.enlargerOffsetXPercent * 0.01f;
  const float outputUvY = 0.5f + (sourceUvY - 0.5f) * scale - params.enlargerOffsetYPercent * 0.01f;
  outX = static_cast<int>(floorf(outputUvX * safeWidth));
  outY = static_cast<int>(floorf(outputUvY * safeHeight));
}

__device__ float grainSynthesisChannelScale(uint32_t channel, const KernelParams &params) {
  const float value = channel == 0u ? params.grainSynthesisRadiusScaleR :
    (channel == 1u ? params.grainSynthesisRadiusScaleG : params.grainSynthesisRadiusScaleB);
  return fmaxf(value, 1.0e-6f);
}

__device__ float grainSynthesisLayerScale(uint32_t layer, const KernelParams &params) {
  if (params.grainSynthesisLayered == 0u) {
    return 1.0f;
  }
  const float value = layer == 0u ? params.grainSynthesisLayerScale0 :
    (layer == 1u ? params.grainSynthesisLayerScale1 : params.grainSynthesisLayerScale2);
  return fmaxf(value, 1.0e-6f);
}

struct GrainSynthesisEval {
  float scaledMeanRadius;
  float maxRadius;
  float maxRadiusSquared;
  float cellSize;
  float meanArea;
  float cellArea;
  float densityToLambda;
  uint32_t grainCap;
};

__device__ GrainSynthesisEval grainSynthesisMakeEval(
  uint32_t layer,
  uint32_t channel,
  const KernelParams &params,
  bool fixedRadius
) {
  GrainSynthesisEval eval{};
  eval.scaledMeanRadius = fmaxf(
    params.grainSynthesisMeanRadiusUm *
      grainSynthesisChannelScale(channel, params) *
      grainSynthesisLayerScale(layer, params),
    1.0e-6f);
  const float ratio = fixedRadius ? 0.0f : fmaxf(params.grainSynthesisRadiusStdDevRatio, 0.0f);
  if (fixedRadius) {
    eval.maxRadius = eval.scaledMeanRadius;
  } else {
    const float logSigma = sqrtf(logf(1.0f + ratio * ratio));
    const float logMean = logf(fmaxf(eval.scaledMeanRadius, 1.0e-6f)) - 0.5f * logSigma * logSigma;
    eval.maxRadius = fmaxf(expf(logMean + logSigma * grainNormalQuantile(params.grainSynthesisMaxRadiusQuantile)), eval.scaledMeanRadius);
  }
  eval.maxRadiusSquared = eval.maxRadius * eval.maxRadius;
  eval.cellSize = fmaxf(eval.scaledMeanRadius * fmaxf(params.grainSynthesisCellSizeRatio, 0.05f), 1.0e-4f);
  eval.meanArea = 3.14159265359f * eval.scaledMeanRadius * eval.scaledMeanRadius * (1.0f + ratio * ratio);
  eval.cellArea = eval.cellSize * eval.cellSize;
  eval.densityToLambda = 2.302585093f / fmaxf(eval.meanArea, 1.0e-12f);
  eval.grainCap = static_cast<uint32_t>(clampf(static_cast<float>(params.grainSynthesisMaxGrainsPerCell), 1.0f, 128.0f));
  return eval;
}

__device__ float grainSynthesisRadius(
  const GrainSynthesisEval &eval,
  const KernelParams &params,
  bool fixedRadius,
  uint32_t seed
) {
  if (fixedRadius) {
    return eval.scaledMeanRadius;
  }
  const float ratio = fmaxf(params.grainSynthesisRadiusStdDevRatio, 0.0f);
  if (ratio <= 1.0e-6f) {
    return eval.scaledMeanRadius;
  }
  const float logSigma = sqrtf(logf(1.0f + ratio * ratio));
  const float logMean = logf(fmaxf(eval.scaledMeanRadius, 1.0e-6f)) - 0.5f * logSigma * logSigma;
  return fminf(expf(logMean + logSigma * grainGaussian(seed)), eval.maxRadius);
}

__device__ float grainSynthesisCellDistanceSquared(float pointX, float pointY, float cellX, float cellY, float cellSize) {
  const float closestX = clampf(pointX, cellX, cellX + cellSize);
  const float closestY = clampf(pointY, cellY, cellY + cellSize);
  const float dx = pointX - closestX;
  const float dy = pointY - closestY;
  return dx * dx + dy * dy;
}

__device__ uint32_t grainSynthesisAdaptiveSampleCount(float targetDensity, uint32_t requestedSamples) {
  if (requestedSamples <= 4u) {
    return requestedSamples;
  }
  const float coverage = clampf(1.0f - expf(-fmaxf(targetDensity, 0.0f) * 2.302585093f), 0.0f, 1.0f);
  const float varianceWeight = sqrtf(clampf((coverage * (1.0f - coverage)) * 4.0f, 0.0f, 1.0f));
  const float sampleScale = 0.25f + (1.0f - 0.25f) * varianceWeight;
  const uint32_t minimumSamples = min(requestedSamples, max(4u, requestedSamples / 16u));
  const uint32_t adaptiveSamples = static_cast<uint32_t>(ceilf(static_cast<float>(requestedSamples) * sampleScale));
  return min(max(adaptiveSamples, minimumSamples), requestedSamples);
}

__device__ float grainSynthesisDensityAtUm(
  const float *density,
  int width,
  int height,
  float filmUmX,
  float filmUmY,
  uint32_t layer,
  uint32_t channel,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const KernelCurveInfo &curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData
) {
  int px = 0;
  int py = 0;
  filmUmToOutputPixel(filmUmX, filmUmY, width, height, params, px, py);
  const uint32_t sx = mirroredIndex(px, static_cast<uint32_t>(width));
  const uint32_t sy = mirroredIndex(py, static_cast<uint32_t>(height));
  const size_t offset = (static_cast<size_t>(sy) * static_cast<size_t>(width) + sx) * 4u;
  const float value = density[offset + channel];
  if (params.grainSynthesisLayered == 0u) {
    return layer == 0u ? fmaxf(value, 0.0f) : 0.0f;
  }
  return interpDensityLayer(value, channel, layer, curveInfo, info, densityCurves, paperScanDensityData);
}

__device__ bool grainSynthesisIndicator(
  const float *density,
  int width,
  int height,
  float pointX,
  float pointY,
  uint32_t layer,
  uint32_t channel,
  const GrainSynthesisEval &eval,
  bool fixedRadius,
  const KernelParams &params,
  const KernelSpectralInfo &info,
  const KernelCurveInfo &curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData
) {
  const int cellMinX = static_cast<int>(floorf((pointX - eval.maxRadius) / eval.cellSize));
  const int cellMaxX = static_cast<int>(floorf((pointX + eval.maxRadius) / eval.cellSize));
  const int cellMinY = static_cast<int>(floorf((pointY - eval.maxRadius) / eval.cellSize));
  const int cellMaxY = static_cast<int>(floorf((pointY + eval.maxRadius) / eval.cellSize));
  for (int cy = cellMinY; cy <= cellMaxY; ++cy) {
    for (int cx = cellMinX; cx <= cellMaxX; ++cx) {
      const float cellOriginX = static_cast<float>(cx) * eval.cellSize;
      const float cellOriginY = static_cast<float>(cy) * eval.cellSize;
      if (grainSynthesisCellDistanceSquared(pointX, pointY, cellOriginX, cellOriginY, eval.cellSize) > eval.maxRadiusSquared) {
        continue;
      }
      const float targetDensity = grainSynthesisDensityAtUm(
        density,
        width,
        height,
        cellOriginX + 0.5f * eval.cellSize,
        cellOriginY + 0.5f * eval.cellSize,
        layer,
        channel,
        params,
        info,
        curveInfo,
        densityCurves,
        paperScanDensityData);
      if (targetDensity <= 0.0f) {
        continue;
      }
      const float expectedGrains = fmaxf(targetDensity, 0.0f) * eval.densityToLambda * eval.cellArea;
      if (expectedGrains <= 1.0e-7f) {
        continue;
      }
      const uint32_t baseSeed = grainCellSeed(cx, cy, channel, layer, params);
      const uint32_t grainCount = grainPoissonSample(expectedGrains, baseSeed, eval.grainCap);
      for (uint32_t grain = 0u; grain < grainCount; ++grain) {
        const uint32_t grainSeed = grainMixSeed(baseSeed, grain * 0x9e3779b9u);
        const float centerX = cellOriginX + grainRand01(grainSeed ^ 0x68bc21ebu) * eval.cellSize;
        const float centerY = cellOriginY + grainRand01(grainSeed ^ 0x02e5be93u) * eval.cellSize;
        const float radius = grainSynthesisRadius(eval, params, fixedRadius, grainSeed ^ 0x85ebca6bu);
        const float dx = pointX - centerX;
        const float dy = pointY - centerY;
        if (dx * dx + dy * dy <= radius * radius) {
          return true;
        }
      }
    }
  }
  return false;
}

__device__ float productionLayerParticleDensity(
  float3 filmDensityCmy,
  uint32_t layer,
  uint32_t channel,
  const KernelParams &params,
  const KernelCurveInfo &curveInfo,
  const KernelSpectralInfo &info,
  const float *densityCurves,
  const float *paperScanDensityData,
  float filmUmX,
  float filmUmY,
  uint32_t baseSeed
) {
  const float pixelArea = fmaxf(params.filmPixelSizeUm * params.filmPixelSizeUm, 1.0e-6f);
  const float densityMin = grainChannelDensityMin(channel, params);
  const float uniformity = grainChannelUniformity(channel, params);

  if (params.grainSublayersEnabled == 0u) {
    if (layer != 0u) {
      return 0.0f;
    }
    const uint32_t subLayerCount = static_cast<uint32_t>(clampf(static_cast<float>(params.grainSubLayerCount), 1.0f, 8.0f));
    const float densityMax = grainDensityCurveMax(channel, curveInfo, densityCurves) + densityMin;
    const float particleArea = fmaxf(params.grainParticleAreaUm2 * grainChannelParticleScale(channel, params), 1.0e-4f);
    const float particles = fmaxf(pixelArea / particleArea / fmaxf(static_cast<float>(subLayerCount), 1.0f), 1.0e-3f);
    const float sourceDensity = fmaxf(channelComponent(filmDensityCmy, channel) + densityMin, 0.0f);
    float accumulated = 0.0f;
    for (uint32_t subLayer = 0u; subLayer < subLayerCount; ++subLayer) {
      const uint32_t particleSeed = grainFilmCellSeed(
        filmUmX,
        filmUmY,
        sqrtf(particleArea),
        baseSeed ^ (channel * 0x85ebca6bu) ^ (subLayer * 0x9e3779b9u));
      accumulated += grainParticleDevelopedDensity(sourceDensity, densityMax, particles, uniformity, 1.0f, particleSeed);
    }
    return fmaxf(accumulated / fmaxf(static_cast<float>(subLayerCount), 1.0f) - densityMin, 0.0f);
  }

  const float densityMaxTotal = layerDensityMaxTotal(channel, curveInfo, info, paperScanDensityData);
  const float layerMax = fmaxf(densityCurveLayerMaximum(layer, channel, curveInfo, info, paperScanDensityData), 0.0f);
  const float layerFraction = layerMax / densityMaxTotal;
  const float layerDensityMin = layerFraction * densityMin;
  const float layerDensityMax = fmaxf(layerMax + layerDensityMin, 1.0e-6f);
  const float layerDensity =
    interpDensityLayer(channelComponent(filmDensityCmy, channel), channel, layer, curveInfo, info, densityCurves, paperScanDensityData) +
    layerDensityMin;
  const float particleArea = fmaxf(
    params.grainParticleAreaUm2 * grainChannelParticleScale(channel, params) * grainLayerParticleScale(layer, params),
    1.0e-4f);
  const float particles = fmaxf(pixelArea * layerFraction / particleArea, 1.0e-3f);
  const uint32_t particleSeed = grainFilmCellSeed(
    filmUmX,
    filmUmY,
    sqrtf(particleArea),
    baseSeed ^ (channel * 0x85ebca6bu) ^ (layer * 0xc2b2ae35u));
  return grainParticleDevelopedDensity(fmaxf(layerDensity, 0.0f), layerDensityMax, particles, uniformity, 1.0f, particleSeed);
}

__device__ float grainLayerBlurSigma(
  uint32_t layer,
  uint32_t channel,
  const KernelParams &params,
  const KernelCurveInfo &curveInfo,
  const KernelSpectralInfo &info,
  const float *paperScanDensityData
) {
  if (params.grainSublayersEnabled == 0u || params.grainBlurDyeCloudsUm <= 0.0f) {
    return 0.0f;
  }
  const float densityMin = grainChannelDensityMin(channel, params);
  const float densityMaxTotal = layerDensityMaxTotal(channel, curveInfo, info, paperScanDensityData);
  const float layerMax = fmaxf(densityCurveLayerMaximum(layer, channel, curveInfo, info, paperScanDensityData), 0.0f);
  const float layerFraction = layerMax / densityMaxTotal;
  const float layerDensityMax = fmaxf(layerMax + layerFraction * densityMin, 1.0e-6f);
  const float particleArea = fmaxf(
    params.grainParticleAreaUm2 * grainChannelParticleScale(channel, params) * grainLayerParticleScale(layer, params),
    1.0e-4f);
  const float pixelArea = fmaxf(params.filmPixelSizeUm * params.filmPixelSizeUm, 1.0e-6f);
  const float particles = fmaxf(pixelArea * layerFraction / particleArea, 1.0e-3f);
  const float odParticle = layerDensityMax / particles;
  return fmaxf(params.grainBlurDyeCloudsUm, 0.0f) * sqrtf(fmaxf(odParticle, 0.0f));
}

__device__ float grainLayerSampleMirrored(const float *layers, int width, int height, int x, int y, uint32_t component) {
  const uint32_t sx = mirroredIndex(x, static_cast<uint32_t>(width));
  const uint32_t sy = mirroredIndex(y, static_cast<uint32_t>(height));
  return layers[(static_cast<size_t>(sy) * static_cast<size_t>(width) + sx) * 9u + component];
}

__device__ float grainSpatialGaussianWeight(float offset, float sigma) {
  return expf(-0.5f * (offset * offset) / fmaxf(sigma * sigma, 1.0e-8f));
}

__device__ float grainLayerGaussian(
  const float *layers,
  int width,
  int height,
  int x,
  int y,
  uint32_t component,
  float sigma,
  bool horizontal
) {
  const int radius = min(static_cast<int>(ceilf(3.0f * sigma)), 64);
  float value = grainLayerSampleMirrored(layers, width, height, x, y, component);
  float weightSum = 1.0f;
  for (int offset = 1; offset <= radius; ++offset) {
    const float offsetf = static_cast<float>(offset);
    const float weight = grainSpatialGaussianWeight(offsetf, sigma);
    value += weight * (horizontal
      ? grainLayerSampleMirrored(layers, width, height, x + offset, y, component) +
        grainLayerSampleMirrored(layers, width, height, x - offset, y, component)
      : grainLayerSampleMirrored(layers, width, height, x, y + offset, component) +
        grainLayerSampleMirrored(layers, width, height, x, y - offset, component));
    weightSum += weight * 2.0f;
  }
  return value / fmaxf(weightSum, 1.0e-8f);
}

__device__ float microstructureSigma(const KernelParams &params) {
  return params.grainMicroStructureSigmaNm * 0.001f / fmaxf(params.filmPixelSizeUm, 1.0e-6f);
}

__device__ float microstructureBlurSigma(const KernelParams &params) {
  return fmaxf(params.grainMicroStructureScale, 0.0f) / fmaxf(params.filmPixelSizeUm, 1.0e-6f);
}

__device__ float grainFinalBlurUm(const KernelParams &params) {
  const float formatScale = powf(fmaxf(grainFilmFormatMm(params.filmFormat) / 35.0f, 1.0e-6f), 0.62f);
  return fmaxf(params.grainFinalBlurUm, 0.0f) * formatScale;
}

__device__ float4 grainFrameSampleMirrored(const float *frame, int width, int height, int x, int y) {
  const uint32_t sx = mirroredIndex(x, static_cast<uint32_t>(width));
  const uint32_t sy = mirroredIndex(y, static_cast<uint32_t>(height));
  const size_t offset = (static_cast<size_t>(sy) * static_cast<size_t>(width) + sx) * 4u;
  return make_float4(frame[offset], frame[offset + 1u], frame[offset + 2u], frame[offset + 3u]);
}

__device__ float4 grainFrameGaussian(
  const float *frame,
  int width,
  int height,
  int x,
  int y,
  float sigma,
  bool horizontal
) {
  const int radius = min(static_cast<int>(ceilf(3.0f * sigma)), 64);
  float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const float weight = grainSpatialGaussianWeight(static_cast<float>(offset), sigma);
    const float4 sample = horizontal
      ? grainFrameSampleMirrored(frame, width, height, x + offset, y)
      : grainFrameSampleMirrored(frame, width, height, x, y + offset);
    value.x += weight * sample.x;
    value.y += weight * sample.y;
    value.z += weight * sample.z;
    value.w += weight * sample.w;
    weightSum += weight;
  }
  const float inv = 1.0f / fmaxf(weightSum, 1.0e-8f);
  return mul4s(value, inv);
}

// grain paths start from film density; preview, production and synthesis resolve differently
// Grain models: production layers, synthesis, microstructure, preview.
__global__ void productionGrainLayersFromDensityKernel(
  const float *density,
  float *layers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData
) {
  const int index3 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index3 >= pixelCount * 9) {
    return;
  }
  const int pixel = index3 / 9;
  const uint32_t component = static_cast<uint32_t>(index3 - pixel * 9);
  const uint32_t layer = component / 3u;
  const uint32_t channel = component - layer * 3u;
  const int x = pixel % width;
  const int y = pixel / width;
  const size_t offset = static_cast<size_t>(pixel) * 4u;
  float filmUmX = 0.0f;
  float filmUmY = 0.0f;
  const KernelParams p = *params;
  outputPixelFilmUm(x, y, width, height, p, filmUmX, filmUmY);
  const uint32_t frameSeed = p.grainAnimate != 0u ? static_cast<uint32_t>(floorf(p.time * 24.0f + 0.5f)) : 0u;
  const uint32_t baseSeed = p.grainSeed ^ frameSeed;
  layers[static_cast<size_t>(index3)] = productionLayerParticleDensity(
    make_float3(density[offset], density[offset + 1u], density[offset + 2u]),
    layer,
    channel,
    p,
    *curveInfo,
    *spectralInfo,
    densityCurves,
    paperScanDensityData,
    filmUmX,
    filmUmY,
    baseSeed);
}

__global__ void grainSynthesisLayersFromDensityKernel(
  const float *density,
  float *layers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData
) {
  const int index3 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index3 >= pixelCount * 9) {
    return;
  }
  const int pixel = index3 / 9;
  const uint32_t component = static_cast<uint32_t>(index3 - pixel * 9);
  const uint32_t layer = component / 3u;
  const uint32_t channel = component - layer * 3u;
  const int x = pixel % width;
  const int y = pixel / width;
  const KernelParams p = *params;
  if (p.grainSynthesisLayered == 0u && layer > 0u) {
    layers[static_cast<size_t>(index3)] = 0.0f;
    return;
  }

  float centerX = 0.0f;
  float centerY = 0.0f;
  outputPixelFilmUm(x, y, width, height, p, centerX, centerY);
  const float targetCenterDensity = grainSynthesisDensityAtUm(
    density,
    width,
    height,
    centerX,
    centerY,
    layer,
    channel,
    p,
    *spectralInfo,
    *curveInfo,
    densityCurves,
    paperScanDensityData);
  const float sigmaUm = fmaxf(p.grainSynthesisObservationSigmaUm, 0.0f);
  if (targetCenterDensity <= 1.0e-7f && sigmaUm <= 1.0e-6f) {
    layers[static_cast<size_t>(index3)] = 0.0f;
    return;
  }

  const bool fixedRadius = p.grainSynthesisRadiusStdDevRatio <= 1.0e-6f;
  const uint32_t requestedSamples = static_cast<uint32_t>(clampf(static_cast<float>(p.grainSynthesisSamples), 1.0f, 1024.0f));
  const uint32_t sampleCount = grainSynthesisAdaptiveSampleCount(targetCenterDensity, requestedSamples);
  const GrainSynthesisEval eval = grainSynthesisMakeEval(layer, channel, p, fixedRadius);
  const uint32_t frameSeed = p.grainAnimate != 0u ? static_cast<uint32_t>(floorf(p.time * 24.0f + 0.5f)) : 0u;
  const uint32_t sampleSeedBase = grainHash(p.grainSeed ^ frameSeed ^ (channel * 0x85ebca6bu) ^ (layer * 0xc2b2ae35u));

  float covered = 0.0f;
  for (uint32_t sampleIndex = 0u; sampleIndex < sampleCount; ++sampleIndex) {
    const uint32_t sampleSeed = grainMixSeed(sampleSeedBase, sampleIndex);
    const float sampleX = sigmaUm > 0.0f ? grainGaussian(sampleSeed ^ 0x23d3c1f1u) * sigmaUm : 0.0f;
    const float sampleY = sigmaUm > 0.0f ? grainGaussian(sampleSeed ^ 0xa349b329u) * sigmaUm : 0.0f;
    covered += grainSynthesisIndicator(
      density,
      width,
      height,
      centerX + sampleX,
      centerY + sampleY,
      layer,
      channel,
      eval,
      fixedRadius,
      p,
      *spectralInfo,
      *curveInfo,
      densityCurves,
      paperScanDensityData) ? 1.0f : 0.0f;
  }

  const float epsilon = fmaxf(p.grainSynthesisCoverageEpsilon, 1.0e-8f);
  const float coverage = clampf(covered / fmaxf(static_cast<float>(sampleCount), 1.0f), 0.0f, 1.0f - epsilon);
  layers[static_cast<size_t>(index3)] = -logf(fmaxf(1.0f - coverage, epsilon)) / 2.302585093f;
}

__global__ void grainLayerBlurKernel(
  const float *sourceLayers,
  float *destinationLayers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *paperScanDensityData,
  bool horizontal
) {
  const int x = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
  const int y = static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.y) + static_cast<int>(threadIdx.y);
  const uint32_t component = static_cast<uint32_t>(blockIdx.z);
  if (component >= 9u) {
    return;
  }
  const uint32_t layer = component / 3u;
  const uint32_t channel = component - layer * 3u;

  __shared__ float blockSigma;
  if (threadIdx.x == 0u && threadIdx.y == 0u) {
    blockSigma = grainLayerBlurSigma(layer, channel, *params, *curveInfo, *spectralInfo, paperScanDensityData);
  }
  __syncthreads();

  if (x >= width || y >= height) {
    return;
  }
  const size_t index3 = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 9u + component;
  destinationLayers[index3] = blockSigma <= 1.0e-4f
    ? sourceLayers[index3]
    : grainLayerGaussian(sourceLayers, width, height, x, y, component, blockSigma, horizontal);
}

__global__ void grainMicrostructureSourceKernel(float *micro, int width, int height, const KernelParams *params) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const KernelParams p = *params;
  const float sigma = microstructureSigma(p);
  const size_t offset = static_cast<size_t>(index) * 4u;
  if (p.grainSublayersEnabled == 0u || sigma <= 0.05f || p.grainMicroStructureScale <= 0.0f) {
    micro[offset] = 1.0f;
    micro[offset + 1u] = 1.0f;
    micro[offset + 2u] = 1.0f;
    micro[offset + 3u] = 1.0f;
    return;
  }
  float filmUmX = 0.0f;
  float filmUmY = 0.0f;
  outputPixelFilmUm(x, y, width, height, p, filmUmX, filmUmY);
  const uint32_t frameSeed = p.grainAnimate != 0u ? static_cast<uint32_t>(floorf(p.time * 24.0f + 0.5f)) : 0u;
  const uint32_t baseSeed = grainFilmCellSeed(
    filmUmX,
    filmUmY,
    fmaxf(p.grainMicroStructureScale, 1.0e-4f),
    p.grainSeed ^ frameSeed ^ 0x23d3c1f1u ^ 0xa349b329u);
  const float logSigma = sqrtf(logf(1.0f + sigma * sigma));
  const float logMean = -0.5f * logSigma * logSigma;
  micro[offset] = expf(logMean + logSigma * grainGaussian(baseSeed ^ 0x165667b1u));
  micro[offset + 1u] = expf(logMean + logSigma * grainGaussian(baseSeed ^ 0x27d4eb2du));
  micro[offset + 2u] = expf(logMean + logSigma * grainGaussian(baseSeed ^ 0x85ebca6bu));
  micro[offset + 3u] = 1.0f;
}

__global__ void grainMicroBlurKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  bool horizontal
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float sigma = microstructureBlurSigma(*params);
  const size_t offset = static_cast<size_t>(index) * 4u;
  const bool bypass = params->grainSublayersEnabled == 0u || sigma <= 0.4f;
  const float4 v = bypass ? sampleFloat4Clamped(source, x, y, width, height) : grainFrameGaussian(source, width, height, x, y, sigma, horizontal);
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void grainResolveDensityKernel(
  const float *layers,
  const float *micro,
  const float *sourceDensity,
  float *destination,
  int pixelCount,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t pixelOffset = static_cast<size_t>(index) * 4u;
  const size_t layerOffset = static_cast<size_t>(index) * 9u;
  if (params->grainSublayersEnabled == 0u) {
    destination[pixelOffset] = fmaxf(layers[layerOffset], 0.0f);
    destination[pixelOffset + 1u] = fmaxf(layers[layerOffset + 1u], 0.0f);
    destination[pixelOffset + 2u] = fmaxf(layers[layerOffset + 2u], 0.0f);
    destination[pixelOffset + 3u] = sourceDensity[pixelOffset + 3u];
    return;
  }
  const float3 densityMin = make_float3(
    grainChannelDensityMin(0u, *params),
    grainChannelDensityMin(1u, *params),
    grainChannelDensityMin(2u, *params));
  float3 density = make_float3(0.0f, 0.0f, 0.0f);
  for (uint32_t layer = 0u; layer < 3u; ++layer) {
    density.x += layers[layerOffset + layer * 3u];
    density.y += layers[layerOffset + layer * 3u + 1u];
    density.z += layers[layerOffset + layer * 3u + 2u];
  }
  density.x *= micro[pixelOffset];
  density.y *= micro[pixelOffset + 1u];
  density.z *= micro[pixelOffset + 2u];
  density = max3s(sub3(density, densityMin), 0.0f);
  destination[pixelOffset] = density.x;
  destination[pixelOffset + 1u] = density.y;
  destination[pixelOffset + 2u] = density.z;
  destination[pixelOffset + 3u] = sourceDensity[pixelOffset + 3u];
}

__global__ void grainSynthesisResolveDensityKernel(
  const float *layers,
  const float *micro,
  const float *sourceDensity,
  float *destination,
  int pixelCount,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t pixelOffset = static_cast<size_t>(index) * 4u;
  const size_t layerOffset = static_cast<size_t>(index) * 9u;
  float3 density = make_float3(0.0f, 0.0f, 0.0f);
  if (params->grainSynthesisLayered == 0u) {
    density.x = fmaxf(layers[layerOffset], 0.0f);
    density.y = fmaxf(layers[layerOffset + 1u], 0.0f);
    density.z = fmaxf(layers[layerOffset + 2u], 0.0f);
  } else {
    for (uint32_t layer = 0u; layer < 3u; ++layer) {
      density.x += fmaxf(layers[layerOffset + layer * 3u], 0.0f);
      density.y += fmaxf(layers[layerOffset + layer * 3u + 1u], 0.0f);
      density.z += fmaxf(layers[layerOffset + layer * 3u + 2u], 0.0f);
    }
  }
  density.x *= fmaxf(micro[pixelOffset], 0.0f);
  density.y *= fmaxf(micro[pixelOffset + 1u], 0.0f);
  density.z *= fmaxf(micro[pixelOffset + 2u], 0.0f);
  const float amount = clampf(params->grainSynthesisAmount, 0.0f, 3.0f);
  destination[pixelOffset] = fmaxf(sourceDensity[pixelOffset] + (density.x - sourceDensity[pixelOffset]) * amount, 0.0f);
  destination[pixelOffset + 1u] = fmaxf(sourceDensity[pixelOffset + 1u] + (density.y - sourceDensity[pixelOffset + 1u]) * amount, 0.0f);
  destination[pixelOffset + 2u] = fmaxf(sourceDensity[pixelOffset + 2u] + (density.z - sourceDensity[pixelOffset + 2u]) * amount, 0.0f);
  destination[pixelOffset + 3u] = sourceDensity[pixelOffset + 3u];
}

__global__ void grainApplyControlsKernel(
  const float *baseDensity,
  const float *grainedDensity,
  float *destination,
  int pixelCount,
  const KernelParams *params
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float3 base = make_float3(baseDensity[offset], baseDensity[offset + 1u], baseDensity[offset + 2u]);
  const float3 grained = make_float3(grainedDensity[offset], grainedDensity[offset + 1u], grainedDensity[offset + 2u]);
  const float3 out = applyGrainControls(base, grained, *params);
  destination[offset] = out.x;
  destination[offset + 1u] = out.y;
  destination[offset + 2u] = out.z;
  destination[offset + 3u] = grainedDensity[offset + 3u];
}

__global__ void grainDensityBlurKernel(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  bool horizontal
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const float sigma = grainFinalBlurUm(*params) / fmaxf(params->filmPixelSizeUm, 1.0e-6f);
  const bool bypass = sigma <= 0.0f || (params->grainSublayersEnabled == 0u && sigma <= 0.4f);
  const float4 v = bypass ? sampleFloat4Clamped(source, x, y, width, height) : grainFrameGaussian(source, width, height, x, y, sigma, horizontal);
  const size_t offset = static_cast<size_t>(index) * 4u;
  destination[offset] = v.x;
  destination[offset + 1u] = v.y;
  destination[offset + 2u] = v.z;
  destination[offset + 3u] = v.w;
}

__global__ void previewGrainFromDensityKernel(
  const float *density,
  float *grainedDensity,
  int width,
  int height,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  const KernelCurveInfo cinfo = *curveInfo;
  const float3 densityCmy = make_float3(density[offset], density[offset + 1u], density[offset + 2u]);
  const int x = index % width;
  const int y = index / width;
  const float scale = fmaxf(p.enlargerScale, 1.0f);
  const float safeWidth = static_cast<float>(width > 0 ? width : 1);
  const float safeHeight = static_cast<float>(height > 0 ? height : 1);
  const float outputUvX = (static_cast<float>(x) + 0.5f) / safeWidth;
  const float outputUvY = (static_cast<float>(y) + 0.5f) / safeHeight;
  const float sourceUvX = 0.5f + (outputUvX - 0.5f) / scale + p.enlargerOffsetXPercent * (0.01f / scale);
  const float sourceUvY = 0.5f + (outputUvY - 0.5f) / scale + p.enlargerOffsetYPercent * (0.01f / scale);
  const float framePixelSizeUm = fmaxf(p.filmPixelSizeUm * scale, 1.0e-6f);
  const float filmUmX = sourceUvX * safeWidth * framePixelSizeUm;
  const float filmUmY = sourceUvY * safeHeight * framePixelSizeUm;
  const float pixelArea = fmaxf(p.filmPixelSizeUm * p.filmPixelSizeUm, 1.0e-6f);
  const float formatScale = powf(fmaxf(grainFilmFormatMm(p.filmFormat) / 35.0f, 1.0e-6f), 0.62f);
  const float finalBlurPx = fmaxf(p.grainFinalBlurUm, 0.0f) * formatScale / fmaxf(p.filmPixelSizeUm, 1.0e-6f);
  const float blurDamping = 1.0f / sqrtf(1.0f + 0.35f * finalBlurPx + 0.12f * fmaxf(p.grainBlurDyeCloudsUm, 0.0f));
  const uint32_t layerCount = p.grainSublayersEnabled != 0u
    ? 3u
    : static_cast<uint32_t>(clampf(static_cast<float>(p.grainSubLayerCount), 1.0f, 8.0f));
  const uint32_t frameSeed = p.grainAnimate != 0u ? static_cast<uint32_t>(floorf(p.time * 24.0f + 0.5f)) : 0u;
  const uint32_t baseSeed = p.grainSeed ^ frameSeed;

  float3 outDensity = densityCmy;
  for (uint32_t channel = 0u; channel < 3u; ++channel) {
    const float densityMin = grainChannelDensityMin(channel, p);
    const float densityMax = grainDensityCurveMax(channel, cinfo, densityCurves) + densityMin;
    const float particleArea = fmaxf(p.grainParticleAreaUm2 * grainChannelParticleScale(channel, p), 1.0e-4f);
    const float particles = fmaxf(pixelArea / particleArea / fmaxf(static_cast<float>(layerCount), 1.0f), 1.0e-3f);
    const float sourceDensity =
      fmaxf((channel == 0u ? densityCmy.x : (channel == 1u ? densityCmy.y : densityCmy.z)) + densityMin, 0.0f);
    float accumulated = 0.0f;
    for (uint32_t layer = 0u; layer < layerCount; ++layer) {
      const uint32_t particleSeed = grainFilmCellSeed(
        filmUmX,
        filmUmY,
        sqrtf(particleArea),
        baseSeed ^ (channel * 0x85ebca6bu) ^ (layer * 0xc2b2ae35u));
      accumulated += grainParticleDevelopedDensity(
        sourceDensity,
        densityMax,
        particles,
        grainChannelUniformity(channel, p),
        blurDamping,
        particleSeed);
    }
    accumulated /= fmaxf(static_cast<float>(layerCount), 1.0f);
    const float value = fmaxf(accumulated - densityMin, 0.0f);
    if (channel == 0u) {
      outDensity.x = value;
    } else if (channel == 1u) {
      outDensity.y = value;
    } else {
      outDensity.z = value;
    }
  }
  outDensity = applyGrainControls(densityCmy, outDensity, p);
  grainedDensity[offset] = outDensity.x;
  grainedDensity[offset + 1u] = outDensity.y;
  grainedDensity[offset + 2u] = outDensity.z;
  grainedDensity[offset + 3u] = density[offset + 3u];
}

// print exposure/density, followed later by scanner and display encoding
// Print path and final output encode.
__global__ void printRawFromFilmDensityKernel(
  const float *filmDensity,
  float *printRaw,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool logOutput
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  const KernelSpectralInfo info = *spectralInfo;
  const float3 density = make_float3(filmDensity[offset], filmDensity[offset + 1u], filmDensity[offset + 2u]);
  const float exposureFactor = printMidgrayExposureFactor(
    p,
    *colorInfo,
    info,
    *filmCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float3 preflash = printRawPreflash(
    p,
    info,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData);
  float3 raw = printRawFromFilmDensity(
    density,
    p,
    info,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData);
  raw = add3(mul3(mul3s(raw, exposureFactor), apdPrinterTimingExposureScale(p, info, academyPrinterDensityData)), preflash);
  raw = max3s(mul3s(raw, exp2f(p.printExposureEv)), 0.0f);
  if (logOutput) {
    constexpr float invLog10 = 1.0f / 2.302585092994046f;
    raw = make_float3(
      logf(raw.x + 1.0e-10f) * invLog10,
      logf(raw.y + 1.0e-10f) * invLog10,
      logf(raw.z + 1.0e-10f) * invLog10);
  }
  printRaw[offset] = raw.x;
  printRaw[offset + 1u] = raw.y;
  printRaw[offset + 2u] = raw.z;
  printRaw[offset + 3u] = filmDensity[offset + 3u];
}

__global__ void printRawFromNegativeLightKernel(
  const float *source,
  float *printRaw,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind,
  const float *inputToReferenceXyz,
  const float *paperHanatosResponse,
  const float *preflashPaperHanatosResponse,
  const float *academyPrinterDensityData
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  const KernelSpectralInfo info = *spectralInfo;
  const KernelColorInfo cinfo = *colorInfo;
  const float3 sourceRgb = make_float3(source[offset], source[offset + 1u], source[offset + 2u]);
  const float3 decoded = decodeInputRgb(sourceRgb, p, cinfo, colorDecodeLut, colorTransferKind);
  const float3 referenceXyz = mulColorMatrix(decoded, p.inputColorSpace, cinfo, inputToReferenceXyz);
  float3 raw = max3s(hanatosRaw(referenceXyz, p, info, paperHanatosResponse), 0.0f);

  constexpr int linearSrgbColorSpace = 17;
  const float3 midgrayXyz = mulColorMatrix(
    make_float3(0.184f, 0.184f, 0.184f), linearSrgbColorSpace, cinfo, inputToReferenceXyz);
  const float3 midgrayRaw = max3s(hanatosRaw(midgrayXyz, p, info, paperHanatosResponse), 1.0e-10f);
  const float midgrayGeomean =
    expf((logf(midgrayRaw.x) + logf(midgrayRaw.y) + logf(midgrayRaw.z)) / 3.0f);
  const float exposureFactor = 1.0f / fmaxf(midgrayGeomean, 1.0e-10f);

  const float3 whiteXyz = mulColorMatrix(
    make_float3(1.0f, 1.0f, 1.0f), linearSrgbColorSpace, cinfo, inputToReferenceXyz);
  const float3 preflashRaw = max3s(hanatosRaw(whiteXyz, p, info, preflashPaperHanatosResponse), 0.0f);
  raw = add3(
    mul3s(
      mul3(raw, apdPrinterTimingExposureScale(p, info, academyPrinterDensityData)),
      exposureFactor * exp2f(p.printExposureEv)),
    mul3s(preflashRaw, exp2f(p.printExposureEv)));
  raw = max3s(raw, 0.0f);
  printRaw[offset] = raw.x;
  printRaw[offset + 1u] = raw.y;
  printRaw[offset + 2u] = raw.z;
  printRaw[offset + 3u] = source[offset + 3u];
}

__global__ void printDensityFromPrintRawKernel(
  const float *printRaw,
  float *printDensity,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *paperLogExposure,
  const float *paperDensityCurves
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  const float3 logRaw = make_float3(
    logf(fmaxf(printRaw[offset], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(printRaw[offset + 1u], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(printRaw[offset + 2u], 0.0f) + 1.0e-10f) * invLog10);
  const float3 density = developPrintDensity(logRaw, *params, *spectralInfo, *paperCurveInfo, paperLogExposure, paperDensityCurves);
  printDensity[offset] = density.x;
  printDensity[offset + 1u] = density.y;
  printDensity[offset + 2u] = density.z;
  printDensity[offset + 3u] = printRaw[offset + 3u];
}

__global__ void makeFrameConstantsKernel(
  KernelFrameConstants *frameConstants,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz
) {
  if (threadIdx.x != 0u || blockIdx.x != 0u) {
    return;
  }
  const KernelParams p = *params;
  const KernelSpectralInfo info = *spectralInfo;
  const KernelColorInfo cinfo = *colorInfo;
  const float printExposureFactor = printMidgrayExposureFactor(
    p,
    cinfo,
    info,
    *filmCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float3 preflashRaw = printRawPreflash(
    p,
    info,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData);
  const float printReferenceBlackY = printReferenceY(
    true,
    p,
    cinfo,
    info,
    *filmCurveInfo,
    *paperCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    paperLogExposure,
    paperDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float printReferenceWhiteY = printReferenceY(
    false,
    p,
    cinfo,
    info,
    *filmCurveInfo,
    *paperCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    paperLogExposure,
    paperDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float3 filmBlack = densityCurveMaxCmy(false, info);
  const float3 filmWhite = make_float3(0.0f, 0.0f, 0.0f);
  CudaScanResult filmDmaxScan{make_float3(0.0f, 0.0f, 0.0f), 0.0f};
  CudaScanResult filmDminScan{make_float3(0.0f, 0.0f, 0.0f), 0.0f};
  if (p.scanNegativeInvert != 0u) {
    filmDmaxScan = scanDensityToOutputRgbLinearY(
      filmBlack, 0.0f, p, cinfo, info, filmChannelDensity, filmBaseDensity,
      paperScanDensityData, scanIlluminantsAndCmfs, scanToOutputRgbData, false);
    filmDminScan = scanDensityToOutputRgbLinearY(
      filmWhite, 0.0f, p, cinfo, info, filmChannelDensity, filmBaseDensity,
      paperScanDensityData, scanIlluminantsAndCmfs, scanToOutputRgbData, false);
  }
  const float3 printGlareRgb = scanIlluminantToOutputRgb(
    p,
    cinfo,
    info,
    scanIlluminantsAndCmfs,
    scanToOutputRgbData);

  frameConstants->print[0] = printExposureFactor;
  frameConstants->print[1] = printReferenceBlackY;
  frameConstants->print[2] = printReferenceWhiteY;
  frameConstants->print[3] = 0.0f;
  frameConstants->film[0] = filmDmaxScan.y;
  frameConstants->film[1] = filmDminScan.y;
  frameConstants->film[2] = 0.0f;
  frameConstants->film[3] = 0.0f;
  frameConstants->glare[0] = printGlareRgb.x;
  frameConstants->glare[1] = printGlareRgb.y;
  frameConstants->glare[2] = printGlareRgb.z;
  frameConstants->glare[3] = 0.0f;
  frameConstants->preflash[0] = preflashRaw.x;
  frameConstants->preflash[1] = preflashRaw.y;
  frameConstants->preflash[2] = preflashRaw.z;
  frameConstants->preflash[3] = 0.0f;
  frameConstants->filmDmaxScan[0] = filmDmaxScan.rgb.x;
  frameConstants->filmDmaxScan[1] = filmDmaxScan.rgb.y;
  frameConstants->filmDmaxScan[2] = filmDmaxScan.rgb.z;
  frameConstants->filmDmaxScan[3] = filmDmaxScan.y;
  frameConstants->filmDminScan[0] = filmDminScan.rgb.x;
  frameConstants->filmDminScan[1] = filmDminScan.rgb.y;
  frameConstants->filmDminScan[2] = filmDminScan.rgb.z;
  frameConstants->filmDminScan[3] = filmDminScan.y;
}

__global__ void finalFromFilmDensityKernel(
  const float *filmDensity,
  float *destination,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind,
  const KernelFrameConstants *frameConstants,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool finalizeOutput
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  const KernelSpectralInfo info = *spectralInfo;
  const KernelColorInfo cinfo = *colorInfo;
  const float3 filmDensityCmy = make_float3(filmDensity[offset], filmDensity[offset + 1u], filmDensity[offset + 2u]);
  float3 rgb = filmDensityCmy;

  if (p.process == 0) {
    const KernelFrameConstants fc = *frameConstants;
    const float exposureFactor = fc.print[0];
    const float3 preflash = make_float3(fc.preflash[0], fc.preflash[1], fc.preflash[2]);
    float3 raw = printRawFromFilmDensity(
      filmDensityCmy,
      p,
      info,
      filmChannelDensity,
      filmBaseDensity,
      paperLogSensitivity,
      thKg3Illuminant,
      customEnlargerFilters,
      neutralPrintFilters,
      academyPrinterDensityData);
    raw = add3(mul3(mul3s(raw, exposureFactor), apdPrinterTimingExposureScale(p, info, academyPrinterDensityData)), preflash);
    raw = max3s(mul3s(raw, exp2f(p.printExposureEv)), 0.0f);
    constexpr float invLog10 = 1.0f / 2.302585092994046f;
    const float3 printLogRaw = make_float3(
      logf(raw.x + 1.0e-10f) * invLog10,
      logf(raw.y + 1.0e-10f) * invLog10,
      logf(raw.z + 1.0e-10f) * invLog10);
    const float3 printDensity = developPrintDensity(printLogRaw, p, info, *paperCurveInfo, paperLogExposure, paperDensityCurves);
    const float retained = retainedSilverDensity(printDensity, p.printBleachBypassAmount, true, info);
    const float3 bypassed = bleachBypassDyeDensity(printDensity, p.printBleachBypassAmount, true, info);
    const CudaScanResult scan = scanDensityToOutputRgbLinearY(
      bypassed,
      retained,
      p,
      cinfo,
      info,
      filmChannelDensity,
      filmBaseDensity,
      paperScanDensityData,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData,
      true);
    rgb = applyScannerBlackWhiteCorrection(scan.rgb, scan.y, fc.print[1], fc.print[2], p);
  } else if (p.process == 1) {
    const float retained = retainedSilverDensity(filmDensityCmy, p.negativeBleachBypassAmount, false, info);
    const float3 bypassed = negativeBleachBypassDyeDensity(filmDensityCmy, p.negativeBleachBypassAmount, p, info);
    const CudaScanResult scan = scanDensityToOutputRgbLinearY(
      bypassed,
      retained,
      p,
      cinfo,
      info,
      filmChannelDensity,
      filmBaseDensity,
      paperScanDensityData,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData,
      false);
    if (p.scanNegativeInvert != 0u) {
      const KernelFrameConstants fc = *frameConstants;
      const float3 dmax = make_float3(fc.filmDmaxScan[0], fc.filmDmaxScan[1], fc.filmDmaxScan[2]);
      const float3 dmin = make_float3(fc.filmDminScan[0], fc.filmDminScan[1], fc.filmDminScan[2]);
      const float3 range = make_float3(
        copysignf(fmaxf(fabsf(dmin.x - dmax.x), 1.0e-6f), dmin.x - dmax.x),
        copysignf(fmaxf(fabsf(dmin.y - dmax.y), 1.0e-6f), dmin.y - dmax.y),
        copysignf(fmaxf(fabsf(dmin.z - dmax.z), 1.0e-6f), dmin.z - dmax.z));
      const float yRange = fmaxf(fc.filmDminScan[3] - fc.filmDmaxScan[3], 1.0e-10f);
      const bool positive = info.filmPositive != 0u;
      const float3 normalized = positive
        ? div3(sub3(scan.rgb, dmax), range)
        : div3(sub3(dmin, scan.rgb), range);
      const float normalizedY = positive
        ? (scan.y - fc.filmDmaxScan[3]) / yRange
        : (fc.filmDminScan[3] - scan.y) / yRange;
      rgb = p.outputRole == 2
        ? normalized
        : applyScannerBlackWhiteCorrection(normalized, normalizedY, 0.0f, 1.0f, p);
    } else {
      rgb = scan.rgb;
    }
  }

  if (finalizeOutput) {
    rgb = finalizeOutputRgb(rgb, p, cinfo, colorEncodeLut, colorTransferKind);
  }
  destination[offset] = rgb.x;
  destination[offset + 1u] = rgb.y;
  destination[offset + 2u] = rgb.z;
  destination[offset + 3u] = filmDensity[offset + 3u];
}

__global__ void finalFromPrintRawKernel(
  const float *printRaw,
  float *destination,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool finalizeOutput
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  const KernelSpectralInfo info = *spectralInfo;
  const KernelColorInfo cinfo = *colorInfo;
  constexpr float invLog10 = 1.0f / 2.302585092994046f;
  const float3 printLogRaw = make_float3(
    logf(fmaxf(printRaw[offset], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(printRaw[offset + 1u], 0.0f) + 1.0e-10f) * invLog10,
    logf(fmaxf(printRaw[offset + 2u], 0.0f) + 1.0e-10f) * invLog10);
  const float3 printDensity = developPrintDensity(printLogRaw, p, info, *paperCurveInfo, paperLogExposure, paperDensityCurves);
  const float retained = retainedSilverDensity(printDensity, p.printBleachBypassAmount, true, info);
  const float3 bypassed = bleachBypassDyeDensity(printDensity, p.printBleachBypassAmount, true, info);
  const CudaScanResult scan = scanDensityToOutputRgbLinearY(
    bypassed,
    retained,
    p,
    cinfo,
    info,
    filmChannelDensity,
    filmBaseDensity,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    scanToOutputRgbData,
    true);
  const float printReferenceBlackY = printReferenceY(
    true,
    p,
    cinfo,
    info,
    *filmCurveInfo,
    *paperCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    paperLogExposure,
    paperDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  const float printReferenceWhiteY = printReferenceY(
    false,
    p,
    cinfo,
    info,
    *filmCurveInfo,
    *paperCurveInfo,
    filmLogExposure,
    filmDensityCurves,
    paperLogExposure,
    paperDensityCurves,
    filmChannelDensity,
    filmBaseDensity,
    paperLogSensitivity,
    thKg3Illuminant,
    customEnlargerFilters,
    neutralPrintFilters,
    academyPrinterDensityData,
    paperScanDensityData,
    scanIlluminantsAndCmfs,
    hanatosRawResponse,
    mallettBasisIlluminant,
    inputToReferenceXyz);
  float3 rgb = p.outputRole == 2
    ? scan.rgb
    : applyScannerBlackWhiteCorrection(scan.rgb, scan.y, printReferenceBlackY, printReferenceWhiteY, p);
  if (finalizeOutput) {
    rgb = finalizeOutputRgb(rgb, p, cinfo, colorEncodeLut, colorTransferKind);
  }
  destination[offset] = rgb.x;
  destination[offset + 1u] = rgb.y;
  destination[offset + 2u] = rgb.z;
  destination[offset + 3u] = printRaw[offset + 3u];
}

__device__ float gaussianWeight(float offset, float sigma) {
  return expf(-0.5f * (offset * offset) / fmaxf(sigma * sigma, 1.0e-8f));
}

// Scanner and print-glare post pass.
__global__ void gaussianBlurXKernel(const float *source, float *destination, int width, int height, float sigma) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  if (sigma <= 1.0e-4f) {
    destination[outOffset] = source[outOffset];
    destination[outOffset + 1u] = source[outOffset + 1u];
    destination[outOffset + 2u] = source[outOffset + 2u];
    destination[outOffset + 3u] = source[outOffset + 3u];
    return;
  }
  const int radius = min(static_cast<int>(ceilf(3.0f * sigma)), 256);
  float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const float weight = gaussianWeight(static_cast<float>(offset), sigma);
    const float4 sample = sampleFloat4Clamped(source, x + offset, y, width, height);
    value.x += weight * sample.x;
    value.y += weight * sample.y;
    value.z += weight * sample.z;
    value.w += weight * sample.w;
    weightSum += weight;
  }
  const float invWeight = 1.0f / fmaxf(weightSum, 1.0e-8f);
  destination[outOffset] = value.x * invWeight;
  destination[outOffset + 1u] = value.y * invWeight;
  destination[outOffset + 2u] = value.z * invWeight;
  destination[outOffset + 3u] = value.w * invWeight;
}

__global__ void gaussianBlurYKernel(const float *source, float *destination, int width, int height, float sigma) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const size_t outOffset = static_cast<size_t>(index) * 4u;
  if (sigma <= 1.0e-4f) {
    destination[outOffset] = source[outOffset];
    destination[outOffset + 1u] = source[outOffset + 1u];
    destination[outOffset + 2u] = source[outOffset + 2u];
    destination[outOffset + 3u] = source[outOffset + 3u];
    return;
  }
  const int radius = min(static_cast<int>(ceilf(3.0f * sigma)), 256);
  float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  float weightSum = 0.0f;
  for (int offset = -radius; offset <= radius; ++offset) {
    const float weight = gaussianWeight(static_cast<float>(offset), sigma);
    const float4 sample = sampleFloat4Clamped(source, x, y + offset, width, height);
    value.x += weight * sample.x;
    value.y += weight * sample.y;
    value.z += weight * sample.z;
    value.w += weight * sample.w;
    weightSum += weight;
  }
  const float invWeight = 1.0f / fmaxf(weightSum, 1.0e-8f);
  destination[outOffset] = value.x * invWeight;
  destination[outOffset + 1u] = value.y * invWeight;
  destination[outOffset + 2u] = value.z * invWeight;
  destination[outOffset + 3u] = value.w * invWeight;
}

__device__ float hash01(int x, int y, uint32_t seed) {
  uint32_t value =
    static_cast<uint32_t>(x) * 1664525u +
    static_cast<uint32_t>(y) * 1013904223u +
    seed * 747796405u +
    2891336453u;
  value ^= value >> 16u;
  value *= 2246822519u;
  value ^= value >> 13u;
  value *= 3266489917u;
  value ^= value >> 16u;
  return (static_cast<float>(value & 0x00ffffffu) + 0.5f) / 16777216.0f;
}

__device__ float lognormalFromMeanStd(float mean, float stddev, int x, int y, uint32_t seed) {
  mean = fmaxf(mean, 0.0f);
  stddev = fmaxf(stddev, 0.0f);
  if (mean <= 0.0f) {
    return 0.0f;
  }
  if (stddev <= 1.0e-10f) {
    return mean;
  }
  const float varianceRatio = (stddev * stddev) / fmaxf(mean * mean, 1.0e-20f);
  const float sigma2 = logf(1.0f + varianceRatio);
  const float sigma = sqrtf(fmaxf(sigma2, 0.0f));
  const float mu = logf(mean) - 0.5f * sigma2;
  const float u1 = fmaxf(hash01(x, y, seed), 1.0e-7f);
  const float u2 = hash01(y, x, seed ^ 0x9e3779b9u);
  constexpr float twoPi = 6.28318530718f;
  const float normal = sqrtf(-2.0f * logf(u1)) * cosf(twoPi * u2);
  return expf(mu + sigma * normal);
}

__global__ void printGlareGenerateKernel(float *glareAmount, int width, int height, const KernelParams *params) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int pixelCount = width * height;
  if (index >= pixelCount) {
    return;
  }
  const int x = index % width;
  const int y = index / width;
  const KernelParams p = *params;
  const float mean = fmaxf(p.glarePercent, 0.0f);
  const float stddev = fmaxf(p.glareRoughness, 0.0f) * mean;
  const float amount = lognormalFromMeanStd(mean, stddev, x, y, p.grainSeed);
  const size_t offset = static_cast<size_t>(index) * 4u;
  glareAmount[offset] = amount;
  glareAmount[offset + 1u] = amount;
  glareAmount[offset + 2u] = amount;
  glareAmount[offset + 3u] = 1.0f;
}

__global__ void printGlareApplyKernel(
  const float *linearSource,
  const float *glareAmount,
  float *destination,
  int pixelCount,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const float3 glareRgb =
    scanIlluminantToOutputRgb(*params, *colorInfo, *spectralInfo, scanIlluminantsAndCmfs, scanToOutputRgbData);
  const float amount = fmaxf(glareAmount[offset], 0.0f) * 0.01f;
  destination[offset] = linearSource[offset] + amount * glareRgb.x;
  destination[offset + 1u] = linearSource[offset + 1u] + amount * glareRgb.y;
  destination[offset + 2u] = linearSource[offset + 2u] + amount * glareRgb.z;
  destination[offset + 3u] = linearSource[offset + 3u];
}

__global__ void scannerFinalizeKernel(
  const float *linearSource,
  const float *unsharpBlur,
  float *destination,
  int pixelCount,
  const KernelParams *params,
  const KernelColorInfo *colorInfo,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind
) {
  const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= pixelCount) {
    return;
  }
  const size_t offset = static_cast<size_t>(index) * 4u;
  const KernelParams p = *params;
  float3 rgb = make_float3(linearSource[offset], linearSource[offset + 1u], linearSource[offset + 2u]);
  if (unsharpBlur && p.scannerEnabled != 0u && p.scannerUnsharpSigmaPx > 0.0f && p.scannerUnsharpAmount > 0.0f) {
    const float3 blurred = make_float3(unsharpBlur[offset], unsharpBlur[offset + 1u], unsharpBlur[offset + 2u]);
    rgb = add3(rgb, mul3s(sub3(rgb, blurred), p.scannerUnsharpAmount));
  }
  rgb = finalizeOutputRgb(rgb, p, *colorInfo, colorEncodeLut, colorTransferKind);
  destination[offset] = rgb.x;
  destination[offset + 1u] = rgb.y;
  destination[offset + 2u] = rgb.z;
  destination[offset + 3u] = linearSource[offset + 3u];
}

template <typename KernelLauncher>
bool timedLaunch(KernelLauncher launch, float *kernelMs, char *error, size_t errorSize) {
  static const bool passTimingEnabled = []() {
    const char *value = std::getenv("SPEKTRAFILM_CUDA_PASS_TIMING");
    return value && *value && std::strcmp(value, "0") != 0 &&
      std::strcmp(value, "off") != 0 && std::strcmp(value, "false") != 0;
  }();
  if (!passTimingEnabled) {
    if (kernelMs) {
      *kernelMs = 0.0f;
    }
    launch();
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
      setError(error, errorSize, "CUDA kernel launch failed", status);
      return false;
    }
    return true;
  }
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t status = cudaEventCreate(&start);
  if (status != cudaSuccess) {
    setError(error, errorSize, "cudaEventCreate(start) failed", status);
    return false;
  }
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    setError(error, errorSize, "cudaEventCreate(stop) failed", status);
    return false;
  }
  cudaEventRecord(start, 0);
  launch();
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaEventRecord(stop, 0);
  }
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(stop);
  }
  if (status != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    setError(error, errorSize, "CUDA kernel failed", status);
    return false;
  }
  float elapsed = 0.0f;
  cudaEventElapsedTime(&elapsed, start, stop);
  if (kernelMs) {
    *kernelMs += elapsed;
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return true;
}

} // namespace

// public launch wrappers below: validate arguments, launch, report CUDA errors/timing
bool spektraCudaInitialize(int *deviceIndex, CudaDeviceInfo *deviceInfo, char *error, size_t errorSize) {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    setError(error, errorSize, "cudaGetDeviceCount failed", status);
    return false;
  }
  if (count <= 0) {
    setError(error, errorSize, "No CUDA devices found");
    return false;
  }

  int selected = 0;
  status = cudaGetDevice(&selected);
  if (status != cudaSuccess || selected < 0 || selected >= count) {
    selected = 0;
    status = cudaSetDevice(selected);
  }
  if (status != cudaSuccess) {
    setError(error, errorSize, "cudaGetDevice/cudaSetDevice failed", status);
    return false;
  }
  cudaDeviceProp selectedProps{};
  status = cudaGetDeviceProperties(&selectedProps, selected);
  if (status != cudaSuccess) {
    setError(error, errorSize, "cudaGetDeviceProperties failed", status);
    return false;
  }

  if (deviceIndex) {
    *deviceIndex = selected;
  }
  if (deviceInfo) {
    std::memset(deviceInfo, 0, sizeof(*deviceInfo));
    std::snprintf(deviceInfo->name, sizeof(deviceInfo->name), "%s", selectedProps.name);
    deviceInfo->major = selectedProps.major;
    deviceInfo->minor = selectedProps.minor;
  }
  return true;
}

bool spektraCudaSmokeCopy(const float *source, float *destination, size_t floatCount, float *kernelMs, char *error, size_t errorSize) {
  if (!source || !destination || floatCount == 0u) {
    setError(error, errorSize, "Invalid CUDA smoke-copy buffers");
    return false;
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  cudaError_t status = cudaEventCreate(&start);
  if (status != cudaSuccess) {
    setError(error, errorSize, "cudaEventCreate(start) failed", status);
    return false;
  }
  status = cudaEventCreate(&stop);
  if (status != cudaSuccess) {
    cudaEventDestroy(start);
    setError(error, errorSize, "cudaEventCreate(stop) failed", status);
    return false;
  }

  constexpr int blockSize = 256;
  const unsigned int blockCount = static_cast<unsigned int>((floatCount + blockSize - 1u) / blockSize);
  cudaEventRecord(start, 0);
  copyFloatKernel<<<blockCount, blockSize>>>(source, destination, floatCount);
  status = cudaGetLastError();
  if (status == cudaSuccess) {
    status = cudaEventRecord(stop, 0);
  }
  if (status == cudaSuccess) {
    status = cudaEventSynchronize(stop);
  }
  if (status != cudaSuccess) {
    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    setError(error, errorSize, "CUDA smoke-copy kernel failed", status);
    return false;
  }

  if (kernelMs) {
    *kernelMs = 0.0f;
    cudaEventElapsedTime(kernelMs, start, stop);
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
  return true;
}

bool spektraCudaCopyFrame(
  const float *source,
  float *destination,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA frame dimensions");
    return false;
  }
  const size_t floatCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  return spektraCudaSmokeCopy(source, destination, floatCount, kernelMs, error, errorSize);
}

bool spektraCudaPackDeviceImageToFloat(
  const void *source,
  int sourceX1,
  int sourceY1,
  int sourceRowBytes,
  int sourceBytesPerComponent,
  int windowX1,
  int windowY1,
  int width,
  int height,
  float *destination,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || width <= 0 || height <= 0 ||
      (sourceBytesPerComponent != 2 && sourceBytesPerComponent != 4)) {
    setError(error, errorSize, "Invalid CUDA device-image pack arguments");
    return false;
  }
  const int pixelCount = width * height;
  const int blockSize = 256;
  const int blockCount = (pixelCount + blockSize - 1) / blockSize;
  return timedLaunch([&]() {
    packDeviceImageToFloatKernel<<<blockCount, blockSize>>>(
      source,
      sourceX1,
      sourceY1,
      sourceRowBytes,
      sourceBytesPerComponent,
      windowX1,
      windowY1,
      width,
      height,
      destination);
  }, kernelMs, error, errorSize);
}

bool spektraCudaUnpackFloatToDeviceImage(
  const float *source,
  void *destination,
  int destinationX1,
  int destinationY1,
  int destinationRowBytes,
  int destinationBytesPerComponent,
  int windowX1,
  int windowY1,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || width <= 0 || height <= 0 ||
      (destinationBytesPerComponent != 2 && destinationBytesPerComponent != 4)) {
    setError(error, errorSize, "Invalid CUDA device-image unpack arguments");
    return false;
  }
  const int pixelCount = width * height;
  const int blockSize = 256;
  const int blockCount = (pixelCount + blockSize - 1) / blockSize;
  return timedLaunch([&]() {
    unpackFloatToDeviceImageKernel<<<blockCount, blockSize>>>(
      source,
      destination,
      destinationX1,
      destinationY1,
      destinationRowBytes,
      destinationBytesPerComponent,
      windowX1,
      windowY1,
      width,
      height);
  }, kernelMs, error, errorSize);
}

bool spektraCudaAutoExposurePreview(
  const float *source,
  float *luminance,
  int width,
  int height,
  int previewWidth,
  int previewHeight,
  const KernelParams *params,
  const KernelColorInfo *colorInfo,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind,
  float meterR,
  float meterG,
  float meterB,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !luminance || !params || !colorInfo || !colorDecodeLut || !colorTransferKind ||
      width <= 0 || height <= 0 || previewWidth <= 0 || previewHeight <= 0) {
    setError(error, errorSize, "Invalid CUDA auto-exposure preview arguments");
    return false;
  }
  const int pixelCount = previewWidth * previewHeight;
  const int blockSize = 256;
  const int blockCount = (pixelCount + blockSize - 1) / blockSize;
  return timedLaunch([&]() {
    autoExposurePreviewKernel<<<blockCount, blockSize>>>(
      source,
      luminance,
      width,
      height,
      previewWidth,
      previewHeight,
      params,
      colorInfo,
      colorDecodeLut,
      colorTransferKind,
      meterR,
      meterG,
      meterB);
  }, kernelMs, error, errorSize);
}

bool spektraCudaMakeFrameConstants(
  KernelFrameConstants *frameConstants,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!frameConstants || !params || !spectralInfo || !colorInfo || !filmCurveInfo || !paperCurveInfo ||
      !filmLogExposure || !filmDensityCurves || !paperLogExposure || !paperDensityCurves ||
      !filmChannelDensity || !filmBaseDensity || !paperLogSensitivity || !thKg3Illuminant ||
      !customEnlargerFilters || !neutralPrintFilters || !academyPrinterDensityData ||
      !paperScanDensityData || !scanIlluminantsAndCmfs || !scanToOutputRgbData ||
      !hanatosRawResponse || !mallettBasisIlluminant || !inputToReferenceXyz) {
    setError(error, errorSize, "Invalid CUDA frame constants arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  return timedLaunch([&]() {
    makeFrameConstantsKernel<<<1u, 1u>>>(
      frameConstants,
      params,
      spectralInfo,
      colorInfo,
      filmCurveInfo,
      paperCurveInfo,
      filmLogExposure,
      filmDensityCurves,
      paperLogExposure,
      paperDensityCurves,
      filmChannelDensity,
      filmBaseDensity,
      paperLogSensitivity,
      thKg3Illuminant,
      customEnlargerFilters,
      neutralPrintFilters,
      academyPrinterDensityData,
      paperScanDensityData,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData,
      hanatosRawResponse,
      mallettBasisIlluminant,
      inputToReferenceXyz);
  }, kernelMs, error, errorSize);
}

bool spektraCudaRawExposure(
  const float *source,
  float *raw,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  const float *inputToSrgb,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !raw || !params || !spectralInfo || !colorInfo || !hanatosRawResponse ||
      !mallettBasisIlluminant || !inputToReferenceXyz || !inputToSrgb || !colorDecodeLut || !colorTransferKind ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA raw exposure arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    rawExposureKernel<<<blockCount, blockSize>>>(
      source,
      raw,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      hanatosRawResponse,
      mallettBasisIlluminant,
      inputToReferenceXyz,
      inputToSrgb,
      colorDecodeLut,
      colorTransferKind
    );
  }, kernelMs, error, errorSize);
}

bool spektraCudaEnlargerResample(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA enlarger-resample arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    enlargerResampleKernel<<<blockCount, blockSize>>>(source, destination, width, height, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaRawToLogRaw(
  const float *raw,
  float *logRaw,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!raw || !logRaw || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA raw-to-log arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    rawToLogRawKernel<<<blockCount, blockSize>>>(raw, logRaw, pixelCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDevelopFromRaw(
  const float *raw,
  float *density,
  int width,
  int height,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *logExposure,
  const float *densityCurves,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!raw || !density || !params || !curveInfo || !logExposure || !densityCurves || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA develop-from-raw arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    developFromRawKernel<<<blockCount, blockSize>>>(raw, density, pixelCount, params, curveInfo, logExposure, densityCurves);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirCorrectionFromDensity(
  const float *density,
  float *correction,
  int width,
  int height,
  const KernelSpectralInfo *spectralInfo,
  const KernelDirInfo *dirInfo,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!density || !correction || !spectralInfo || !dirInfo || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR correction arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirCorrectionFromDensityKernel<<<blockCount, blockSize>>>(density, correction, pixelCount, spectralInfo, dirInfo);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !blurInfo || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirBlurXKernel<<<blockCount, blockSize>>>(source, destination, width, height, blurInfo);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !blurInfo || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirBlurYKernel<<<blockCount, blockSize>>>(source, destination, width, height, blurInfo);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirTailBlurX(
  const float *source,
  float *tailPlanes,
  int width,
  int height,
  const KernelGaussianBlurInfo *tailBlurInfos,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !tailPlanes || !tailBlurInfos || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR tail blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirTailBlurXKernel<<<blockCount, blockSize>>>(source, tailPlanes, width, height, tailBlurInfos);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirTailBlurYAccumulate(
  const float *tailPlanes,
  float *correctionInOut,
  int width,
  int height,
  const KernelParams *params,
  const KernelGaussianBlurInfo *tailBlurInfos,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!tailPlanes || !correctionInOut || !params || !tailBlurInfos || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR tail blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirTailBlurYAccumulateKernel<<<blockCount, blockSize>>>(
      tailPlanes,
      correctionInOut,
      width,
      height,
      params,
      tailBlurInfos);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDirRedevelop(
  const float *logRaw,
  const float *correction,
  float *density,
  int width,
  int height,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *logExposure,
  const float *correctedDensityCurves,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!logRaw || !correction || !density || !params || !curveInfo || !logExposure || !correctedDensityCurves ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA DIR redevelop arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    dirRedevelopKernel<<<blockCount, blockSize>>>(
      logRaw,
      correction,
      density,
      pixelCount,
      params,
      curveInfo,
      logExposure,
      correctedDensityCurves);
  }, kernelMs, error, errorSize);
}

bool spektraCudaClearFrame(
  float *destination,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!destination || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA clear-frame arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    clearFrameKernel<<<blockCount, blockSize>>>(destination, pixelCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentIndex,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !components || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA diffusion blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionBlurXKernel<<<blockCount, blockSize>>>(source, destination, width, height, components, componentIndex);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionBlurYAccumulate(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentIndex,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !components || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA diffusion blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionBlurYAccumulateKernel<<<blockCount, blockSize>>>(source, accum, width, height, components, componentIndex);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionGroupBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !components || width <= 0 || height <= 0 || componentCount == 0u || componentCount > 4u) {
    setError(error, errorSize, "Invalid CUDA diffusion group blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionGroupBlurXKernel<<<blockCount, blockSize>>>(
      source,
      destination,
      width,
      height,
      components,
      componentStart,
      componentCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionGroupBlurYAccumulate(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !components || width <= 0 || height <= 0 || componentCount == 0u || componentCount > 4u) {
    setError(error, errorSize, "Invalid CUDA diffusion group blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionGroupBlurYAccumulateKernel<<<blockCount, blockSize>>>(
      source,
      accum,
      width,
      height,
      components,
      componentStart,
      componentCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionDownsample(
  const float *source,
  float *destination,
  int width,
  int height,
  uint32_t scale,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || width <= 0 || height <= 0 || scale == 0u) {
    setError(error, errorSize, "Invalid CUDA diffusion downsample arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  const int reducedWidth = (width + static_cast<int>(scale) - 1) / static_cast<int>(scale);
  const int reducedHeight = (height + static_cast<int>(scale) - 1) / static_cast<int>(scale);
  constexpr int blockSize = 256;
  const int pixelCount = reducedWidth * reducedHeight;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionDownsampleKernel<<<blockCount, blockSize>>>(source, destination, width, height, scale);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionReducedGroupBlurX(
  const float *source,
  float *destination,
  int reducedWidth,
  int reducedHeight,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !components || reducedWidth <= 0 || reducedHeight <= 0 ||
      scale == 0u || componentCount == 0u || componentCount > 4u) {
    setError(error, errorSize, "Invalid CUDA reduced diffusion group blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = reducedWidth * reducedHeight;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionReducedGroupBlurXKernel<<<blockCount, blockSize>>>(
      source,
      destination,
      reducedWidth,
      reducedHeight,
      scale,
      components,
      componentStart,
      componentCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionReducedGroupBlurY(
  const float *source,
  float *destination,
  int reducedWidth,
  int reducedHeight,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !components || reducedWidth <= 0 || reducedHeight <= 0 ||
      scale == 0u || componentCount == 0u || componentCount > 4u) {
    setError(error, errorSize, "Invalid CUDA reduced diffusion group blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = reducedWidth * reducedHeight;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionReducedGroupBlurYKernel<<<blockCount, blockSize>>>(
      source,
      destination,
      reducedWidth,
      reducedHeight,
      scale,
      components,
      componentStart,
      componentCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionReducedGroupUpsampleAccumulate(
  const float *source,
  float *accum,
  int width,
  int height,
  uint32_t scale,
  const KernelDiffusionComponent *components,
  uint32_t componentStart,
  uint32_t componentCount,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !components || width <= 0 || height <= 0 ||
      scale == 0u || componentCount == 0u || componentCount > 4u) {
    setError(error, errorSize, "Invalid CUDA reduced diffusion upsample arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionReducedGroupUpsampleAccumulateKernel<<<blockCount, blockSize>>>(
      source,
      accum,
      width,
      height,
      scale,
      components,
      componentStart,
      componentCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaDiffusionResolve(
  const float *source,
  const float *accum,
  float *destination,
  int width,
  int height,
  const KernelDiffusionInfo *info,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !destination || !info || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA diffusion resolve arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    diffusionResolveKernel<<<blockCount, blockSize>>>(source, accum, destination, pixelCount, info);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationBoostInfo(
  const float *raw,
  float *boostInfo,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!raw || !boostInfo || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation boost-info arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  const int pixelCount = width * height;
  return timedLaunch([&]() {
    halationBoostInfoKernel<<<1, 1>>>(raw, boostInfo, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationBoostApply(
  const float *raw,
  const float *boostInfo,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  (void)params;
  if (!raw || !boostInfo || !destination || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation boost-apply arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationBoostApplyKernel<<<blockCount, blockSize>>>(raw, boostInfo, destination, pixelCount);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationChannelBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  uint32_t mode,
  uint32_t component,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationChannelBlurXKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, mode, component);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationChannelBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  uint32_t mode,
  uint32_t component,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationChannelBlurYKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, mode, component);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationScatterTailBlurYAccumulate(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelParams *params,
  uint32_t component,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation scatter-tail blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationScatterTailBlurYAccumulateKernel<<<blockCount, blockSize>>>(source, accum, width, height, params, component);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationScatterResolve(
  const float *raw,
  const float *core,
  const float *tail,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!raw || !core || !tail || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation scatter-resolve arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationScatterResolveKernel<<<blockCount, blockSize>>>(raw, core, tail, destination, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationBounceBlurYAccumulate(
  const float *source,
  float *accum,
  int width,
  int height,
  const KernelParams *params,
  uint32_t bounce,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !accum || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation bounce blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationBounceBlurYAccumulateKernel<<<blockCount, blockSize>>>(source, accum, width, height, params, bounce);
  }, kernelMs, error, errorSize);
}

bool spektraCudaHalationResolveRaw(
  const float *raw,
  const float *halation,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!raw || !halation || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA halation resolve-raw arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    halationResolveRawKernel<<<blockCount, blockSize>>>(raw, halation, destination, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaProductionGrainLayersFromDensity(
  const float *density,
  float *layers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!density || !layers || !params || !spectralInfo || !curveInfo || !densityCurves || !paperScanDensityData ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA production grain layer arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int elementCount = width * height * 9;
  const unsigned int blockCount = static_cast<unsigned int>((elementCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    productionGrainLayersFromDensityKernel<<<blockCount, blockSize>>>(
      density, layers, width, height, params, spectralInfo, curveInfo, densityCurves, paperScanDensityData);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainSynthesisLayersFromDensity(
  const float *density,
  float *layers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!density || !layers || !params || !spectralInfo || !curveInfo || !densityCurves || !paperScanDensityData ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain synthesis layer arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int elementCount = width * height * 9;
  const unsigned int blockCount = static_cast<unsigned int>((elementCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainSynthesisLayersFromDensityKernel<<<blockCount, blockSize>>>(
      density, layers, width, height, params, spectralInfo, curveInfo, densityCurves, paperScanDensityData);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainLayerBlurX(
  const float *sourceLayers,
  float *destinationLayers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  (void)densityCurves;
  if (!sourceLayers || !destinationLayers || !params || !spectralInfo || !curveInfo || !paperScanDensityData ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain layer blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  const dim3 blockSize(16u, 16u, 1u);
  const dim3 blockCount(
    static_cast<unsigned int>((width + static_cast<int>(blockSize.x) - 1) / static_cast<int>(blockSize.x)),
    static_cast<unsigned int>((height + static_cast<int>(blockSize.y) - 1) / static_cast<int>(blockSize.y)),
    9u);
  return timedLaunch([&]() {
    grainLayerBlurKernel<<<blockCount, blockSize>>>(
      sourceLayers, destinationLayers, width, height, params, spectralInfo, curveInfo, paperScanDensityData, true);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainLayerBlurY(
  const float *sourceLayers,
  float *destinationLayers,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  const float *paperScanDensityData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  (void)densityCurves;
  if (!sourceLayers || !destinationLayers || !params || !spectralInfo || !curveInfo || !paperScanDensityData ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain layer blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  const dim3 blockSize(16u, 16u, 1u);
  const dim3 blockCount(
    static_cast<unsigned int>((width + static_cast<int>(blockSize.x) - 1) / static_cast<int>(blockSize.x)),
    static_cast<unsigned int>((height + static_cast<int>(blockSize.y) - 1) / static_cast<int>(blockSize.y)),
    9u);
  return timedLaunch([&]() {
    grainLayerBlurKernel<<<blockCount, blockSize>>>(
      sourceLayers, destinationLayers, width, height, params, spectralInfo, curveInfo, paperScanDensityData, false);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainMicrostructureSource(
  float *micro,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!micro || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain microstructure source arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainMicrostructureSourceKernel<<<blockCount, blockSize>>>(micro, width, height, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainMicroBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain micro blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainMicroBlurKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, true);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainMicroBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain micro blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainMicroBlurKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, false);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainResolveDensity(
  const float *layers,
  const float *micro,
  const float *sourceDensity,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!layers || !micro || !sourceDensity || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain resolve-density arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainResolveDensityKernel<<<blockCount, blockSize>>>(layers, micro, sourceDensity, destination, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainSynthesisResolveDensity(
  const float *layers,
  const float *micro,
  const float *sourceDensity,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!layers || !micro || !sourceDensity || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain synthesis resolve-density arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainSynthesisResolveDensityKernel<<<blockCount, blockSize>>>(layers, micro, sourceDensity, destination, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainApplyControls(
  const float *baseDensity,
  const float *grainedDensity,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!baseDensity || !grainedDensity || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain controls arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainApplyControlsKernel<<<blockCount, blockSize>>>(baseDensity, grainedDensity, destination, pixelCount, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainDensityBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain density blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainDensityBlurKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, true);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGrainDensityBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA grain density blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    grainDensityBlurKernel<<<blockCount, blockSize>>>(source, destination, width, height, params, false);
  }, kernelMs, error, errorSize);
}

bool spektraCudaPreviewGrainFromDensity(
  const float *density,
  float *grainedDensity,
  int width,
  int height,
  const KernelParams *params,
  const KernelCurveInfo *curveInfo,
  const float *densityCurves,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!density || !grainedDensity || !params || !curveInfo || !densityCurves || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA preview-grain arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    previewGrainFromDensityKernel<<<blockCount, blockSize>>>(
      density,
      grainedDensity,
      width,
      height,
      params,
      curveInfo,
      densityCurves);
  }, kernelMs, error, errorSize);
}

bool spektraCudaPrintRawFromFilmDensity(
  const float *filmDensity,
  float *printRaw,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool logOutput,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!filmDensity || !printRaw || !params || !spectralInfo || !colorInfo || !filmCurveInfo || !paperCurveInfo ||
      !filmLogExposure || !filmDensityCurves || !paperLogExposure || !paperDensityCurves ||
      !filmChannelDensity || !filmBaseDensity || !paperLogSensitivity || !thKg3Illuminant ||
      !customEnlargerFilters || !neutralPrintFilters || !academyPrinterDensityData ||
      !hanatosRawResponse || !mallettBasisIlluminant || !inputToReferenceXyz ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA print-raw arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    printRawFromFilmDensityKernel<<<blockCount, blockSize>>>(
      filmDensity,
      printRaw,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      filmCurveInfo,
      paperCurveInfo,
      filmLogExposure,
      filmDensityCurves,
      paperLogExposure,
      paperDensityCurves,
      filmChannelDensity,
      filmBaseDensity,
      paperLogSensitivity,
      thKg3Illuminant,
      customEnlargerFilters,
      neutralPrintFilters,
      academyPrinterDensityData,
      hanatosRawResponse,
      mallettBasisIlluminant,
      inputToReferenceXyz,
      logOutput);
  }, kernelMs, error, errorSize);
}

bool spektraCudaPrintRawFromNegativeLight(
  const float *source,
  float *printRaw,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *colorDecodeLut,
  const uint32_t *colorTransferKind,
  const float *inputToReferenceXyz,
  const float *paperHanatosResponse,
  const float *preflashPaperHanatosResponse,
  const float *academyPrinterDensityData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !printRaw || !params || !spectralInfo || !colorInfo || !colorDecodeLut ||
      !colorTransferKind || !inputToReferenceXyz || !paperHanatosResponse ||
      !preflashPaperHanatosResponse || !academyPrinterDensityData || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA process-negative arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    printRawFromNegativeLightKernel<<<blockCount, blockSize>>>(
      source,
      printRaw,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      colorDecodeLut,
      colorTransferKind,
      inputToReferenceXyz,
      paperHanatosResponse,
      preflashPaperHanatosResponse,
      academyPrinterDensityData);
  }, kernelMs, error, errorSize);
}

bool spektraCudaPrintDensityFromPrintRaw(
  const float *printRaw,
  float *printDensity,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!printRaw || !printDensity || !params || !spectralInfo || !paperCurveInfo ||
      !paperLogExposure || !paperDensityCurves || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA print-density arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    printDensityFromPrintRawKernel<<<blockCount, blockSize>>>(
      printRaw,
      printDensity,
      pixelCount,
      params,
      spectralInfo,
      paperCurveInfo,
      paperLogExposure,
      paperDensityCurves);
  }, kernelMs, error, errorSize);
}

bool spektraCudaFinalFromFilmDensity(
  const float *filmDensity,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind,
  const KernelFrameConstants *frameConstants,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool finalizeOutput,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!filmDensity || !destination || !params || !spectralInfo || !colorInfo || !filmCurveInfo || !paperCurveInfo ||
      !filmLogExposure || !filmDensityCurves || !paperLogExposure || !paperDensityCurves ||
      !filmChannelDensity || !filmBaseDensity || !paperLogSensitivity || !thKg3Illuminant ||
      !customEnlargerFilters || !neutralPrintFilters || !academyPrinterDensityData ||
      !paperScanDensityData || !scanIlluminantsAndCmfs || !scanToOutputRgbData ||
      !colorEncodeLut || !colorTransferKind || !frameConstants || !hanatosRawResponse || !mallettBasisIlluminant ||
      !inputToReferenceXyz || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA final-from-film-density arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    finalFromFilmDensityKernel<<<blockCount, blockSize>>>(
      filmDensity,
      destination,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      filmCurveInfo,
      paperCurveInfo,
      filmLogExposure,
      filmDensityCurves,
      paperLogExposure,
      paperDensityCurves,
      filmChannelDensity,
      filmBaseDensity,
      paperLogSensitivity,
      thKg3Illuminant,
      customEnlargerFilters,
      neutralPrintFilters,
      academyPrinterDensityData,
      paperScanDensityData,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData,
      colorEncodeLut,
      colorTransferKind,
      frameConstants,
      hanatosRawResponse,
      mallettBasisIlluminant,
      inputToReferenceXyz,
      finalizeOutput);
  }, kernelMs, error, errorSize);
}

bool spektraCudaFinalFromPrintRaw(
  const float *printRaw,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const KernelCurveInfo *filmCurveInfo,
  const KernelCurveInfo *paperCurveInfo,
  const float *filmLogExposure,
  const float *filmDensityCurves,
  const float *paperLogExposure,
  const float *paperDensityCurves,
  const float *filmChannelDensity,
  const float *filmBaseDensity,
  const float *paperLogSensitivity,
  const float *thKg3Illuminant,
  const float *customEnlargerFilters,
  const float *neutralPrintFilters,
  const float *academyPrinterDensityData,
  const float *paperScanDensityData,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind,
  const float *hanatosRawResponse,
  const float *mallettBasisIlluminant,
  const float *inputToReferenceXyz,
  bool finalizeOutput,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!printRaw || !destination || !params || !spectralInfo || !colorInfo || !filmCurveInfo || !paperCurveInfo ||
      !filmLogExposure || !filmDensityCurves || !paperLogExposure || !paperDensityCurves ||
      !filmChannelDensity || !filmBaseDensity || !paperLogSensitivity || !thKg3Illuminant ||
      !customEnlargerFilters || !neutralPrintFilters || !academyPrinterDensityData ||
      !paperScanDensityData || !scanIlluminantsAndCmfs || !scanToOutputRgbData ||
      !colorEncodeLut || !colorTransferKind || !hanatosRawResponse || !mallettBasisIlluminant ||
      !inputToReferenceXyz || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA final-from-print-raw arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    finalFromPrintRawKernel<<<blockCount, blockSize>>>(
      printRaw,
      destination,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      filmCurveInfo,
      paperCurveInfo,
      filmLogExposure,
      filmDensityCurves,
      paperLogExposure,
      paperDensityCurves,
      filmChannelDensity,
      filmBaseDensity,
      paperLogSensitivity,
      thKg3Illuminant,
      customEnlargerFilters,
      neutralPrintFilters,
      academyPrinterDensityData,
      paperScanDensityData,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData,
      colorEncodeLut,
      colorTransferKind,
      hanatosRawResponse,
      mallettBasisIlluminant,
      inputToReferenceXyz,
      finalizeOutput);
  }, kernelMs, error, errorSize);
}

bool spektraCudaGaussianBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  float sigma,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA gaussian-blur-x arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    gaussianBlurXKernel<<<blockCount, blockSize>>>(source, destination, width, height, fmaxf(sigma, 0.0f));
  }, kernelMs, error, errorSize);
}

bool spektraCudaGaussianBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  float sigma,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!source || !destination || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA gaussian-blur-y arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    gaussianBlurYKernel<<<blockCount, blockSize>>>(source, destination, width, height, fmaxf(sigma, 0.0f));
  }, kernelMs, error, errorSize);
}

bool spektraCudaPrintGlareGenerate(
  float *glareAmount,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!glareAmount || !params || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA print-glare-generate arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    printGlareGenerateKernel<<<blockCount, blockSize>>>(glareAmount, width, height, params);
  }, kernelMs, error, errorSize);
}

bool spektraCudaPrintGlareApply(
  const float *linearSource,
  const float *glareAmount,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  const KernelSpectralInfo *spectralInfo,
  const KernelColorInfo *colorInfo,
  const float *scanIlluminantsAndCmfs,
  const float *scanToOutputRgbData,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!linearSource || !glareAmount || !destination || !params || !spectralInfo || !colorInfo ||
      !scanIlluminantsAndCmfs || !scanToOutputRgbData || width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA print-glare-apply arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    printGlareApplyKernel<<<blockCount, blockSize>>>(
      linearSource,
      glareAmount,
      destination,
      pixelCount,
      params,
      spectralInfo,
      colorInfo,
      scanIlluminantsAndCmfs,
      scanToOutputRgbData);
  }, kernelMs, error, errorSize);
}

bool spektraCudaScannerFinalize(
  const float *linearSource,
  const float *unsharpBlur,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  const KernelColorInfo *colorInfo,
  const float *colorEncodeLut,
  const uint32_t *colorTransferKind,
  float *kernelMs,
  char *error,
  size_t errorSize
) {
  if (!linearSource || !destination || !params || !colorInfo || !colorEncodeLut || !colorTransferKind ||
      width <= 0 || height <= 0) {
    setError(error, errorSize, "Invalid CUDA scanner-finalize arguments");
    return false;
  }
  if (kernelMs) {
    *kernelMs = 0.0f;
  }
  constexpr int blockSize = 256;
  const int pixelCount = width * height;
  const unsigned int blockCount = static_cast<unsigned int>((pixelCount + blockSize - 1) / blockSize);
  return timedLaunch([&]() {
    scannerFinalizeKernel<<<blockCount, blockSize>>>(
      linearSource,
      unsharpBlur,
      destination,
      pixelCount,
      params,
      colorInfo,
      colorEncodeLut,
      colorTransferKind);
  }, kernelMs, error, errorSize);
}

} // namespace spektrafilm
