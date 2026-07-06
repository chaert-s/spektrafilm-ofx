#include "SpektraVulkanRenderer.h"
#include "SpektraProfileCurves.h"

#include <vulkan/vulkan.h>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <dlfcn.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace spektrafilm {

namespace {

int moduleImageAnchor = 0;

using PerfClock = std::chrono::steady_clock;

double elapsedMilliseconds(PerfClock::time_point start, PerfClock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

bool envFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value) {
    return false;
  }
  return std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 ||
         std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

bool envFlagEnabledOrDefault(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return defaultValue;
  }
  return std::strcmp(value, "1") == 0 ||
         std::strcmp(value, "true") == 0 ||
         std::strcmp(value, "TRUE") == 0 ||
         std::strcmp(value, "yes") == 0 ||
         std::strcmp(value, "YES") == 0 ||
         std::strcmp(value, "on") == 0 ||
         std::strcmp(value, "ON") == 0;
}

bool envFlagEnabledOrDefaultAny(
  const std::initializer_list<const char *> &names,
  bool defaultValue
) {
  for (const char *name : names) {
    const char *value = std::getenv(name);
    if (value && value[0] != '\0') {
      return envFlagEnabledOrDefault(name, defaultValue);
    }
  }
  return defaultValue;
}

std::string envString(const char *name, const char *defaultValue) {
  const char *value = std::getenv(name);
  return value && value[0] ? std::string(value) : std::string(defaultValue);
}

std::string envStringAny(
  const std::initializer_list<const char *> &names,
  const char *defaultValue
) {
  for (const char *name : names) {
    const char *value = std::getenv(name);
    if (value && value[0] != '\0') {
      return std::string(value);
    }
  }
  return std::string(defaultValue);
}

std::filesystem::path envPath(const char *name) {
  const char *value = std::getenv(name);
  return value && value[0] ? std::filesystem::path(value) : std::filesystem::path();
}

std::filesystem::path modulePath() {
#if defined(_WIN32)
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&moduleImageAnchor),
        &module
      )) {
    return {};
  }

  std::vector<wchar_t> path(MAX_PATH);
  for (;;) {
    const DWORD length = GetModuleFileNameW(module, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0) {
      return {};
    }
    if (length < path.size() - 1) {
      return std::filesystem::path(std::wstring(path.data(), length));
    }
    path.resize(path.size() * 2u);
  }
#elif defined(__linux__)
  Dl_info imageInfo{};
  if (dladdr(&moduleImageAnchor, &imageInfo) == 0 || !imageInfo.dli_fname) {
    return {};
  }
  return std::filesystem::path(imageInfo.dli_fname);
#else
  return {};
#endif
}

std::filesystem::path findResourcePath(const std::filesystem::path &relativePath) {
  std::vector<std::filesystem::path> directories;
  if (std::filesystem::path resourceDir = envPath("SPEKTRAFILM_RESOURCE_DIR"); !resourceDir.empty()) {
    directories.push_back(resourceDir);
  }

  const std::filesystem::path imagePath = modulePath();
  if (!imagePath.empty()) {
    const std::filesystem::path contentsPath = imagePath.parent_path().parent_path();
    if (!contentsPath.empty()) {
      directories.push_back(contentsPath / "Resources");
    }
    if (!imagePath.parent_path().empty()) {
      directories.push_back(imagePath.parent_path());
    }
  }

  directories.push_back(std::filesystem::current_path());

  for (const std::filesystem::path &directory : directories) {
    const std::filesystem::path candidate = directory / relativePath;
    std::error_code ec;
    if (std::filesystem::is_regular_file(candidate, ec)) {
      return candidate;
    }
  }
  return {};
}

std::vector<uint32_t> readSpirvFile(const std::filesystem::path &path, std::string &error) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    error = "Unable to open Vulkan shader module " + path.string() + ".";
    return {};
  }
  const std::streamoff byteSize = static_cast<std::streamoff>(input.tellg());
  if (byteSize <= 0 || (static_cast<uint64_t>(byteSize) % sizeof(uint32_t)) != 0u) {
    error = "Vulkan shader module has an invalid SPIR-V byte size: " + path.string() + ".";
    return {};
  }
  input.seekg(0, std::ios::beg);
  std::vector<uint32_t> data(static_cast<size_t>(byteSize) / sizeof(uint32_t));
  if (!input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byteSize))) {
    error = "Unable to read Vulkan shader module " + path.string() + ".";
    return {};
  }
  return data;
}

std::vector<float> readFloatResourceFile(
  const std::filesystem::path &path,
  uint32_t expectedElementCount,
  std::string &error
) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    error = "Unable to open Vulkan float resource " + path.string() + ".";
    return {};
  }
  const std::streamoff byteSize = static_cast<std::streamoff>(input.tellg());
  const uint64_t expectedBytes = static_cast<uint64_t>(expectedElementCount) * sizeof(float);
  if (byteSize <= 0 || static_cast<uint64_t>(byteSize) != expectedBytes) {
    error = "Vulkan float resource has an unexpected byte size: " + path.string() + ".";
    return {};
  }
  input.seekg(0, std::ios::beg);
  std::vector<float> data(expectedElementCount);
  if (!input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byteSize))) {
    error = "Unable to read Vulkan float resource " + path.string() + ".";
    return {};
  }
  return data;
}

bool isSupportedRgba(const ImageView &source, const MutableImageView &destination) {
  return source.components == 4 &&
         destination.components == 4 &&
         (source.bytesPerComponent == 2 || source.bytesPerComponent == 4) &&
         (destination.bytesPerComponent == 2 || destination.bytesPerComponent == 4);
}

bool windowFitsView(
  const ImageView &view,
  const RenderWindow &window,
  int32_t width,
  int32_t height
) {
  const int32_t sourceX = window.x1 - view.x1;
  const int32_t sourceY = window.y1 - view.y1;
  return sourceX >= 0 &&
         sourceY >= 0 &&
         sourceX + width <= view.width &&
         sourceY + height <= view.height;
}

bool windowFitsView(
  const MutableImageView &view,
  const RenderWindow &window,
  int32_t width,
  int32_t height
) {
  const int32_t destinationX = window.x1 - view.x1;
  const int32_t destinationY = window.y1 - view.y1;
  return destinationX >= 0 &&
         destinationY >= 0 &&
         destinationX + width <= view.width &&
         destinationY + height <= view.height;
}

bool copyWindowToMappedBytes(
  const ImageView &source,
  const RenderWindow &window,
  int32_t width,
  int32_t height,
  void *destinationBytes,
  size_t destinationByteCount
) {
  if (!source.data || !destinationBytes || !windowFitsView(source, window, width, height)) {
    return false;
  }

  const int32_t pixelBytes = source.components * source.bytesPerComponent;
  const int32_t rowBytes = width * pixelBytes;
  const size_t byteCount = static_cast<size_t>(height) * static_cast<size_t>(rowBytes);
  if (destinationByteCount < byteCount) {
    return false;
  }

  const auto *sourceBase = static_cast<const uint8_t *>(source.data);
  auto *destinationBase = static_cast<uint8_t *>(destinationBytes);
  const int32_t sourceX = window.x1 - source.x1;
  const int32_t sourceY0 = window.y1 - source.y1;
  if (sourceX == 0 && source.rowBytes == rowBytes) {
    std::memcpy(
      destinationBase,
      sourceBase + static_cast<size_t>(sourceY0) * static_cast<size_t>(source.rowBytes),
      byteCount
    );
    return true;
  }

  for (int32_t y = 0; y < height; ++y) {
    const int32_t sourceY = sourceY0 + y;
    const auto *src = sourceBase +
      static_cast<size_t>(sourceY) * static_cast<size_t>(source.rowBytes) +
      static_cast<size_t>(sourceX) * static_cast<size_t>(pixelBytes);
    uint8_t *dst = destinationBase + static_cast<size_t>(y) * static_cast<size_t>(rowBytes);
    std::memcpy(dst, src, static_cast<size_t>(rowBytes));
  }
  return true;
}

bool copyMappedBytesToWindow(
  const void *sourceBytes,
  size_t sourceByteCount,
  const MutableImageView &destination,
  const RenderWindow &window,
  int32_t width,
  int32_t height
) {
  if (!sourceBytes || !destination.data || !windowFitsView(destination, window, width, height)) {
    return false;
  }

  const int32_t pixelBytes = destination.components * destination.bytesPerComponent;
  const int32_t rowBytes = width * pixelBytes;
  const size_t byteCount = static_cast<size_t>(height) * static_cast<size_t>(rowBytes);
  if (sourceByteCount < byteCount) {
    return false;
  }

  auto *destinationBase = static_cast<uint8_t *>(destination.data);
  const auto *sourceBase = static_cast<const uint8_t *>(sourceBytes);
  const int32_t destinationX = window.x1 - destination.x1;
  const int32_t destinationY0 = window.y1 - destination.y1;
  if (destinationX == 0 && destination.rowBytes == rowBytes) {
    std::memcpy(
      destinationBase + static_cast<size_t>(destinationY0) * static_cast<size_t>(destination.rowBytes),
      sourceBase,
      byteCount
    );
    return true;
  }

  for (int32_t y = 0; y < height; ++y) {
    const int32_t destinationY = destinationY0 + y;
    auto *dst = destinationBase +
      static_cast<size_t>(destinationY) * static_cast<size_t>(destination.rowBytes) +
      static_cast<size_t>(destinationX) * static_cast<size_t>(pixelBytes);
    const uint8_t *src = sourceBase + static_cast<size_t>(y) * static_cast<size_t>(rowBytes);
    std::memcpy(dst, src, static_cast<size_t>(rowBytes));
  }
  return true;
}

bool copyMappedBytesRegionToWindow(
  const void *sourceBytes,
  size_t sourceByteCount,
  int32_t sourceWidth,
  int32_t sourceHeight,
  int32_t sourceX,
  int32_t sourceY,
  const MutableImageView &destination,
  const RenderWindow &window,
  int32_t width,
  int32_t height
) {
  if (!sourceBytes || !destination.data || sourceWidth <= 0 || sourceHeight <= 0 ||
      sourceX < 0 || sourceY < 0 || sourceX + width > sourceWidth || sourceY + height > sourceHeight ||
      !windowFitsView(destination, window, width, height)) {
    return false;
  }

  const int32_t pixelBytes = destination.components * destination.bytesPerComponent;
  const int32_t sourceRowBytes = sourceWidth * pixelBytes;
  const int32_t copyRowBytes = width * pixelBytes;
  const size_t requiredBytes =
    static_cast<size_t>(sourceHeight) * static_cast<size_t>(sourceRowBytes);
  if (sourceByteCount < requiredBytes) {
    return false;
  }

  const auto *sourceBase = static_cast<const uint8_t *>(sourceBytes);
  auto *destinationBase = static_cast<uint8_t *>(destination.data);
  const int32_t destinationX = window.x1 - destination.x1;
  const int32_t destinationY0 = window.y1 - destination.y1;
  for (int32_t y = 0; y < height; ++y) {
    const uint8_t *src = sourceBase +
      static_cast<size_t>(sourceY + y) * static_cast<size_t>(sourceRowBytes) +
      static_cast<size_t>(sourceX) * static_cast<size_t>(pixelBytes);
    uint8_t *dst = destinationBase +
      static_cast<size_t>(destinationY0 + y) * static_cast<size_t>(destination.rowBytes) +
      static_cast<size_t>(destinationX) * static_cast<size_t>(pixelBytes);
    std::memcpy(dst, src, static_cast<size_t>(copyRowBytes));
  }
  return true;
}

uint32_t halfToFloatBits(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
  uint32_t exponent = (h >> 10) & 0x1fu;
  uint32_t mantissa = h & 0x03ffu;

  if (exponent == 0) {
    if (mantissa == 0) {
      return sign;
    }
    exponent = 1;
    while ((mantissa & 0x0400u) == 0) {
      mantissa <<= 1;
      --exponent;
    }
    mantissa &= 0x03ffu;
  } else if (exponent == 31) {
    return sign | 0x7f800000u | (mantissa << 13);
  }

  exponent = exponent + (127 - 15);
  return sign | (exponent << 23) | (mantissa << 13);
}

float halfToFloat(uint16_t h) {
  const uint32_t bits = halfToFloatBits(h);
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

bool copySourceToFloatStaging(
  const ImageView &source,
  const RenderWindow &window,
  int32_t width,
  int32_t height,
  float *staging
) {
  if (!source.data || !staging || !windowFitsView(source, window, width, height)) {
    return false;
  }

  const auto *sourceBase = static_cast<const uint8_t *>(source.data);
  if (source.bytesPerComponent == 4) {
    const size_t rowByteCount = static_cast<size_t>(width) * 4u * sizeof(float);
    for (int32_t y = 0; y < height; ++y) {
      const int32_t sourceY = window.y1 + y - source.y1;
      const int32_t sourceX = window.x1 - source.x1;
      const auto *row = sourceBase +
        static_cast<size_t>(sourceY) * static_cast<size_t>(source.rowBytes) +
        static_cast<size_t>(sourceX) * 4u * sizeof(float);
      std::memcpy(
        staging + static_cast<size_t>(y) * static_cast<size_t>(width) * 4u,
        row,
        rowByteCount
      );
    }
    return true;
  }

  for (int32_t y = 0; y < height; ++y) {
    const int32_t sourceY = window.y1 + y - source.y1;
    const int32_t sourceX = window.x1 - source.x1;
    const auto *row = sourceBase +
      static_cast<size_t>(sourceY) * static_cast<size_t>(source.rowBytes) +
      static_cast<size_t>(sourceX) * static_cast<size_t>(source.components) * static_cast<size_t>(source.bytesPerComponent);
    for (int32_t x = 0; x < width; ++x) {
      float *dst = staging + (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
      if (source.bytesPerComponent == 4) {
        const auto *src = reinterpret_cast<const float *>(row + static_cast<size_t>(x) * 4u * sizeof(float));
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
      } else {
        const auto *src = reinterpret_cast<const uint16_t *>(row + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
        dst[0] = halfToFloat(src[0]);
        dst[1] = halfToFloat(src[1]);
        dst[2] = halfToFloat(src[2]);
        dst[3] = halfToFloat(src[3]);
      }
    }
  }
  return true;
}

std::vector<float> makeLinearSensitivity(const float *logSensitivity, uint32_t wavelengthCount) {
  std::vector<float> linear(static_cast<size_t>(wavelengthCount) * 3u, 0.0f);
  if (!logSensitivity) {
    return linear;
  }
  for (uint32_t i = 0; i < wavelengthCount * 3u; ++i) {
    const float value = std::pow(10.0f, logSensitivity[i]);
    linear[i] = std::isfinite(value) ? value : 0.0f;
  }
  return linear;
}

float smoothErfEdge(float wavelength, float center, float width) {
  return std::erf((wavelength - center) / width) * 0.5f + 0.5f;
}

std::vector<float> applyCameraBandPass(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity,
  const RenderParams &params
) {
  std::vector<float> filtered = linearSensitivity;
  if (!filmCurves.wavelengths ||
      filtered.size() < static_cast<size_t>(filmCurves.wavelengthCount) * 3u ||
      (!params.cameraUvFilterEnabled && !params.cameraIrFilterEnabled)) {
    return filtered;
  }

  constexpr float kPythonUvTransitionNm = 8.0f;
  constexpr float kPythonIrTransitionNm = 15.0f;
  std::array<float, 3> numerator = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> denominator = {0.0f, 0.0f, 0.0f};
  std::vector<float> transmissionByWavelength(filmCurves.wavelengthCount, 1.0f);
  for (uint32_t wavelength = 0; wavelength < filmCurves.wavelengthCount; ++wavelength) {
    const float wl = filmCurves.wavelengths[wavelength];
    const float uvTransmission = params.cameraUvFilterEnabled
      ? smoothErfEdge(wl, params.cameraUvCutNm, kPythonUvTransitionNm)
      : 1.0f;
    const float irTransmission = params.cameraIrFilterEnabled
      ? smoothErfEdge(wl, params.cameraIrCutNm, -kPythonIrTransitionNm)
      : 1.0f;
    const float transmission = uvTransmission * irTransmission;
    transmissionByWavelength[wavelength] = transmission;
    const uint32_t offset = wavelength * 3u;
    const float illuminant = filmCurves.referenceIlluminantSpectrum ? filmCurves.referenceIlluminantSpectrum[wavelength] : 1.0f;
    for (uint32_t channel = 0; channel < 3u; ++channel) {
      const float response = linearSensitivity[offset + channel] * illuminant;
      denominator[channel] += response;
      numerator[channel] += response * transmission;
    }
  }

  std::array<float, 3> normalization = {1.0f, 1.0f, 1.0f};
  for (uint32_t channel = 0; channel < 3u; ++channel) {
    normalization[channel] = numerator[channel] / std::max(denominator[channel], 1.0e-10f);
    normalization[channel] = std::max(normalization[channel], 1.0e-10f);
  }
  for (uint32_t wavelength = 0; wavelength < filmCurves.wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    for (uint32_t channel = 0; channel < 3u; ++channel) {
      filtered[offset + channel] *= transmissionByWavelength[wavelength] / normalization[channel];
    }
  }
  return filtered;
}

std::array<float, 9> makeMallettRawMatrix(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity
) {
  std::array<float, 9> matrix{};
  const uint32_t wavelengthCount = filmCurves.wavelengthCount;
  if (!filmCurves.mallettBasisIlluminant ||
      linearSensitivity.size() < static_cast<size_t>(wavelengthCount) * 3u) {
    return matrix;
  }
  for (uint32_t wavelength = 0; wavelength < wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    for (uint32_t outChannel = 0; outChannel < 3u; ++outChannel) {
      for (uint32_t inChannel = 0; inChannel < 3u; ++inChannel) {
        matrix[outChannel * 3u + inChannel] +=
          linearSensitivity[offset + outChannel] * filmCurves.mallettBasisIlluminant[offset + inChannel];
      }
    }
  }

  const float normalization = std::max(filmCurves.mallettRawMidgrayGreen, 1.0e-10f);
  for (float &value : matrix) {
    value /= normalization;
  }
  return matrix;
}

std::array<float, 3> densityCurveMaximums(const ProfileCurveSet &curves) {
  std::array<float, 3> maximums = {0.0f, 0.0f, 0.0f};
  if (!curves.densityCurves || curves.exposureCount == 0u) {
    return maximums;
  }
  for (uint32_t index = 0; index < curves.exposureCount; ++index) {
    const float *density = curves.densityCurves + static_cast<size_t>(index) * 3u;
    maximums[0] = std::max(maximums[0], density[0]);
    maximums[1] = std::max(maximums[1], density[1]);
    maximums[2] = std::max(maximums[2], density[2]);
  }
  return maximums;
}

std::vector<float> makePackedCurveExposure(const float *logExposure, uint32_t exposureCount) {
  std::vector<float> packed(static_cast<size_t>(exposureCount) * 2u, 0.0f);
  for (uint32_t index = 0u; index < exposureCount; ++index) {
    const size_t offset = static_cast<size_t>(index) * 2u;
    packed[offset] = logExposure[index];
    if (index + 1u < exposureCount) {
      packed[offset + 1u] = 1.0f / std::max(logExposure[index + 1u] - logExposure[index], 1.0e-9f);
    }
  }
  return packed;
}

std::vector<float> makePackedSpectralDensity(
  const float *channelDensity,
  const float *baseDensity,
  uint32_t wavelengthCount
) {
  std::vector<float> packed(static_cast<size_t>(wavelengthCount) * 4u, 0.0f);
  for (uint32_t wavelength = 0u; wavelength < wavelengthCount; ++wavelength) {
    const size_t sourceOffset = static_cast<size_t>(wavelength) * 3u;
    const size_t offset = static_cast<size_t>(wavelength) * 4u;
    packed[offset] = channelDensity[sourceOffset];
    packed[offset + 1u] = channelDensity[sourceOffset + 1u];
    packed[offset + 2u] = channelDensity[sourceOffset + 2u];
    packed[offset + 3u] = baseDensity[wavelength];
  }
  return packed;
}

std::vector<float> makeScanProducts(
  const float *filmIlluminant,
  const float *paperIlluminant,
  const float *filmBaseDensity,
  const float *paperBaseDensity,
  const float *cmfs,
  uint32_t wavelengthCount
) {
  const size_t packedFloatCount = static_cast<size_t>(wavelengthCount) * 8u;
  const size_t legacyFloatCount = static_cast<size_t>(wavelengthCount) * 8u;
  std::vector<float> products(packedFloatCount + legacyFloatCount + 4u, 0.0f);
  const float *illuminants[] = {filmIlluminant, paperIlluminant};
  const float *baseDensities[] = {filmBaseDensity, paperBaseDensity};
  for (uint32_t stage = 0u; stage < 2u; ++stage) {
    float normalization = 0.0f;
    const size_t packedStageOffset = static_cast<size_t>(stage) * wavelengthCount * 4u;
    const size_t legacyStageOffset = packedFloatCount + static_cast<size_t>(stage) * wavelengthCount * 4u;
    for (uint32_t wavelength = 0u; wavelength < wavelengthCount; ++wavelength) {
      const size_t channelOffset = static_cast<size_t>(wavelength) * 3u;
      const size_t packedOffset = packedStageOffset + static_cast<size_t>(wavelength) * 4u;
      const size_t legacyOffset = legacyStageOffset + static_cast<size_t>(wavelength) * 4u;
      const float illuminant = illuminants[stage][wavelength];
      const float baseTransmittanceRaw = std::pow(10.0f, -baseDensities[stage][wavelength]);
      const float baseTransmittance = std::isfinite(baseTransmittanceRaw) ? baseTransmittanceRaw : 0.0f;
      for (uint32_t channel = 0u; channel < 3u; ++channel) {
        const float product = illuminant * cmfs[channelOffset + channel];
        products[packedOffset + channel] = baseTransmittance * product;
        products[legacyOffset + channel] = product;
      }
      normalization += illuminant * cmfs[channelOffset + 1u];
    }
    products[packedFloatCount + legacyFloatCount + stage] =
      1.0f / std::max(normalization, 1.0e-10f);
  }
  return products;
}

std::vector<float> makeHanatosRawResponse(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity,
  const std::vector<float> &hanatosSpectra,
  const HanatosSpectraLutInfo &hanatos,
  RgbToRawMethod method
) {
  const size_t responseCount = static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) * 3u;
  std::vector<float> response(responseCount, 0.0f);
  const size_t expectedSpectra =
    static_cast<size_t>(hanatos.width) *
    static_cast<size_t>(hanatos.height) *
    static_cast<size_t>(hanatos.wavelengthCount);
  if (hanatos.width == 0u ||
      hanatos.height == 0u ||
      hanatos.wavelengthCount == 0u ||
      hanatosSpectra.size() < expectedSpectra ||
      linearSensitivity.size() < static_cast<size_t>(hanatos.wavelengthCount) * 3u) {
    return response;
  }

  std::vector<float> hanatos2026Window(hanatos.wavelengthCount, 1.0f);
  std::array<float, 3> hanatos2026Normalization = {1.0f, 1.0f, 1.0f};
  const bool useHanatos2026 =
    method == RgbToRawMethod::Hanatos2026 &&
    filmCurves.hanatos2026WindowParams &&
    filmCurves.referenceIlluminantSpectrum;
  if (useHanatos2026) {
    constexpr float kSqrt2 = 1.4142135623730951f;
    const float cUv = filmCurves.hanatos2026WindowParams[0];
    const float sigmaUv = filmCurves.hanatos2026WindowParams[1];
    const float cIr = filmCurves.hanatos2026WindowParams[2];
    const float sigmaIr = filmCurves.hanatos2026WindowParams[3];
    if (sigmaUv > 0.0f && sigmaIr > 0.0f) {
      std::array<float, 3> numerator = {0.0f, 0.0f, 0.0f};
      std::array<float, 3> denominator = {0.0f, 0.0f, 0.0f};
      for (uint32_t wavelength = 0; wavelength < hanatos.wavelengthCount; ++wavelength) {
        const float wl = filmCurves.wavelengths ? filmCurves.wavelengths[wavelength] : 0.0f;
        const float edgeUv = smoothErfEdge(wl, cUv, sigmaUv * kSqrt2);
        const float edgeIr = smoothErfEdge(wl, cIr, -sigmaIr * kSqrt2);
        const float window = edgeUv * edgeIr;
        hanatos2026Window[wavelength] = window;
        const float illuminant = filmCurves.referenceIlluminantSpectrum[wavelength];
        const uint32_t sensitivityOffset = wavelength * 3u;
        for (uint32_t channel = 0; channel < 3u; ++channel) {
          const float referenceResponse = linearSensitivity[sensitivityOffset + channel] * illuminant;
          denominator[channel] += referenceResponse;
          numerator[channel] += referenceResponse * window;
        }
      }
      for (uint32_t channel = 0; channel < 3u; ++channel) {
        hanatos2026Normalization[channel] =
          numerator[channel] / std::max(denominator[channel], 1.0e-10f);
        hanatos2026Normalization[channel] = std::max(hanatos2026Normalization[channel], 1.0e-10f);
      }
    }
  }
  if (!useHanatos2026 && !filmCurves.bandpassHanatos2025) {
    return response;
  }

  for (uint32_t x = 0; x < hanatos.width; ++x) {
    for (uint32_t y = 0; y < hanatos.height; ++y) {
      std::array<float, 3> raw = {0.0f, 0.0f, 0.0f};
      const size_t spectraOffset =
        (static_cast<size_t>(x) * static_cast<size_t>(hanatos.height) + static_cast<size_t>(y)) *
        static_cast<size_t>(hanatos.wavelengthCount);
      for (uint32_t wavelength = 0; wavelength < hanatos.wavelengthCount; ++wavelength) {
        const float spectrum = hanatosSpectra[spectraOffset + wavelength];
        const uint32_t sensitivityOffset = wavelength * 3u;
        if (useHanatos2026) {
          const float window = hanatos2026Window[wavelength];
          raw[0] += spectrum * linearSensitivity[sensitivityOffset] * window / hanatos2026Normalization[0];
          raw[1] += spectrum * linearSensitivity[sensitivityOffset + 1u] * window / hanatos2026Normalization[1];
          raw[2] += spectrum * linearSensitivity[sensitivityOffset + 2u] * window / hanatos2026Normalization[2];
        } else {
          raw[0] += spectrum * linearSensitivity[sensitivityOffset] * filmCurves.bandpassHanatos2025[sensitivityOffset];
          raw[1] += spectrum * linearSensitivity[sensitivityOffset + 1u] * filmCurves.bandpassHanatos2025[sensitivityOffset + 1u];
          raw[2] += spectrum * linearSensitivity[sensitivityOffset + 2u] * filmCurves.bandpassHanatos2025[sensitivityOffset + 2u];
        }
      }
      const size_t responseOffset =
        (static_cast<size_t>(x) * static_cast<size_t>(hanatos.height) + static_cast<size_t>(y)) * 3u;
      response[responseOffset] = raw[0];
      response[responseOffset + 1u] = raw[1];
      response[responseOffset + 2u] = raw[2];
    }
  }
  return response;
}

// Native port of Andrea Volpato's spektrafilm/forum gamut-compression approach:
// remap the Hanatos response toward the visible locus and let shaders select it.
float reinhardKnee(float value, float threshold, float limit, float power) {
  if (!std::isfinite(value) || value <= threshold) {
    return value;
  }
  const float scale = std::max(limit - threshold, 1.0e-12f);
  const float x = (value - threshold) / scale;
  const float y = x / std::pow(1.0f + std::pow(x, power), 1.0f / power);
  return threshold + scale * y;
}

std::array<float, 2> filmReferenceIlluminantXy(const ProfileCurveSet &filmCurves) {
  const float *cmfs = standardObserverCmfs();
  if (!filmCurves.referenceIlluminantSpectrum || !cmfs || filmCurves.wavelengthCount == 0u) {
    return {1.0f / 3.0f, 1.0f / 3.0f};
  }
  std::array<float, 3> xyz = {0.0f, 0.0f, 0.0f};
  for (uint32_t wavelength = 0; wavelength < filmCurves.wavelengthCount; ++wavelength) {
    const float illuminant = filmCurves.referenceIlluminantSpectrum[wavelength];
    const uint32_t offset = wavelength * 3u;
    xyz[0] += illuminant * cmfs[offset];
    xyz[1] += illuminant * cmfs[offset + 1u];
    xyz[2] += illuminant * cmfs[offset + 2u];
  }
  const float sum = xyz[0] + xyz[1] + xyz[2];
  if (!(sum > 1.0e-12f) || !std::isfinite(sum)) {
    return {1.0f / 3.0f, 1.0f / 3.0f};
  }
  return {xyz[0] / sum, xyz[1] / sum};
}

std::vector<std::array<float, 2>> spectralLocusXy(const ProfileCurveSet &filmCurves) {
  std::vector<std::array<float, 2>> locus;
  const float *cmfs = standardObserverCmfs();
  if (!filmCurves.wavelengths || !cmfs) {
    return locus;
  }
  for (uint32_t wavelength = 0; wavelength < filmCurves.wavelengthCount; ++wavelength) {
    if (filmCurves.wavelengths[wavelength] > 700.0f + 1.0e-4f) {
      continue;
    }
    const uint32_t offset = wavelength * 3u;
    const float x = cmfs[offset];
    const float y = cmfs[offset + 1u];
    const float z = cmfs[offset + 2u];
    const float sum = x + y + z;
    if (sum > 1.0e-12f && std::isfinite(sum)) {
      locus.push_back({x / sum, y / sum});
    }
  }
  if (!locus.empty()) {
    locus.push_back(locus.front());
  }
  return locus;
}

float rayPolygonDistance(
  std::array<float, 2> origin,
  std::array<float, 2> direction,
  const std::vector<std::array<float, 2>> &polygon
) {
  float tMin = INFINITY;
  for (size_t edgeIndex = 0; edgeIndex + 1u < polygon.size(); ++edgeIndex) {
    const float ax = polygon[edgeIndex][0];
    const float ay = polygon[edgeIndex][1];
    const float ex = polygon[edgeIndex + 1u][0] - ax;
    const float ey = polygon[edgeIndex + 1u][1] - ay;
    const float denom = direction[0] * ey - direction[1] * ex;
    if (std::abs(denom) <= 1.0e-12f) {
      continue;
    }
    const float ox = origin[0] - ax;
    const float oy = origin[1] - ay;
    const float t = (-ox * ey + oy * ex) / denom;
    const float s = (-ox * direction[1] + oy * direction[0]) / denom;
    if (t > 1.0e-9f && s >= 0.0f && s <= 1.0f && t < tMin) {
      tMin = t;
    }
  }
  return tMin;
}

std::array<float, 2> compressXyRadial(
  std::array<float, 2> xy,
  std::array<float, 2> whiteXy,
  const std::vector<std::array<float, 2>> &locus
) {
  const float dx = xy[0] - whiteXy[0];
  const float dy = xy[1] - whiteXy[1];
  const float distance = std::sqrt(dx * dx + dy * dy);
  if (!(distance > 1.0e-9f) || locus.size() < 4u) {
    return xy;
  }
  const std::array<float, 2> direction = {dx / distance, dy / distance};
  const float boundary = rayPolygonDistance(whiteXy, direction, locus);
  if (!(boundary > 1.0e-12f) || !std::isfinite(boundary)) {
    return xy;
  }
  const float normalized = distance / boundary;
  const float compressed = reinhardKnee(normalized, 0.0f, 1.0f, 6.0f);
  const float newDistance = compressed * boundary;
  return {whiteXy[0] + direction[0] * newDistance, whiteXy[1] + direction[1] * newDistance};
}

std::array<float, 3> sampleHanatosResponseBilinear(
  const std::vector<float> &response,
  const HanatosSpectraLutInfo &hanatos,
  float x,
  float y
) {
  x = std::clamp(x, 0.0f, static_cast<float>(hanatos.width - 1u));
  y = std::clamp(y, 0.0f, static_cast<float>(hanatos.height - 1u));
  const uint32_t x0 = static_cast<uint32_t>(std::floor(x));
  const uint32_t y0 = static_cast<uint32_t>(std::floor(y));
  const uint32_t x1 = std::min(x0 + 1u, hanatos.width - 1u);
  const uint32_t y1 = std::min(y0 + 1u, hanatos.height - 1u);
  const float tx = x - static_cast<float>(x0);
  const float ty = y - static_cast<float>(y0);
  const auto valueAt = [&](uint32_t xi, uint32_t yi) -> std::array<float, 3> {
    const size_t offset = (static_cast<size_t>(xi) * hanatos.height + yi) * 3u;
    return {response[offset], response[offset + 1u], response[offset + 2u]};
  };
  const std::array<float, 3> v00 = valueAt(x0, y0);
  const std::array<float, 3> v10 = valueAt(x1, y0);
  const std::array<float, 3> v01 = valueAt(x0, y1);
  const std::array<float, 3> v11 = valueAt(x1, y1);
  std::array<float, 3> out{};
  for (uint32_t channel = 0; channel < 3u; ++channel) {
    const float a = v00[channel] + (v10[channel] - v00[channel]) * tx;
    const float b = v01[channel] + (v11[channel] - v01[channel]) * tx;
    out[channel] = a + (b - a) * ty;
  }
  return out;
}

std::vector<float> remapHanatosResponseForInputGamutCompression(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &baseResponse,
  const HanatosSpectraLutInfo &hanatos
) {
  std::vector<float> remapped(baseResponse.size(), 0.0f);
  if (hanatos.width < 2u || hanatos.height < 2u || baseResponse.size() < static_cast<size_t>(hanatos.width) * hanatos.height * 3u) {
    return remapped;
  }
  const std::array<float, 2> whiteXy = filmReferenceIlluminantXy(filmCurves);
  const std::vector<std::array<float, 2>> locus = spectralLocusXy(filmCurves);
  if (locus.size() < 4u) {
    return baseResponse;
  }
  for (uint32_t x = 0; x < hanatos.width; ++x) {
    const float tx = static_cast<float>(x) / static_cast<float>(hanatos.width - 1u);
    const float rootTx = std::sqrt(std::max(tx, 0.0f));
    for (uint32_t y = 0; y < hanatos.height; ++y) {
      const float ty = static_cast<float>(y) / static_cast<float>(hanatos.height - 1u);
      const std::array<float, 2> xy = {1.0f - rootTx, ty * rootTx};
      const std::array<float, 2> compressedXy = compressXyRadial(xy, whiteXy, locus);
      const float oneMinusX = std::max(1.0f - compressedXy[0], 1.0e-10f);
      const float sampleTx = std::clamp(oneMinusX * oneMinusX, 0.0f, 1.0f);
      const float sampleTy = std::clamp(compressedXy[1] / oneMinusX, 0.0f, 1.0f);
      const std::array<float, 3> sampled = sampleHanatosResponseBilinear(
        baseResponse,
        hanatos,
        sampleTx * static_cast<float>(hanatos.width - 1u),
        sampleTy * static_cast<float>(hanatos.height - 1u)
      );
      const size_t offset = (static_cast<size_t>(x) * hanatos.height + y) * 3u;
      remapped[offset] = sampled[0];
      remapped[offset + 1u] = sampled[1];
      remapped[offset + 2u] = sampled[2];
    }
  }
  return remapped;
}

std::vector<float> makeHanatosRawResponsePair(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity,
  const std::vector<float> &hanatosSpectra,
  const HanatosSpectraLutInfo &hanatos,
  RgbToRawMethod method
) {
  std::vector<float> response = makeHanatosRawResponse(filmCurves, linearSensitivity, hanatosSpectra, hanatos, method);
  std::vector<float> compressed = remapHanatosResponseForInputGamutCompression(filmCurves, response, hanatos);
  std::vector<float> packed;
  packed.reserve((response.size() + compressed.size()) / 3u * 4u);
  const auto appendPacked = [&](const std::vector<float> &source) {
    for (size_t offset = 0u; offset + 2u < source.size(); offset += 3u) {
      packed.push_back(source[offset]);
      packed.push_back(source[offset + 1u]);
      packed.push_back(source[offset + 2u]);
      packed.push_back(0.0f);
    }
  };
  appendPacked(response);
  appendPacked(compressed);
  return packed;
}

std::vector<float> makeHanatosResponsePairFromWeights(
  const ProfileCurveSet &referenceCurves,
  const std::vector<float> &spectralWeights,
  const std::vector<float> &hanatosSpectra,
  const HanatosSpectraLutInfo &hanatos
) {
  const size_t responseCount = static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) * 3u;
  std::vector<float> response(responseCount, 0.0f);
  const size_t expectedSpectra = static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) * static_cast<size_t>(hanatos.wavelengthCount);
  if (hanatos.width == 0u ||
      hanatos.height == 0u ||
      hanatos.wavelengthCount == 0u ||
      hanatosSpectra.size() < expectedSpectra ||
      spectralWeights.size() < static_cast<size_t>(hanatos.wavelengthCount) * 3u) {
    return response;
  }
  for (uint32_t x = 0; x < hanatos.width; ++x) {
    for (uint32_t y = 0; y < hanatos.height; ++y) {
      float raw[3] = {0.0f, 0.0f, 0.0f};
      const size_t spectraOffset = (static_cast<size_t>(x) * hanatos.height + y) * hanatos.wavelengthCount;
      for (uint32_t wavelength = 0u; wavelength < hanatos.wavelengthCount; ++wavelength) {
        const float spectrum = hanatosSpectra[spectraOffset + wavelength];
        const uint32_t weightOffset = wavelength * 3u;
        raw[0] += spectrum * spectralWeights[weightOffset];
        raw[1] += spectrum * spectralWeights[weightOffset + 1u];
        raw[2] += spectrum * spectralWeights[weightOffset + 2u];
      }
      const size_t responseOffset = (static_cast<size_t>(x) * hanatos.height + y) * 3u;
      response[responseOffset] = raw[0];
      response[responseOffset + 1u] = raw[1];
      response[responseOffset + 2u] = raw[2];
    }
  }
  std::vector<float> compressed = remapHanatosResponseForInputGamutCompression(referenceCurves, response, hanatos);
  std::vector<float> packed;
  packed.reserve((response.size() + compressed.size()) / 3u * 4u);
  const auto appendPacked = [&](const std::vector<float> &source) {
    for (size_t offset = 0u; offset + 2u < source.size(); offset += 3u) {
      packed.push_back(source[offset]);
      packed.push_back(source[offset + 1u]);
      packed.push_back(source[offset + 2u]);
      packed.push_back(0.0f);
    }
  };
  appendPacked(response);
  appendPacked(compressed);
  return packed;
}

float filteredEnlargerIlluminantCpu(
  uint32_t wavelength,
  const RenderParams &params,
  float cFilter,
  float mFilterShift,
  float yFilterShift
) {
  const uint32_t film = std::min<uint32_t>(
    static_cast<uint32_t>(std::max(params.film, 0)),
    std::max<uint32_t>(kSpektraFilmCount, 1u) - 1u
  );
  const uint32_t paper = std::min<uint32_t>(
    static_cast<uint32_t>(std::max(params.paper, 0)),
    std::max<uint32_t>(kSpektraPaperCount, 1u) - 1u
  );
  const uint32_t neutralOffset = (paper * kSpektraFilmCount + film) * 3u;
  const float *neutral = neutralPrintFilters();
  const float cc0 = std::max(neutral[neutralOffset] + cFilter, 0.0f);
  const float cc1 = std::max(neutral[neutralOffset + 1u] + mFilterShift, 0.0f);
  const float cc2 = std::max(neutral[neutralOffset + 2u] + yFilterShift, 0.0f);
  const float wheel0 = std::pow(10.0f, -cc0 / 100.0f);
  const float wheel1 = std::pow(10.0f, -cc1 / 100.0f);
  const float wheel2 = std::pow(10.0f, -cc2 / 100.0f);
  const uint32_t offset = wavelength * 3u;
  const float *filters = customEnlargerFilters();
  const float f0 = std::clamp(filters[offset], 0.0f, 1.0f);
  const float f1 = std::clamp(filters[offset + 1u], 0.0f, 1.0f);
  const float f2 = std::clamp(filters[offset + 2u], 0.0f, 1.0f);
  const float dim0 = 1.0f - (1.0f - f0) * (1.0f - wheel0);
  const float dim1 = 1.0f - (1.0f - f1) * (1.0f - wheel1);
  const float dim2 = 1.0f - (1.0f - f2) * (1.0f - wheel2);
  return thKg3Illuminant()[wavelength] * dim0 * dim1 * dim2;
}

std::vector<float> makeProcessNegativePaperWeights(
  const ProfileCurveSet &paperCurves,
  const std::vector<float> &paperSensitivityLinear,
  const RenderParams &params,
  bool preflash
) {
  std::vector<float> weights(static_cast<size_t>(paperCurves.wavelengthCount) * 3u, 0.0f);
  if (paperSensitivityLinear.size() < weights.size()) {
    return weights;
  }
  if (params.printTiming == PrintTimingMode::ApdPrinterDensity) {
    std::array<float, 3> normalization = {0.0f, 0.0f, 0.0f};
    for (uint32_t wavelength = 0u; wavelength < paperCurves.wavelengthCount; ++wavelength) {
      const uint32_t offset = wavelength * 3u;
      normalization[0] += std::max(academyPrinterDensityData()[offset], 0.0f);
      normalization[1] += std::max(academyPrinterDensityData()[offset + 1u], 0.0f);
      normalization[2] += std::max(academyPrinterDensityData()[offset + 2u], 0.0f);
    }
    const float preflashScale = preflash ? std::max(params.preflashExposure, 0.0f) : 1.0f;
    for (uint32_t wavelength = 0u; wavelength < paperCurves.wavelengthCount; ++wavelength) {
      const uint32_t offset = wavelength * 3u;
      weights[offset] = preflashScale * std::max(academyPrinterDensityData()[offset], 0.0f) / std::max(normalization[0], 1.0e-10f);
      weights[offset + 1u] = preflashScale * std::max(academyPrinterDensityData()[offset + 1u], 0.0f) / std::max(normalization[1], 1.0e-10f);
      weights[offset + 2u] = preflashScale * std::max(academyPrinterDensityData()[offset + 2u], 0.0f) / std::max(normalization[2], 1.0e-10f);
    }
    return weights;
  }
  const float cFilter = preflash ? 0.0f : params.filterC;
  const float mFilter = preflash ? params.preflashMFilterShift : params.filterMShift;
  const float yFilter = preflash ? params.preflashYFilterShift : params.filterYShift;
  const float preflashScale = preflash ? std::max(params.preflashExposure, 0.0f) : 1.0f;
  for (uint32_t wavelength = 0u; wavelength < paperCurves.wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    const float illuminant = filteredEnlargerIlluminantCpu(wavelength, params, cFilter, mFilter, yFilter);
    weights[offset] = preflashScale * illuminant * paperSensitivityLinear[offset];
    weights[offset + 1u] = preflashScale * illuminant * paperSensitivityLinear[offset + 1u];
    weights[offset + 2u] = preflashScale * illuminant * paperSensitivityLinear[offset + 2u];
  }
  return weights;
}

uint32_t findMemoryTypeIndex(
  VkPhysicalDevice physicalDevice,
  uint32_t typeBits,
  VkMemoryPropertyFlags requiredFlags,
  VkMemoryPropertyFlags preferredFlags = 0
) {
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
  uint32_t fallback = std::numeric_limits<uint32_t>::max();
  for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
    if (((typeBits & (1u << i)) != 0u) &&
        ((memoryProperties.memoryTypes[i].propertyFlags & requiredFlags) == requiredFlags)) {
      const VkMemoryPropertyFlags flags = memoryProperties.memoryTypes[i].propertyFlags;
      if ((flags & preferredFlags) == preferredFlags) {
        return i;
      }
      if (fallback == std::numeric_limits<uint32_t>::max()) {
        fallback = i;
      }
    }
  }
  return fallback;
}

int physicalDeviceTypeScore(VkPhysicalDeviceType type) {
  switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return 4000;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return 3000;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
      return 2000;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return 1000;
    default:
      return 0;
  }
}

bool chooseComputeQueueFamily(
  VkPhysicalDevice candidate,
  uint32_t &queueFamily,
  bool &dedicatedCompute
) {
  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, nullptr);
  std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount, queueFamilies.data());

  uint32_t fallback = std::numeric_limits<uint32_t>::max();
  for (uint32_t i = 0; i < queueFamilyCount; ++i) {
    const VkQueueFlags flags = queueFamilies[i].queueFlags;
    if (queueFamilies[i].queueCount == 0 || (flags & VK_QUEUE_COMPUTE_BIT) == 0u) {
      continue;
    }
    if ((flags & VK_QUEUE_GRAPHICS_BIT) == 0u) {
      queueFamily = i;
      dedicatedCompute = true;
      return true;
    }
    if (fallback == std::numeric_limits<uint32_t>::max()) {
      fallback = i;
    }
  }

  if (fallback != std::numeric_limits<uint32_t>::max()) {
    queueFamily = fallback;
    dedicatedCompute = false;
    return true;
  }
  return false;
}

const char *vkResultName(VkResult result) {
  switch (result) {
    case VK_SUCCESS: return "VK_SUCCESS";
    case VK_NOT_READY: return "VK_NOT_READY";
    case VK_TIMEOUT: return "VK_TIMEOUT";
    case VK_EVENT_SET: return "VK_EVENT_SET";
    case VK_EVENT_RESET: return "VK_EVENT_RESET";
    case VK_INCOMPLETE: return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY: return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY: return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED: return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST: return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED: return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT: return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT: return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT: return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER: return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS: return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED: return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_FRAGMENTED_POOL: return "VK_ERROR_FRAGMENTED_POOL";
    default: return "VK_ERROR_UNKNOWN";
  }
}

std::string vkError(const char *context, VkResult result) {
  return std::string(context) + " failed with " + vkResultName(result) + ".";
}

bool isDeviceLostError(const std::string &error) {
  return error.find("VK_ERROR_DEVICE_LOST") != std::string::npos;
}

GpuRenderTilingMode resolveVulkanTileMode(const RenderParams &params) {
  const std::string mode = envString("SPEKTRAFILM_VULKAN_TILE_MODE", "");
  if (mode == "legacy" || mode == "full-frame" || mode == "fullframe") {
    return GpuRenderTilingMode::LegacyFullFrame;
  }
  if (mode == "tiled" || mode == "tile") {
    return GpuRenderTilingMode::Tiled;
  }
  return params.gpuRenderTiling;
}

uint32_t envTileDimension(const char *name, uint32_t fallback) {
  const std::string text = envString(name, "");
  if (text.empty()) {
    return fallback;
  }
  char *end = nullptr;
  const long value = std::strtol(text.c_str(), &end, 10);
  if (!end || *end != '\0' || value < 64 || value > 8192) {
    return fallback;
  }
  return static_cast<uint32_t>(value);
}

constexpr uint32_t kVulkanGrainSpatialRadiusPx = 64u;
constexpr uint32_t kVulkanSpatialEffectRadiusPx = 256u;
constexpr uint32_t kDefaultVulkanTileWidth = 512u;
constexpr uint32_t kDefaultVulkanTileHeight = 256u;

uint32_t estimateVulkanTileOverlap(const RenderParams &params) {
  uint32_t overlap = 0u;
  const bool finalOutput = params.renderOutput == RenderOutputMode::FinalPreview;
  const bool rcmOutput = finalOutput && params.outputRole == OutputRole::Rcm;
  const bool finalPrintSimulation =
    finalOutput && params.process == ProcessMode::PrintSimulation;
  const bool finalProcessNegative =
    finalOutput && params.process == ProcessMode::ProcessNegative;
  const bool finalPostProcessPath =
    finalOutput &&
    (params.process == ProcessMode::ScanNegative ||
     params.process == ProcessMode::ProcessNegative ||
     finalPrintSimulation);

  if (params.cameraDiffusionEnabled &&
      params.process != ProcessMode::ProcessNegative &&
      params.cameraDiffusionStrength > 0.0f &&
      params.cameraDiffusionSpatialScale > 0.0f) {
    overlap += kVulkanSpatialEffectRadiusPx;
  }
  if (params.halationEnabled &&
      params.process != ProcessMode::ProcessNegative &&
      ((params.scatterAmount > 0.0f && params.scatterScale > 0.0f) ||
       (params.halationAmount > 0.0f &&
        params.halationScale > 0.0f &&
        (params.halationStrengthR > 0.0f ||
         params.halationStrengthG > 0.0f ||
         params.halationStrengthB > 0.0f)))) {
    overlap += kVulkanSpatialEffectRadiusPx;
  }
  if (params.dirCouplersAmount > 0.0f &&
      params.process != ProcessMode::ProcessNegative &&
      (params.dirCouplersDiffusionUm > 0.0f ||
       (params.dirCouplersDiffusionTailUm > 0.0f && params.dirCouplersDiffusionTailWeight > 0.0f))) {
    overlap += kVulkanSpatialEffectRadiusPx;
  }
  if (params.grainEnabled &&
      params.process != ProcessMode::ProcessNegative &&
      (params.grainModel == GrainModel::Production ||
       params.grainModel == GrainModel::GrainSynthesis)) {
    overlap += kVulkanGrainSpatialRadiusPx;
  }
  if ((finalPrintSimulation || finalProcessNegative) &&
      params.printDiffusionEnabled &&
      params.printDiffusionStrength > 0.0f &&
      params.printDiffusionSpatialScale > 0.0f) {
    overlap += kVulkanSpatialEffectRadiusPx;
  }
  if (finalPostProcessPath &&
      !rcmOutput &&
      params.scannerEnabled &&
      ((params.glarePercent > 0.0f && params.glareBlur > 0.0f) ||
       params.scannerMtf50LpMm > 0.0f ||
       (params.scannerUnsharpRadiusUm > 0.0f && params.scannerUnsharpAmount > 0.0f))) {
    overlap += kVulkanSpatialEffectRadiusPx;
  }
  return overlap;
}

uint32_t alignedReducedDimension(
  uint32_t localSize,
  uint32_t tileOrigin,
  uint32_t fullSize,
  uint32_t scale
) {
  if (localSize == 0u || fullSize == 0u) {
    return 0u;
  }
  scale = std::max(scale, 1u);
  const uint64_t start = static_cast<uint64_t>(tileOrigin) / scale;
  const uint64_t localEnd =
    std::min<uint64_t>(static_cast<uint64_t>(tileOrigin) + localSize, fullSize);
  const uint64_t end = (localEnd + scale - 1u) / scale;
  return static_cast<uint32_t>(std::max<uint64_t>(end > start ? end - start : 0u, 1u));
}

struct VulkanCorePushConstants {
  uint32_t width = 0;
  uint32_t height = 0;
  float filmExposureEv = 0.0f;
  float filmGamma = 1.0f;
  uint32_t exposureCount = 0;
  int32_t inputColorSpace = 0;
  int32_t rgbToRawMethod = 0;
  uint32_t colorSpaceCount = 0;
  uint32_t transferLutSize = 0;
  float colorDecodeMin = 0.0f;
  float colorDecodeMax = 1.0f;
  uint32_t hanatosWidth = 0;
  uint32_t hanatosHeight = 0;
  uint32_t _pad0 = 0;
  uint32_t _pad1 = 0;
  uint32_t _pad2 = 0;
  int32_t filmPushPullMode = 0;
  float filmPushPullStops = 0.0f;
  uint32_t fullWidth = 0;
  uint32_t fullHeight = 0;
  uint32_t tileOriginX = 0;
  uint32_t tileOriginY = 0;
  uint32_t activeOriginX = 0;
  uint32_t activeOriginY = 0;
  uint32_t activeWidth = 0;
  uint32_t activeHeight = 0;
};

static_assert(sizeof(VulkanCorePushConstants) == 104u);

struct VulkanDiffusionInfo {
  uint32_t componentCount = 0;
  float scatterFraction = 0.0f;
  uint32_t _pad0 = 0;
  uint32_t _pad1 = 0;
};

struct VulkanDiffusionComponent {
  float sigmaPx = 0.0f;
  float weightR = 0.0f;
  float weightG = 0.0f;
  float weightB = 0.0f;
};

static_assert(sizeof(VulkanDiffusionInfo) == 16u);
static_assert(sizeof(VulkanDiffusionComponent) == 16u);

constexpr uint32_t kCoreFrameFloatCount = 105u;
constexpr uint32_t kCoreFrameIntCount = 29u;
constexpr uint32_t kCoreHalationSetCount = 13u;
constexpr uint32_t kCoreDiffusionSetCount = 3u;
constexpr uint32_t kCorePrintDiffusionSetCount = 2u;
constexpr uint32_t kCoreDirSetCount = 7u;
constexpr uint32_t kCoreScannerPostSetCount = 9u;
constexpr uint32_t kCoreGrainSetCount = 5u;
constexpr uint32_t kCoreDescriptorSetCount =
  3u + kCoreHalationSetCount + kCoreDiffusionSetCount + kCorePrintDiffusionSetCount + kCoreDirSetCount +
  kCoreScannerPostSetCount + kCoreGrainSetCount;
constexpr uint32_t kMaxVulkanDiffusionComponents = 32u;
constexpr uint32_t kDirFloatCount = 18u;
enum : uint32_t {
  kHalationOpClear = 0u,
  kHalationOpBlurX = 1u,
  kHalationOpBlurYStore = 2u,
  kHalationOpBlurYAccumulate = 3u,
  kHalationOpScatterResolve = 4u,
  kHalationOpBounceResolveLog = 5u,
  kHalationOpRawToLog = 6u,
  kHalationOpBoostMax = 7u,
  kHalationOpBoostReduceMax = 8u,
  kHalationOpBoostApply = 9u,
};

constexpr uint32_t kHalationBoostMaxChunkPixels = 256u;

enum : uint32_t {
  kHalationSigmaScatterCore = 0u,
  kHalationSigmaScatterTail = 1u,
  kHalationSigmaBounce = 2u,
};

enum : uint32_t {
  kDiffusionOpClear = 0u,
  kDiffusionOpBlurX = 1u,
  kDiffusionOpBlurYAccumulate = 2u,
  kDiffusionOpResolve = 3u,
  kDiffusionOpRawToLog = 4u,
  kDiffusionOpGroupBlurX = 5u,
  kDiffusionOpGroupBlurYAccumulate = 6u,
  kDiffusionOpDownsample = 7u,
  kDiffusionOpDownsampleBlurX = 8u,
  kDiffusionOpDownsampleBlurY = 9u,
  kDiffusionOpDownsampleUpsampleAccumulate = 10u,
  kDiffusionOpDownsampleGroupBlurX = 11u,
  kDiffusionOpDownsampleGroupBlurY = 12u,
  kDiffusionOpDownsampleGroupUpsampleAccumulate = 13u,
};

enum : uint32_t {
  kPrintScanOpFull = 0u,
  kPrintScanOpPrintRaw = 1u,
  kPrintScanOpFinalFromPrintRaw = 2u,
  kPrintScanOpFrameConstants = 3u,
  kPrintScanOpPrintRawFromNegativeLight = 4u,
};

enum : uint32_t {
  kDirOpCorrectionFromDensity = 0u,
  kDirOpBlurX = 1u,
  kDirOpBlurYStore = 2u,
  kDirOpTailClear = 3u,
  kDirOpTailBlurX = 4u,
  kDirOpTailBlurYAccumulate = 5u,
  kDirOpRedevelop = 6u,
};

enum : uint32_t {
  kScannerPostOpPrintGlareGenerate = 0u,
  kScannerPostOpPrintGlareBlurX = 1u,
  kScannerPostOpPrintGlareBlurY = 2u,
  kScannerPostOpPrintGlareApply = 3u,
  kScannerPostOpScannerBlurX = 4u,
  kScannerPostOpScannerBlurY = 5u,
  kScannerPostOpUnsharpBlurX = 6u,
  kScannerPostOpUnsharpBlurY = 7u,
  kScannerPostOpFinalize = 8u,
};

enum : uint32_t {
  kGrainOpPreview = 0u,
  kGrainOpProductionLayers = 1u,
  kGrainOpLayerBlurX = 2u,
  kGrainOpLayerBlurY = 3u,
  kGrainOpMicroSource = 4u,
  kGrainOpMicroBlurX = 5u,
  kGrainOpMicroBlurY = 6u,
  kGrainOpResolveDensity = 7u,
  kGrainOpDensityBlurX = 8u,
  kGrainOpDensityBlurY = 9u,
  kGrainOpApplyControls = 10u,
  kGrainOpSynthesisLayers = 11u,
  kGrainOpSynthesisResolveDensity = 12u,
  kGrainOpCopyDensity = 13u,
};

struct DiffusionGroup {
  float lambdaUm = 0.0f;
  float spread = 1.0f;
  uint32_t count = 1u;
  float alpha = 3.0f;
};

struct DiffusionFamilyShape {
  DiffusionGroup core;
  DiffusionGroup halo;
  DiffusionGroup bloom;
  float weightCore = 0.0f;
  float weightHalo = 0.0f;
  float weightBloom = 0.0f;
  float warmthBase = 0.0f;
  float totalGain = 1.0f;
};

struct DiffusionSettings {
  DiffusionFilterFamily family = DiffusionFilterFamily::BlackProMist;
  float strength = 0.5f;
  float spatialScale = 1.0f;
  float haloWarmth = 0.0f;
  float coreIntensity = 1.0f;
  float coreSize = 1.0f;
  float haloIntensity = 1.0f;
  float haloSize = 1.0f;
  float bloomIntensity = 1.0f;
  float bloomSize = 1.0f;
};

struct VulkanSharedBackend {
  struct QueueSlot {
    VkQueue queue = VK_NULL_HANDLE;
    std::mutex submitMutex;
  };

  uint32_t generation = 0;
  std::string lastError;
  bool available = false;
  std::atomic<bool> poisoned{false};

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties physicalDeviceProperties{};
  VkDevice device = VK_NULL_HANDLE;
  uint32_t computeQueueFamily = 0;
  std::vector<std::unique_ptr<QueueSlot>> queueSlots;
  std::atomic<uint32_t> nextQueueSlot{0};

  VkShaderModule copyShaderModule = VK_NULL_HANDLE;
  VkShaderModule formatConvertShaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout copyPipelineLayout = VK_NULL_HANDLE;
  VkPipeline copyPipeline = VK_NULL_HANDLE;
  VkPipeline formatConvertPipeline = VK_NULL_HANDLE;
  VkShaderModule filmExposureShaderModule = VK_NULL_HANDLE;
  VkShaderModule curveDevelopShaderModule = VK_NULL_HANDLE;
  VkShaderModule printScanShaderModule = VK_NULL_HANDLE;
  VkShaderModule halationShaderModule = VK_NULL_HANDLE;
  VkShaderModule diffusionShaderModule = VK_NULL_HANDLE;
  VkShaderModule dirShaderModule = VK_NULL_HANDLE;
  VkShaderModule scannerPostShaderModule = VK_NULL_HANDLE;
  VkShaderModule grainShaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout coreDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout corePipelineLayout = VK_NULL_HANDLE;
  VkPipeline filmExposurePipeline = VK_NULL_HANDLE;
  VkPipeline curveDevelopPipeline = VK_NULL_HANDLE;
  VkPipeline printScanPipeline = VK_NULL_HANDLE;
  VkPipeline halationPipeline = VK_NULL_HANDLE;
  VkPipeline diffusionPipeline = VK_NULL_HANDLE;
  VkPipeline dirPipeline = VK_NULL_HANDLE;
  VkPipeline scannerPostPipeline = VK_NULL_HANDLE;
  VkPipeline grainPipeline = VK_NULL_HANDLE;
  uint64_t transientBudgetBytes = 0;

  explicit VulkanSharedBackend(uint32_t backendGeneration) : generation(backendGeneration) {
    initialize();
  }

  ~VulkanSharedBackend() {
    cleanup();
  }

  bool isAvailable() const {
    return available && !poisoned.load();
  }

  uint32_t queueCount() const {
    return static_cast<uint32_t>(queueSlots.size());
  }

  uint32_t assignQueueIndex() {
    const uint32_t count = queueCount();
    if (count == 0u) {
      return 0u;
    }
    return nextQueueSlot.fetch_add(1u) % count;
  }

  VkQueue queueForIndex(uint32_t queueIndex) const {
    if (queueSlots.empty()) {
      return VK_NULL_HANDLE;
    }
    return queueSlots[static_cast<size_t>(queueIndex % queueCount())]->queue;
  }

  VkResult submit(uint32_t queueIndex, const VkSubmitInfo &submitInfo, VkFence fence) {
    if (queueSlots.empty() || poisoned.load()) {
      return poisoned.load() ? VK_ERROR_DEVICE_LOST : VK_ERROR_INITIALIZATION_FAILED;
    }
    QueueSlot &slot = *queueSlots[static_cast<size_t>(queueIndex % queueCount())];
    std::lock_guard<std::mutex> lock(slot.submitMutex);
    return vkQueueSubmit(slot.queue, 1, &submitInfo, fence);
  }

  void markLost() {
    poisoned.store(true);
  }

private:
  static uint32_t envPositiveUint(const char *name, uint32_t fallback) {
    const char *text = std::getenv(name);
    if (!text || text[0] == '\0') {
      return fallback;
    }
    char *end = nullptr;
    const unsigned long value = std::strtoul(text, &end, 10);
    if (!end || *end != '\0' || value == 0ul || value > std::numeric_limits<uint32_t>::max()) {
      return fallback;
    }
    return static_cast<uint32_t>(value);
  }

  static uint64_t envPositiveMb(const char *name, uint64_t fallbackBytes) {
    const char *text = std::getenv(name);
    if (!text || text[0] == '\0') {
      return fallbackBytes;
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (!end || *end != '\0' || value == 0ull ||
        value > (std::numeric_limits<uint64_t>::max() / (1024ull * 1024ull))) {
      return fallbackBytes;
    }
    return static_cast<uint64_t>(value) * 1024ull * 1024ull;
  }

  void initialize() {
    lastError.clear();

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "SpektraFilm OFX";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "SpektraFilm";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &appInfo;
    VkResult result = vkCreateInstance(&instanceInfo, nullptr, &instance);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateInstance", result);
      return;
    }

    if (!selectPhysicalDevice() || !createLogicalDevice() || !createCopyPipeline() || !createCorePipelines()) {
      cleanup();
      return;
    }

    transientBudgetBytes = computeTransientBudgetBytes();
    available = true;
  }

  bool selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    VkResult result = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (result != VK_SUCCESS || deviceCount == 0) {
      lastError = result == VK_SUCCESS ? "No Vulkan physical device is available." : vkError("vkEnumeratePhysicalDevices", result);
      return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    result = vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    if (result != VK_SUCCESS) {
      lastError = vkError("vkEnumeratePhysicalDevices", result);
      return false;
    }

    if (const char *requestedIndexText = std::getenv("SPEKTRAFILM_VULKAN_DEVICE_INDEX");
        requestedIndexText && requestedIndexText[0]) {
      char *end = nullptr;
      const unsigned long requestedIndex = std::strtoul(requestedIndexText, &end, 10);
      if (!end || *end != '\0' || requestedIndex >= deviceCount) {
        lastError = "SPEKTRAFILM_VULKAN_DEVICE_INDEX does not identify an available Vulkan physical device.";
        return false;
      }

      uint32_t queueFamily = 0;
      bool dedicatedCompute = false;
      VkPhysicalDevice candidate = devices[static_cast<size_t>(requestedIndex)];
      if (!chooseComputeQueueFamily(candidate, queueFamily, dedicatedCompute)) {
        lastError = "The requested Vulkan physical device does not expose a compute-capable queue family.";
        return false;
      }
      physicalDevice = candidate;
      computeQueueFamily = queueFamily;
      vkGetPhysicalDeviceProperties(physicalDevice, &physicalDeviceProperties);
      return true;
    }

    int bestScore = std::numeric_limits<int>::min();
    for (VkPhysicalDevice candidate : devices) {
      uint32_t queueFamily = 0;
      bool dedicatedCompute = false;
      if (!chooseComputeQueueFamily(candidate, queueFamily, dedicatedCompute)) {
        continue;
      }

      VkPhysicalDeviceProperties properties{};
      vkGetPhysicalDeviceProperties(candidate, &properties);
      const int score = physicalDeviceTypeScore(properties.deviceType) + (dedicatedCompute ? 100 : 0);
      if (score > bestScore) {
        bestScore = score;
        physicalDevice = candidate;
        physicalDeviceProperties = properties;
        computeQueueFamily = queueFamily;
      }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
      lastError = "No Vulkan compute-capable queue family is available.";
      return false;
    }
    return true;
  }

  bool createLogicalDevice() {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    if (computeQueueFamily >= queueFamilyCount || queueFamilies[computeQueueFamily].queueCount == 0u) {
      lastError = "The selected Vulkan queue family is not available.";
      return false;
    }

    const uint32_t requestedQueueCount = envPositiveUint("SPEKTRAFILM_VULKAN_QUEUE_COUNT", 1u);
    const uint32_t queueCount = std::max(1u, std::min(requestedQueueCount, queueFamilies[computeQueueFamily].queueCount));
    std::vector<float> priorities(queueCount, 1.0f);

    VkDeviceQueueCreateInfo queueInfo{};
    queueInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueInfo.queueFamilyIndex = computeQueueFamily;
    queueInfo.queueCount = queueCount;
    queueInfo.pQueuePriorities = priorities.data();

    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queueInfo;

    VkResult result = vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDevice", result);
      return false;
    }

    queueSlots.reserve(queueCount);
    for (uint32_t queueIndex = 0; queueIndex < queueCount; ++queueIndex) {
      auto slot = std::make_unique<QueueSlot>();
      vkGetDeviceQueue(device, computeQueueFamily, queueIndex, &slot->queue);
      queueSlots.push_back(std::move(slot));
    }
    return true;
  }

  bool loadShaderModule(const char *relativePath, VkShaderModule &shaderModule) {
    const std::filesystem::path shaderPath = findResourcePath(relativePath);
    if (shaderPath.empty()) {
      lastError = std::string("Unable to locate Vulkan shader resource ") + relativePath + ".";
      return false;
    }

    std::string spirvError;
    const std::vector<uint32_t> spirv = readSpirvFile(shaderPath, spirvError);
    if (spirv.empty()) {
      lastError = spirvError;
      return false;
    }

    VkShaderModuleCreateInfo shaderInfo{};
    shaderInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderInfo.pCode = spirv.data();
    const VkResult result = vkCreateShaderModule(device, &shaderInfo, nullptr, &shaderModule);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateShaderModule", result);
      return false;
    }
    return true;
  }

  bool createComputePipeline(VkShaderModule shaderModule, VkPipelineLayout layout, VkPipeline &pipeline) {
    VkPipelineShaderStageCreateInfo stageInfo{};
    stageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stageInfo.module = shaderModule;
    stageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = stageInfo;
    pipelineInfo.layout = layout;
    const VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateComputePipelines", result);
      return false;
    }
    return true;
  }

  bool createCopyPipeline() {
    if (!loadShaderModule("shaders/SpektraCopy.comp.spv", copyShaderModule) ||
        !loadShaderModule("shaders/SpektraFormatConvert.comp.spv", formatConvertShaderModule)) {
      return false;
    }

    VkDescriptorSetLayoutBinding bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorInfo{};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorInfo.bindingCount = static_cast<uint32_t>(sizeof(bindings) / sizeof(bindings[0]));
    descriptorInfo.pBindings = bindings;
    VkResult result = vkCreateDescriptorSetLayout(device, &descriptorInfo, nullptr, &copyDescriptorSetLayout);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDescriptorSetLayout", result);
      return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(uint32_t) * 2u;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &copyDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &copyPipelineLayout);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreatePipelineLayout", result);
      return false;
    }

    return createComputePipeline(copyShaderModule, copyPipelineLayout, copyPipeline) &&
           createComputePipeline(formatConvertShaderModule, copyPipelineLayout, formatConvertPipeline);
  }

  bool createCorePipelines() {
    if (!loadShaderModule("shaders/SpektraFilmExposure.comp.spv", filmExposureShaderModule) ||
        !loadShaderModule("shaders/SpektraCurveDevelop.comp.spv", curveDevelopShaderModule) ||
        !loadShaderModule("shaders/SpektraPrintScan.comp.spv", printScanShaderModule) ||
        !loadShaderModule("shaders/SpektraHalation.comp.spv", halationShaderModule) ||
        !loadShaderModule("shaders/SpektraDiffusion.comp.spv", diffusionShaderModule) ||
        !loadShaderModule("shaders/SpektraDir.comp.spv", dirShaderModule) ||
        !loadShaderModule("shaders/SpektraScannerPost.comp.spv", scannerPostShaderModule) ||
        !loadShaderModule("shaders/SpektraGrain.comp.spv", grainShaderModule)) {
      return false;
    }

    VkDescriptorSetLayoutBinding bindings[31]{};
    for (uint32_t bindingIndex = 0; bindingIndex < 31u; ++bindingIndex) {
      bindings[bindingIndex].binding = bindingIndex;
      bindings[bindingIndex].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[bindingIndex].descriptorCount = 1;
      bindings[bindingIndex].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptorInfo{};
    descriptorInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorInfo.bindingCount = static_cast<uint32_t>(sizeof(bindings) / sizeof(bindings[0]));
    descriptorInfo.pBindings = bindings;
    VkResult result = vkCreateDescriptorSetLayout(device, &descriptorInfo, nullptr, &coreDescriptorSetLayout);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDescriptorSetLayout", result);
      return false;
    }

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(VulkanCorePushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &coreDescriptorSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    result = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &corePipelineLayout);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreatePipelineLayout", result);
      return false;
    }

    return createComputePipeline(filmExposureShaderModule, corePipelineLayout, filmExposurePipeline) &&
           createComputePipeline(curveDevelopShaderModule, corePipelineLayout, curveDevelopPipeline) &&
           createComputePipeline(printScanShaderModule, corePipelineLayout, printScanPipeline) &&
           createComputePipeline(halationShaderModule, corePipelineLayout, halationPipeline) &&
           createComputePipeline(diffusionShaderModule, corePipelineLayout, diffusionPipeline) &&
           createComputePipeline(dirShaderModule, corePipelineLayout, dirPipeline) &&
           createComputePipeline(scannerPostShaderModule, corePipelineLayout, scannerPostPipeline) &&
           createComputePipeline(grainShaderModule, corePipelineLayout, grainPipeline);
  }

  uint64_t computeTransientBudgetBytes() const {
    constexpr uint64_t kMb = 1024ull * 1024ull;
    constexpr uint64_t kMinBudget = 512ull * kMb;
    constexpr uint64_t kMaxBudget = 8192ull * kMb;

    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    uint64_t largestDeviceLocalHeap = 0;
    uint64_t largestHeap = 0;
    for (uint32_t heapIndex = 0; heapIndex < memoryProperties.memoryHeapCount; ++heapIndex) {
      const uint64_t heapSize = static_cast<uint64_t>(memoryProperties.memoryHeaps[heapIndex].size);
      largestHeap = std::max(largestHeap, heapSize);
      if ((memoryProperties.memoryHeaps[heapIndex].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0u) {
        largestDeviceLocalHeap = std::max(largestDeviceLocalHeap, heapSize);
      }
    }

    const uint64_t baseHeap = largestDeviceLocalHeap != 0 ? largestDeviceLocalHeap : largestHeap;
    const uint64_t autoBudget = std::clamp(baseHeap / 2u, kMinBudget, kMaxBudget);
    return envPositiveMb("SPEKTRAFILM_VULKAN_TRANSIENT_BUDGET_MB", autoBudget);
  }

  void cleanup() {
    available = false;
    if (device != VK_NULL_HANDLE) {
      if (!poisoned.load()) {
        vkDeviceWaitIdle(device);
      }
      if (grainPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, grainPipeline, nullptr);
        grainPipeline = VK_NULL_HANDLE;
      }
      if (scannerPostPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, scannerPostPipeline, nullptr);
        scannerPostPipeline = VK_NULL_HANDLE;
      }
      if (dirPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, dirPipeline, nullptr);
        dirPipeline = VK_NULL_HANDLE;
      }
      if (diffusionPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, diffusionPipeline, nullptr);
        diffusionPipeline = VK_NULL_HANDLE;
      }
      if (halationPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, halationPipeline, nullptr);
        halationPipeline = VK_NULL_HANDLE;
      }
      if (printScanPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, printScanPipeline, nullptr);
        printScanPipeline = VK_NULL_HANDLE;
      }
      if (curveDevelopPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, curveDevelopPipeline, nullptr);
        curveDevelopPipeline = VK_NULL_HANDLE;
      }
      if (filmExposurePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, filmExposurePipeline, nullptr);
        filmExposurePipeline = VK_NULL_HANDLE;
      }
      if (grainShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, grainShaderModule, nullptr);
        grainShaderModule = VK_NULL_HANDLE;
      }
      if (corePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, corePipelineLayout, nullptr);
        corePipelineLayout = VK_NULL_HANDLE;
      }
      if (coreDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, coreDescriptorSetLayout, nullptr);
        coreDescriptorSetLayout = VK_NULL_HANDLE;
      }
      if (curveDevelopShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, curveDevelopShaderModule, nullptr);
        curveDevelopShaderModule = VK_NULL_HANDLE;
      }
      if (filmExposureShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, filmExposureShaderModule, nullptr);
        filmExposureShaderModule = VK_NULL_HANDLE;
      }
      if (printScanShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, printScanShaderModule, nullptr);
        printScanShaderModule = VK_NULL_HANDLE;
      }
      if (halationShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, halationShaderModule, nullptr);
        halationShaderModule = VK_NULL_HANDLE;
      }
      if (diffusionShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, diffusionShaderModule, nullptr);
        diffusionShaderModule = VK_NULL_HANDLE;
      }
      if (dirShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, dirShaderModule, nullptr);
        dirShaderModule = VK_NULL_HANDLE;
      }
      if (scannerPostShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, scannerPostShaderModule, nullptr);
        scannerPostShaderModule = VK_NULL_HANDLE;
      }
      if (copyPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, copyPipeline, nullptr);
        copyPipeline = VK_NULL_HANDLE;
      }
      if (formatConvertPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(device, formatConvertPipeline, nullptr);
        formatConvertPipeline = VK_NULL_HANDLE;
      }
      if (copyPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(device, copyPipelineLayout, nullptr);
        copyPipelineLayout = VK_NULL_HANDLE;
      }
      if (copyDescriptorSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(device, copyDescriptorSetLayout, nullptr);
        copyDescriptorSetLayout = VK_NULL_HANDLE;
      }
      if (copyShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, copyShaderModule, nullptr);
        copyShaderModule = VK_NULL_HANDLE;
      }
      if (formatConvertShaderModule != VK_NULL_HANDLE) {
        vkDestroyShaderModule(device, formatConvertShaderModule, nullptr);
        formatConvertShaderModule = VK_NULL_HANDLE;
      }
      queueSlots.clear();
      vkDestroyDevice(device, nullptr);
      device = VK_NULL_HANDLE;
    }
    if (instance != VK_NULL_HANDLE) {
      vkDestroyInstance(instance, nullptr);
      instance = VK_NULL_HANDLE;
    }
  }
};

std::mutex gVulkanSharedBackendMutex;
std::weak_ptr<VulkanSharedBackend> gVulkanSharedBackend;
std::atomic<uint32_t> gVulkanSharedBackendGeneration{0};

std::shared_ptr<VulkanSharedBackend> acquireVulkanSharedBackend(std::string &error) {
  std::lock_guard<std::mutex> lock(gVulkanSharedBackendMutex);
  if (std::shared_ptr<VulkanSharedBackend> existing = gVulkanSharedBackend.lock()) {
    if (existing->isAvailable()) {
      return existing;
    }
    gVulkanSharedBackend.reset();
  }

  auto backend = std::make_shared<VulkanSharedBackend>(gVulkanSharedBackendGeneration.fetch_add(1u) + 1u);
  if (!backend->isAvailable()) {
    error = backend->lastError.empty() ? "Unable to initialize shared Vulkan backend." : backend->lastError;
    return nullptr;
  }
  gVulkanSharedBackend = backend;
  return backend;
}

void clearVulkanSharedBackendIfSame(const std::shared_ptr<VulkanSharedBackend> &backend) {
  if (!backend) {
    return;
  }
  std::lock_guard<std::mutex> lock(gVulkanSharedBackendMutex);
  if (gVulkanSharedBackend.lock() == backend) {
    gVulkanSharedBackend.reset();
  }
}

float filmFormatLongEdgeMm(FilmFormat format) {
  switch (format) {
    case FilmFormat::Standard8:
      return 4.8f;
    case FilmFormat::Super8:
      return 5.79f;
    case FilmFormat::Standard16:
      return 10.26f;
    case FilmFormat::Super16:
      return 12.52f;
    case FilmFormat::Super35:
      return 24.89f;
    case FilmFormat::Standard65:
      return 52.48f;
    case FilmFormat::Imax70:
      return 70.41f;
    case FilmFormat::Standard35:
    default:
      return 35.0f;
  }
}

float resolvedEnlargerScale(const RenderParams &params) {
  return std::clamp(params.enlargerScale, 1.0f, 32.0f);
}

float filmPushPullGamma(float stops) {
  const float clampedStops = std::clamp(stops, -2.0f, 2.0f);
  constexpr float kNormalEcn2Seconds = 180.0f;
  constexpr float kPull1Seconds = 150.0f;
  constexpr float kPush1Seconds = 220.0f;
  constexpr float kPush2Seconds = 280.0f;
  if (clampedStops < 0.0f) {
    return std::pow(kPull1Seconds / kNormalEcn2Seconds, -clampedStops);
  }
  if (clampedStops <= 1.0f) {
    return std::exp(std::log(kPush1Seconds / kNormalEcn2Seconds) * clampedStops);
  }
  const float segment = clampedStops - 1.0f;
  return std::exp(
    std::log(kPush1Seconds / kNormalEcn2Seconds) +
    std::log(kPush2Seconds / kPush1Seconds) * segment
  );
}

float printPushPullGamma(float stops) {
  const float clampedStops = std::clamp(stops, -2.0f, 2.0f);
  return std::exp2(clampedStops * 0.25f);
}

float scannerSigmaUmFromMtf50(float mtf50LpMm) {
  if (!std::isfinite(mtf50LpMm) || mtf50LpMm <= 0.0f) {
    return 0.0f;
  }
  constexpr float kPi = 3.14159265358979323846f;
  return 1000.0f * std::sqrt(std::log(2.0f) / (2.0f * kPi * kPi)) / mtf50LpMm;
}

float sampleTransferLut(float value, int32_t colorSpace, const float *luts) {
  if (!luts || kSpektraColorTransferLutSize <= 1u) {
    return value;
  }
  const float decodeMin = colorDecodeLutMin();
  const float decodeMax = colorDecodeLutMax();
  const float range = std::max(decodeMax - decodeMin, 1.0e-6f);
  const float step = range / static_cast<float>(kSpektraColorTransferLutSize - 1u);
  const size_t offset = static_cast<size_t>(colorSpace) * static_cast<size_t>(kSpektraColorTransferLutSize);
  if (value <= decodeMin) {
    const float y0 = luts[offset];
    const float y1 = luts[offset + 1u];
    return y0 + (value - decodeMin) * ((y1 - y0) / std::max(step, 1.0e-12f));
  }
  if (value >= decodeMax) {
    const float y0 = luts[offset + kSpektraColorTransferLutSize - 2u];
    const float y1 = luts[offset + kSpektraColorTransferLutSize - 1u];
    return y1 + (value - decodeMax) * ((y1 - y0) / std::max(step, 1.0e-12f));
  }
  const float t = (value - decodeMin) / range;
  const float position = t * static_cast<float>(kSpektraColorTransferLutSize - 1u);
  const uint32_t lo = static_cast<uint32_t>(std::floor(position));
  const uint32_t hi = std::min(lo + 1u, kSpektraColorTransferLutSize - 1u);
  const float f = position - static_cast<float>(lo);
  return luts[offset + lo] + (luts[offset + hi] - luts[offset + lo]) * f;
}

float autoExposureMeterY(float r, float g, float b, const RenderParams &params) {
  const int32_t colorSpace = std::clamp(
    static_cast<int32_t>(params.inputColorSpace),
    0,
    static_cast<int32_t>(kSpektraColorSpaceCount - 1u)
  );
  const uint32_t *transferKinds = colorTransferKinds();
  if (transferKinds && transferKinds[colorSpace] != 0u) {
    const float *decodeLuts = colorDecodeLuts();
    r = sampleTransferLut(r, colorSpace, decodeLuts);
    g = sampleTransferLut(g, colorSpace, decodeLuts);
    b = sampleTransferLut(b, colorSpace, decodeLuts);
  }
  const float *meterMatrices = inputMeterXyzMatrices();
  if (!meterMatrices) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }
  const float *matrix = meterMatrices + static_cast<size_t>(colorSpace) * 9u;
  return matrix[3] * r + matrix[4] * g + matrix[5] * b;
}

void autoExposurePreviewShape(int32_t width, int32_t height, int32_t &previewWidth, int32_t &previewHeight) {
  constexpr int32_t kPreviewMaxSize = 256;
  const int32_t longEdge = std::max(width, height);
  if (longEdge <= kPreviewMaxSize) {
    previewWidth = width;
    previewHeight = height;
    return;
  }
  const double scale = static_cast<double>(kPreviewMaxSize) / static_cast<double>(longEdge);
  previewWidth = std::max(1, static_cast<int32_t>(std::lround(static_cast<double>(width) * scale)));
  previewHeight = std::max(1, static_cast<int32_t>(std::lround(static_cast<double>(height) * scale)));
}

float measureAutoExposureEv(
  const float *source,
  int32_t width,
  int32_t height,
  const RenderParams &params
) {
  if (!source || width <= 0 || height <= 0) {
    return 0.0f;
  }

  int32_t previewWidth = width;
  int32_t previewHeight = height;
  autoExposurePreviewShape(width, height, previewWidth, previewHeight);
  std::vector<float> luminance;
  luminance.reserve(static_cast<size_t>(previewWidth) * static_cast<size_t>(previewHeight));

  for (int32_t y = 0; y < previewHeight; ++y) {
    const int32_t sourceY = std::min(
      height - 1,
      static_cast<int32_t>((static_cast<int64_t>(y) * height) / previewHeight)
    );
    for (int32_t x = 0; x < previewWidth; ++x) {
      const int32_t sourceX = std::min(
        width - 1,
        static_cast<int32_t>((static_cast<int64_t>(x) * width) / previewWidth)
      );
      const float *pixel =
        source + (static_cast<size_t>(sourceY) * static_cast<size_t>(width) + static_cast<size_t>(sourceX)) * 4u;
      luminance.push_back(autoExposureMeterY(pixel[0], pixel[1], pixel[2], params));
    }
  }
  if (luminance.empty()) {
    return 0.0f;
  }

  double meteredY = 0.0;
  if (params.autoExposureMethod == AutoExposureMethod::Median) {
    const size_t mid = luminance.size() / 2u;
    std::nth_element(luminance.begin(), luminance.begin() + static_cast<std::ptrdiff_t>(mid), luminance.end());
    meteredY = luminance[mid];
    if ((luminance.size() & 1u) == 0u) {
      const float upper = luminance[mid];
      std::nth_element(
        luminance.begin(),
        luminance.begin() + static_cast<std::ptrdiff_t>(mid - 1u),
        luminance.begin() + static_cast<std::ptrdiff_t>(mid)
      );
      meteredY = 0.5 * (static_cast<double>(luminance[mid - 1u]) + static_cast<double>(upper));
    }
  } else {
    const double normX = static_cast<double>(previewWidth) / static_cast<double>(std::max(previewWidth, previewHeight));
    const double normY = static_cast<double>(previewHeight) / static_cast<double>(std::max(previewWidth, previewHeight));
    constexpr double kSigma = 0.2;
    double weightedSum = 0.0;
    double weightSum = 0.0;
    size_t index = 0;
    for (int32_t y = 0; y < previewHeight; ++y) {
      const double yf = (static_cast<double>(y) / static_cast<double>(previewHeight) - 0.5) * normY;
      for (int32_t x = 0; x < previewWidth; ++x, ++index) {
        const double xf = (static_cast<double>(x) / static_cast<double>(previewWidth) - 0.5) * normX;
        const double weight = std::exp(-(xf * xf + yf * yf) / (2.0 * kSigma * kSigma));
        weightedSum += static_cast<double>(luminance[index]) * weight;
        weightSum += weight;
      }
    }
    meteredY = weightedSum / std::max(weightSum, 1.0e-30);
  }

  const double exposure = meteredY / 0.184;
  if (!(exposure > 0.0) || !std::isfinite(exposure)) {
    return 0.0f;
  }
  const double ev = -std::log2(exposure);
  return std::isfinite(ev) ? static_cast<float>(ev) : 0.0f;
}

bool isDefaultHalationStrength(const RenderParams &params) {
  return std::abs(params.halationStrengthR - 0.05f) <= 1.0e-6f &&
         std::abs(params.halationStrengthG - 0.015f) <= 1.0e-6f &&
         std::abs(params.halationStrengthB) <= 1.0e-6f;
}

DiffusionFamilyShape diffusionShape(DiffusionFilterFamily family) {
  switch (family) {
    case DiffusionFilterFamily::Glimmerglass:
      return {{10.0f, 1.5f, 2u, 3.0f}, {50.0f, 2.0f, 3u, 3.0f}, {260.0f, 2.5f, 4u, 3.2f}, 0.60f, 0.30f, 0.10f, 0.0f, 0.65f};
    case DiffusionFilterFamily::ProMist:
      return {{14.0f, 1.5f, 2u, 3.0f}, {150.0f, 2.0f, 3u, 3.0f}, {650.0f, 2.5f, 4u, 2.9f}, 0.28f, 0.42f, 0.30f, 0.40f, 1.05f};
    case DiffusionFilterFamily::CineBloom:
      return {{20.0f, 1.5f, 2u, 3.0f}, {200.0f, 2.0f, 3u, 3.0f}, {1000.0f, 2.5f, 4u, 2.5f}, 0.22f, 0.30f, 0.48f, 0.85f, 1.00f};
    case DiffusionFilterFamily::BlackProMist:
    default:
      return {{16.0f, 1.5f, 2u, 3.0f}, {95.0f, 2.0f, 3u, 3.0f}, {380.0f, 2.5f, 4u, 3.5f}, 0.40f, 0.47f, 0.13f, 0.65f, 0.75f};
  }
}

float diffusionScatterFraction(float strength, float familyGain) {
  if (strength <= 0.0f) {
    return 0.0f;
  }
  constexpr std::array<float, 5> kBreaks = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f};
  constexpr std::array<float, 5> kFractions = {0.10f, 0.20f, 0.35f, 0.55f, 0.75f};
  const float logStrength = std::log2(std::max(strength, 1.0e-6f));
  if (logStrength <= std::log2(kBreaks.front())) {
    return std::clamp(kFractions.front() * familyGain, 0.0f, 0.99f);
  }
  if (logStrength >= std::log2(kBreaks.back())) {
    return std::clamp(kFractions.back() * familyGain, 0.0f, 0.99f);
  }
  for (size_t i = 0; i + 1u < kBreaks.size(); ++i) {
    const float x0 = std::log2(kBreaks[i]);
    const float x1 = std::log2(kBreaks[i + 1u]);
    if (logStrength >= x0 && logStrength <= x1) {
      const float t = std::clamp((logStrength - x0) / std::max(x1 - x0, 1.0e-6f), 0.0f, 1.0f);
      return std::clamp((kFractions[i] + (kFractions[i + 1u] - kFractions[i]) * t) * familyGain, 0.0f, 0.99f);
    }
  }
  return 0.0f;
}

std::vector<float> diffusionLambdas(const DiffusionGroup &group) {
  std::vector<float> lambdas(group.count, group.lambdaUm);
  if (group.count <= 1u || group.spread <= 1.0f) {
    return lambdas;
  }
  const float logLo = std::log(group.lambdaUm / group.spread);
  const float logHi = std::log(group.lambdaUm * group.spread);
  for (uint32_t i = 0; i < group.count; ++i) {
    const float t = group.count == 1u ? 0.0f : static_cast<float>(i) / static_cast<float>(group.count - 1u);
    lambdas[i] = std::exp(logLo + (logHi - logLo) * t);
  }
  return lambdas;
}

std::vector<float> diffusionWeights(const DiffusionGroup &group, bool bloom) {
  const std::vector<float> lambdas = diffusionLambdas(group);
  std::vector<float> weights(lambdas.size(), 1.0f);
  if (bloom) {
    for (size_t i = 0; i < weights.size(); ++i) {
      weights[i] = std::pow(std::max(lambdas[i], 1.0e-6f), 2.0f - group.alpha);
    }
  }
  float sum = 0.0f;
  for (float weight : weights) {
    sum += weight;
  }
  for (float &weight : weights) {
    weight /= std::max(sum, 1.0e-6f);
  }
  return weights;
}

std::array<std::vector<float>, 3> haloChannelWeights(const std::vector<float> &weights, float warmth) {
  constexpr std::array<float, 3> kWarmthAxis = {1.30f, 0.15f, -1.45f};
  std::array<std::vector<float>, 3> out;
  const size_t count = weights.size();
  for (auto &channel : out) {
    channel = weights;
  }
  if (count < 2u) {
    return out;
  }
  warmth = std::clamp(warmth, -1.5f, 1.5f);
  std::vector<float> gradient(count, 0.0f);
  float totalWeight = 0.0f;
  float weightedGradient = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    gradient[i] = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(count - 1u);
    totalWeight += weights[i];
    weightedGradient += weights[i] * gradient[i];
  }
  const float gradientMean = weightedGradient / std::max(totalWeight, 1.0e-6f);
  for (float &value : gradient) {
    value -= gradientMean;
  }
  for (size_t channel = 0; channel < 3u; ++channel) {
    float sum = 0.0f;
    for (size_t i = 0; i < count; ++i) {
      out[channel][i] = std::max(weights[i] * (1.0f + warmth * kWarmthAxis[channel] * gradient[i]), 0.0f);
      sum += out[channel][i];
    }
    for (size_t i = 0; i < count; ++i) {
      out[channel][i] *= totalWeight / std::max(sum, 1.0e-6f);
    }
  }
  return out;
}

void appendDiffusionGroupComponents(
  std::vector<VulkanDiffusionComponent> &components,
  const DiffusionGroup &group,
  const std::vector<float> &weights,
  const std::array<float, 3> &channelScale,
  float groupWeight,
  float spatialScale,
  float pixelSizeUm
) {
  constexpr std::array<std::array<float, 2>, 3> kExpGaussianFit = {{
    {0.1633f, 0.5360f},
    {0.6496f, 1.5236f},
    {0.1870f, 2.7684f},
  }};
  const std::vector<float> lambdas = diffusionLambdas(group);
  for (size_t i = 0; i < lambdas.size(); ++i) {
    for (const auto &fit : kExpGaussianFit) {
      const float sigmaPx = std::max(lambdas[i] * fit[1] * spatialScale / std::max(pixelSizeUm, 1.0e-6f), 1.0e-6f);
      const float weight = groupWeight * weights[i] * fit[0];
      const float weightR = weight * channelScale[0];
      const float weightG = weight * channelScale[1];
      const float weightB = weight * channelScale[2];
      if (weightR == 0.0f && weightG == 0.0f && weightB == 0.0f) {
        continue;
      }
      components.push_back({sigmaPx, weightR, weightG, weightB});
    }
  }
}

void clusterDiffusionComponents(std::vector<VulkanDiffusionComponent> &components, float sigmaRatio) {
  if (components.size() < 2u || sigmaRatio <= 0.0f) {
    return;
  }
  std::sort(
    components.begin(),
    components.end(),
    [](const VulkanDiffusionComponent &a, const VulkanDiffusionComponent &b) {
      return a.sigmaPx < b.sigmaPx;
    }
  );
  std::vector<VulkanDiffusionComponent> clustered;
  clustered.reserve(components.size());
  for (const VulkanDiffusionComponent &component : components) {
    if (clustered.empty()) {
      clustered.push_back(component);
      continue;
    }
    VulkanDiffusionComponent &last = clustered.back();
    const float denom = std::max(std::max(last.sigmaPx, component.sigmaPx), 1.0e-6f);
    if (std::abs(component.sigmaPx - last.sigmaPx) / denom <= sigmaRatio) {
      const float lastWeight = std::abs(last.weightR) + std::abs(last.weightG) + std::abs(last.weightB);
      const float componentWeight = std::abs(component.weightR) + std::abs(component.weightG) + std::abs(component.weightB);
      const float totalWeight = lastWeight + componentWeight;
      if (totalWeight > 1.0e-8f) {
        last.sigmaPx = (last.sigmaPx * lastWeight + component.sigmaPx * componentWeight) / totalWeight;
      }
      last.weightR += component.weightR;
      last.weightG += component.weightG;
      last.weightB += component.weightB;
    } else {
      clustered.push_back(component);
    }
  }
  components.swap(clustered);
}

uint32_t diffusionDownsampleScaleForSigma(const std::string &mode, float sigmaPx) {
  if (mode == "off") {
    return 1u;
  }
  if (mode == "auto") {
    if (sigmaPx >= 48.0f) {
      return 8u;
    }
    if (sigmaPx >= 24.0f) {
      return 4u;
    }
    if (sigmaPx >= 12.0f) {
      return 2u;
    }
    return 1u;
  }
  if (mode == "8") {
    return sigmaPx >= 48.0f ? 8u : 1u;
  }
  if (mode == "4") {
    return sigmaPx >= 24.0f ? 4u : 1u;
  }
  if (mode == "2") {
    return sigmaPx >= 12.0f ? 2u : 1u;
  }
  return 1u;
}

bool anyDiffusionComponentDownsamples(
  const std::vector<VulkanDiffusionComponent> &components,
  const std::string &mode
) {
  for (const VulkanDiffusionComponent &component : components) {
    if (diffusionDownsampleScaleForSigma(mode, component.sigmaPx) > 1u) {
      return true;
    }
  }
  return false;
}

std::vector<VulkanDiffusionComponent> makeDiffusionComponents(
  const DiffusionSettings &settings,
  float pixelSizeUm,
  VulkanDiffusionInfo &info,
  float clusterSigmaRatio = 0.0f
) {
  DiffusionFamilyShape shape = diffusionShape(settings.family);
  const float coreIntensity = std::max(settings.coreIntensity, 0.0f);
  const float haloIntensity = std::max(settings.haloIntensity, 0.0f);
  const float bloomIntensity = std::max(settings.bloomIntensity, 0.0f);
  float wc = shape.weightCore * coreIntensity;
  float wh = shape.weightHalo * haloIntensity;
  float wb = shape.weightBloom * bloomIntensity;
  const float total = wc + wh + wb;
  if (total > 0.0f) {
    wc /= total;
    wh /= total;
    wb /= total;
  } else {
    info = {};
    return {};
  }
  shape.core.lambdaUm *= std::max(settings.coreSize, 1.0e-6f);
  shape.halo.lambdaUm *= std::max(settings.haloSize, 1.0e-6f);
  shape.bloom.lambdaUm *= std::max(settings.bloomSize, 1.0e-6f);

  info.scatterFraction = diffusionScatterFraction(settings.strength, shape.totalGain);
  std::vector<VulkanDiffusionComponent> components;
  if (info.scatterFraction <= 0.0f || settings.spatialScale <= 0.0f) {
    info.componentCount = 0u;
    return components;
  }

  appendDiffusionGroupComponents(components, shape.core, diffusionWeights(shape.core, false),
    {1.0f, 1.0f, 1.0f}, wc, settings.spatialScale, pixelSizeUm);

  const std::vector<float> haloWeights = diffusionWeights(shape.halo, false);
  const auto haloPerChannel = haloChannelWeights(haloWeights, shape.warmthBase + settings.haloWarmth);
  const std::vector<float> haloLambdas = diffusionLambdas(shape.halo);
  constexpr std::array<std::array<float, 2>, 3> kExpGaussianFit = {{
    {0.1633f, 0.5360f},
    {0.6496f, 1.5236f},
    {0.1870f, 2.7684f},
  }};
  for (size_t i = 0; i < haloLambdas.size(); ++i) {
    for (const auto &fit : kExpGaussianFit) {
      const float sigmaPx = std::max(haloLambdas[i] * fit[1] * settings.spatialScale / std::max(pixelSizeUm, 1.0e-6f), 1.0e-6f);
      const float weightR = wh * haloPerChannel[0][i] * fit[0];
      const float weightG = wh * haloPerChannel[1][i] * fit[0];
      const float weightB = wh * haloPerChannel[2][i] * fit[0];
      if (weightR == 0.0f && weightG == 0.0f && weightB == 0.0f) {
        continue;
      }
      components.push_back({sigmaPx, weightR, weightG, weightB});
    }
  }

  appendDiffusionGroupComponents(components, shape.bloom, diffusionWeights(shape.bloom, true),
    {1.0f, 1.0f, 1.0f}, wb, settings.spatialScale, pixelSizeUm);

  clusterDiffusionComponents(components, clusterSigmaRatio);
  if (components.size() > kMaxVulkanDiffusionComponents) {
    components.resize(kMaxVulkanDiffusionComponents);
  }
  info.componentCount = static_cast<uint32_t>(components.size());
  return components;
}

float interpLinearDensityCurve(
  const std::vector<float> &x,
  const float *y,
  uint32_t channel,
  float target
) {
  if (x.empty() || !y) {
    return 0.0f;
  }
  if (x.size() == 1u) {
    return y[channel];
  }
  const bool ascending = x.back() >= x.front();
  if ((ascending && target <= x.front()) || (!ascending && target >= x.front())) {
    return y[channel];
  }
  if ((ascending && target >= x.back()) || (!ascending && target <= x.back())) {
    return y[(x.size() - 1u) * 3u + channel];
  }

  uint32_t lo = 0u;
  uint32_t hi = static_cast<uint32_t>(x.size() - 1u);
  while (hi - lo > 1u) {
    const uint32_t mid = (lo + hi) >> 1u;
    if ((ascending && x[mid] <= target) || (!ascending && x[mid] >= target)) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const float denom = std::max(std::abs(x[hi] - x[lo]), 1.0e-9f);
  const float t = std::clamp((target - x[lo]) / denom, 0.0f, 1.0f);
  const float y0 = y[lo * 3u + channel];
  const float y1 = y[hi * 3u + channel];
  return y0 + (y1 - y0) * t;
}

std::array<float, kDirFloatCount> makeDirFloatParams(
  const ProfileCurveSet &filmCurves,
  const RenderParams &params,
  float filmPixelSizeUm
) {
  const float amount = std::max(params.dirCouplersAmount, 0.0f);
  const float sameLayer = std::max(params.dirCouplersInhibitionSameLayer, 0.0f);
  const float interlayer = std::max(params.dirCouplersInhibitionInterlayer, 0.0f);
  const std::array<float, 3> densityMaximums = densityCurveMaximums(filmCurves);
  std::array<float, kDirFloatCount> out{};
  out[0] = params.dirCouplersGammaSameLayerR * sameLayer * amount;
  out[4] = params.dirCouplersGammaSameLayerG * sameLayer * amount;
  out[8] = params.dirCouplersGammaSameLayerB * sameLayer * amount;
  out[1] = params.dirCouplersGammaRToG * interlayer * amount;
  out[2] = params.dirCouplersGammaRToB * interlayer * amount;
  out[3] = params.dirCouplersGammaGToR * interlayer * amount;
  out[5] = params.dirCouplersGammaGToB * interlayer * amount;
  out[6] = params.dirCouplersGammaBToR * interlayer * amount;
  out[7] = params.dirCouplersGammaBToG * interlayer * amount;
  out[9] = densityMaximums[0];
  out[10] = densityMaximums[1];
  out[11] = densityMaximums[2];
  out[12] = std::max(params.dirCouplersDiffusionUm, 0.0f) / std::max(filmPixelSizeUm, 1.0e-6f);
  out[13] = std::max(params.dirCouplersDiffusionTailUm, 0.0f) * 0.5360f / std::max(filmPixelSizeUm, 1.0e-6f);
  out[14] = std::max(params.dirCouplersDiffusionTailUm, 0.0f) * 1.5236f / std::max(filmPixelSizeUm, 1.0e-6f);
  out[15] = std::max(params.dirCouplersDiffusionTailUm, 0.0f) * 2.7684f / std::max(filmPixelSizeUm, 1.0e-6f);
  out[16] = std::clamp(params.dirCouplersDiffusionTailWeight, 0.0f, 1.0f);
  out[17] = filmCurves.type && std::strcmp(filmCurves.type, "positive") == 0 ? 1.0f : 0.0f;
  return out;
}

std::vector<float> makeDirCorrectedDensityCurves(
  const ProfileCurveSet &filmCurves,
  const std::array<float, kDirFloatCount> &dirFloats
) {
  const bool positive = dirFloats[17] > 0.5f;
  std::vector<float> corrected(static_cast<size_t>(filmCurves.exposureCount) * 3u, 0.0f);
  for (uint32_t receiver = 0; receiver < 3u; ++receiver) {
    std::vector<float> logExposure0(filmCurves.exposureCount, 0.0f);
    for (uint32_t i = 0; i < filmCurves.exposureCount; ++i) {
      const float d0 = filmCurves.densityCurves[i * 3u];
      const float d1 = filmCurves.densityCurves[i * 3u + 1u];
      const float d2 = filmCurves.densityCurves[i * 3u + 2u];
      const float silver0 = positive ? dirFloats[9] - d0 : d0;
      const float silver1 = positive ? dirFloats[10] - d1 : d1;
      const float silver2 = positive ? dirFloats[11] - d2 : d2;
      float amount = 0.0f;
      if (receiver == 0u) {
        amount = silver0 * dirFloats[0] + silver1 * dirFloats[3] + silver2 * dirFloats[6];
      } else if (receiver == 1u) {
        amount = silver0 * dirFloats[1] + silver1 * dirFloats[4] + silver2 * dirFloats[7];
      } else {
        amount = silver0 * dirFloats[2] + silver1 * dirFloats[5] + silver2 * dirFloats[8];
      }
      logExposure0[i] = filmCurves.logExposure[i] - amount;
    }
    for (uint32_t i = 0; i < filmCurves.exposureCount; ++i) {
      corrected[i * 3u + receiver] = interpLinearDensityCurve(
        logExposure0,
        filmCurves.densityCurves,
        receiver,
        filmCurves.logExposure[i]
      );
    }
  }
  return corrected;
}

std::array<float, 3> scanIlluminantToOutputRgb(
  const ProfileCurveSet *paperCurves,
  const RenderParams &params
) {
  if (!paperCurves || paperCurves->wavelengthCount == 0u || !paperCurves->scanIlluminant ||
      !paperCurves->scanToOutputRgb || !standardObserverCmfs()) {
    return {1.0f, 1.0f, 1.0f};
  }

  std::array<float, 3> xyz = {0.0f, 0.0f, 0.0f};
  float normalization = 0.0f;
  const float *cmfs = standardObserverCmfs();
  for (uint32_t wavelength = 0; wavelength < paperCurves->wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    const float illuminant = paperCurves->scanIlluminant[wavelength];
    xyz[0] += illuminant * cmfs[offset];
    xyz[1] += illuminant * cmfs[offset + 1u];
    xyz[2] += illuminant * cmfs[offset + 2u];
    normalization += illuminant * cmfs[offset + 1u];
  }

  const float invNormalization = 1.0f / std::max(normalization, 1.0e-10f);
  xyz[0] *= invNormalization;
  xyz[1] *= invNormalization;
  xyz[2] *= invNormalization;

  const uint32_t colorSpace = std::min<uint32_t>(
    static_cast<uint32_t>(std::max(static_cast<int32_t>(params.outputColorSpace), 0)),
    kSpektraColorSpaceCount - 1u
  );
  const float *matrix = paperCurves->scanToOutputRgb + static_cast<size_t>(colorSpace) * 9u;
  return {
    matrix[0] * xyz[0] + matrix[1] * xyz[1] + matrix[2] * xyz[2],
    matrix[3] * xyz[0] + matrix[4] * xyz[1] + matrix[5] * xyz[2],
    matrix[6] * xyz[0] + matrix[7] * xyz[1] + matrix[8] * xyz[2],
  };
}

} // namespace

struct VulkanRenderer::Impl {
  struct ScratchBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
    VkDeviceSize allocationSize = 0;
    VkMemoryPropertyFlags memoryFlags = 0;
    void *mapped = nullptr;
  };

  struct CopyFrameResources {
    ScratchBuffer source;
    ScratchBuffer destination;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
  };

  struct CoreFrameResources {
    ScratchBuffer sourceStaging;
    ScratchBuffer source;
    ScratchBuffer filmRaw;
    ScratchBuffer filmDensity;
    ScratchBuffer destination;
    ScratchBuffer destinationHalf;
    ScratchBuffer destinationStaging;
    ScratchBuffer destinationHalfStaging;
    ScratchBuffer halationRawA;
    ScratchBuffer halationRawB;
    ScratchBuffer halationRawC;
    ScratchBuffer halationRawD;
    ScratchBuffer halationBoostedRaw;
    ScratchBuffer halationBoostChunks;
    ScratchBuffer halationBoostInfo;
    ScratchBuffer halationBoostInfoReadback;
    ScratchBuffer tiledHalationBoostInfo;
    ScratchBuffer halationLogRaw;
    ScratchBuffer diffusionTemp;
    ScratchBuffer diffusionAccum;
    ScratchBuffer diffusionDownsampleSource;
    ScratchBuffer diffusionDownsampleTemp;
    ScratchBuffer diffusionDownsampleBlur;
    ScratchBuffer cameraDiffusionRaw;
    ScratchBuffer printRaw;
    ScratchBuffer printDiffusionRaw;
    ScratchBuffer dirCorrectionA;
    ScratchBuffer dirCorrectionB;
    ScratchBuffer dirCorrectionC;
    ScratchBuffer dirDensity;
    ScratchBuffer cameraDiffusionInfo;
    ScratchBuffer cameraDiffusionComponents;
    ScratchBuffer printDiffusionInfo;
    ScratchBuffer printDiffusionComponents;
    ScratchBuffer dirFloats;
    ScratchBuffer dirCorrectedDensityCurves;
    ScratchBuffer scannerPostA;
    ScratchBuffer scannerPostB;
    ScratchBuffer scannerPostC;
    ScratchBuffer printGlareA;
    ScratchBuffer printGlareB;
    ScratchBuffer grainDensityA;
    ScratchBuffer grainDensityB;
    ScratchBuffer grainMicroA;
    ScratchBuffer grainMicroB;
    ScratchBuffer grainLayerA;
    ScratchBuffer grainLayerB;
    ScratchBuffer frameFloats;
    ScratchBuffer frameInts;
    ScratchBuffer filteredEnlargerResponse;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet exposureDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet developDescriptorSet = VK_NULL_HANDLE;
    VkDescriptorSet finalDescriptorSet = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, kCoreHalationSetCount> halationDescriptorSets{};
    std::array<VkDescriptorSet, kCoreDiffusionSetCount> diffusionDescriptorSets{};
    std::array<VkDescriptorSet, kCorePrintDiffusionSetCount> printDiffusionDescriptorSets{};
    std::array<VkDescriptorSet, kCoreDirSetCount> dirDescriptorSets{};
    std::array<VkDescriptorSet, kCoreScannerPostSetCount> scannerPostDescriptorSets{};
    std::array<VkDescriptorSet, kCoreGrainSetCount> grainDescriptorSets{};
    VkDescriptorPool formatDescriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet destinationFormatDescriptorSet = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
  };

  struct StaticFilmResources {
    int32_t film = -1;
    int32_t paper = -1;
    RgbToRawMethod rgbToRawMethod = RgbToRawMethod::Hanatos2026;
    bool cameraUvFilterEnabled = false;
    float cameraUvCutNm = 0.0f;
    bool cameraIrFilterEnabled = false;
    float cameraIrCutNm = 0.0f;
    PrintTimingMode processNegativePrintTiming = PrintTimingMode::FilteredEnlarger;
    float processNegativeFilterC = 0.0f;
    float processNegativeFilterMShift = 0.0f;
    float processNegativeFilterYShift = 0.0f;
    float processNegativePreflashMFilterShift = 0.0f;
    float processNegativePreflashYFilterShift = 0.0f;
    bool printScanResources = false;
    bool grainResources = false;
    const ProfileCurveSet *curves = nullptr;
    const ProfileCurveSet *paperCurves = nullptr;
    uint32_t exposureCount = 0;
    uint32_t paperExposureCount = 0;
    uint32_t wavelengthCount = 0;
    uint32_t filmPositive = 0;
    uint32_t hanatosWidth = 0;
    uint32_t hanatosHeight = 0;
    std::array<float, 3> filmDensityMaximum = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> paperDensityMaximum = {0.0f, 0.0f, 0.0f};
    ScratchBuffer logExposure;
    ScratchBuffer densityCurves;
    ScratchBuffer densityCurveLayers;
    ScratchBuffer densityCurveLayerMaxima;
    ScratchBuffer inputToReferenceXyz;
    ScratchBuffer inputToSrgb;
    ScratchBuffer colorDecodeLuts;
    ScratchBuffer colorTransferKinds;
    ScratchBuffer mallettRawMatrix;
    ScratchBuffer hanatosRawResponse;
    ScratchBuffer paperHanatosResponse;
    ScratchBuffer preflashPaperHanatosResponse;
    ScratchBuffer paperLogExposure;
    ScratchBuffer paperDensityCurves;
    ScratchBuffer filmChannelDensity;
    ScratchBuffer filmBaseDensity;
    ScratchBuffer paperLogSensitivity;
    ScratchBuffer thKg3Illuminant;
    ScratchBuffer customEnlargerFilters;
    ScratchBuffer neutralPrintFilters;
    ScratchBuffer paperChannelDensity;
    ScratchBuffer paperBaseDensity;
    ScratchBuffer filmScanIlluminant;
    ScratchBuffer paperScanIlluminant;
    ScratchBuffer standardObserverCmfs;
    ScratchBuffer filmScanToOutputRgb;
    ScratchBuffer paperScanToOutputRgb;
    ScratchBuffer colorEncodeLuts;
    ScratchBuffer academyPrinterDensityData;
  };

  RendererDiagnostics diagnostics;
  std::string lastError;
  std::vector<float> outputGamutCompressionData;
  bool available = false;
  bool preferPrivateScratch = true;
  bool grainBlurRecurrence = true;
  uint32_t diffusionGroupSize = 2u;
  float diffusionClusterSigmaRatio = 0.10f;
  std::string threadgroupMode = "auto";
  std::string blurBackend = "custom";
  std::string blurDownsample = "auto";
  std::string intermediatePrecision = "float";
  std::string diffusionClusterSigma = "0.10";
  std::string dirTailBackend = "mps";
  bool halationGroupedTail = false;
  bool scannerMps = false;
  std::recursive_mutex renderMutex;
  std::shared_ptr<VulkanSharedBackend> backend;
  uint32_t queueIndex = 0;
  uint64_t transientCachedBytes = 0;
  uint64_t lastUseSerial = 0;

  struct ActiveTileContext {
    bool enabled = false;
    uint32_t fullWidth = 0;
    uint32_t fullHeight = 0;
    uint32_t tileOriginX = 0;
    uint32_t tileOriginY = 0;
    uint32_t centerOriginX = 0;
    uint32_t centerOriginY = 0;
    uint32_t centerWidth = 0;
    uint32_t centerHeight = 0;
    bool halationBoostMilestoneEnabled = false;
    bool fullFrameSource = false;
  };
  ActiveTileContext activeTileContext;

  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkPhysicalDeviceProperties physicalDeviceProperties{};
  VkDevice device = VK_NULL_HANDLE;
  VkQueue computeQueue = VK_NULL_HANDLE;
  uint32_t computeQueueFamily = 0;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkShaderModule copyShaderModule = VK_NULL_HANDLE;
  VkShaderModule formatConvertShaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout copyDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout copyPipelineLayout = VK_NULL_HANDLE;
  VkPipeline copyPipeline = VK_NULL_HANDLE;
  VkPipeline formatConvertPipeline = VK_NULL_HANDLE;
  CopyFrameResources copyFrame;
  VkShaderModule filmExposureShaderModule = VK_NULL_HANDLE;
  VkShaderModule curveDevelopShaderModule = VK_NULL_HANDLE;
  VkShaderModule printScanShaderModule = VK_NULL_HANDLE;
  VkShaderModule halationShaderModule = VK_NULL_HANDLE;
  VkShaderModule diffusionShaderModule = VK_NULL_HANDLE;
  VkShaderModule dirShaderModule = VK_NULL_HANDLE;
  VkShaderModule scannerPostShaderModule = VK_NULL_HANDLE;
  VkShaderModule grainShaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout coreDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout corePipelineLayout = VK_NULL_HANDLE;
  VkPipeline filmExposurePipeline = VK_NULL_HANDLE;
  VkPipeline curveDevelopPipeline = VK_NULL_HANDLE;
  VkPipeline printScanPipeline = VK_NULL_HANDLE;
  VkPipeline halationPipeline = VK_NULL_HANDLE;
  VkPipeline diffusionPipeline = VK_NULL_HANDLE;
  VkPipeline dirPipeline = VK_NULL_HANDLE;
  VkPipeline scannerPostPipeline = VK_NULL_HANDLE;
  VkPipeline grainPipeline = VK_NULL_HANDLE;
  CoreFrameResources coreFrame;
  StaticFilmResources staticFilm;
  std::vector<float> hanatosSpectraData;
  static std::mutex transientRegistryMutex;
  static std::vector<Impl *> transientRegistry;
  static uint64_t transientUseClock;

  Impl() {
    preferPrivateScratch = envStringAny(
      {"SPEKTRAFILM_VULKAN_SCRATCH_STORAGE", "SPEKTRAFILM_SCRATCH_STORAGE"},
      "private"
    ) != "shared";
    grainBlurRecurrence = envFlagEnabledOrDefaultAny(
      {"SPEKTRAFILM_VULKAN_GRAIN_BLUR_RECURRENCE", "SPEKTRAFILM_GRAIN_BLUR_RECURRENCE"},
      true
    );
    const std::string diffusionGroupSizeText = envStringAny(
      {"SPEKTRAFILM_VULKAN_DIFFUSION_GROUP_SIZE", "SPEKTRAFILM_DIFFUSION_GROUP_SIZE"},
      "2"
    );
    if (diffusionGroupSizeText == "1") {
      diffusionGroupSize = 1u;
    } else if (diffusionGroupSizeText == "2") {
      diffusionGroupSize = 2u;
    } else {
      diffusionGroupSize = 4u;
    }
    blurBackend = envStringAny(
      {"SPEKTRAFILM_VULKAN_BLUR_BACKEND", "SPEKTRAFILM_BLUR_BACKEND"},
      "custom"
    );
    if (blurBackend != "custom" && blurBackend != "mps" && blurBackend != "auto") {
      blurBackend = "custom";
    }
    blurDownsample = envStringAny(
      {"SPEKTRAFILM_VULKAN_BLUR_DOWNSAMPLE", "SPEKTRAFILM_BLUR_DOWNSAMPLE"},
      "auto"
    );
    if (blurDownsample != "off" && blurDownsample != "2" && blurDownsample != "4" &&
        blurDownsample != "8" && blurDownsample != "auto") {
      blurDownsample = "off";
    }
    intermediatePrecision = envStringAny(
      {"SPEKTRAFILM_VULKAN_INTERMEDIATE_PRECISION", "SPEKTRAFILM_INTERMEDIATE_PRECISION"},
      "float"
    );
    if (intermediatePrecision != "float") {
      intermediatePrecision = "float";
    }
    diffusionClusterSigma = envStringAny(
      {"SPEKTRAFILM_VULKAN_DIFFUSION_CLUSTER_SIGMA", "SPEKTRAFILM_DIFFUSION_CLUSTER_SIGMA"},
      "0.10"
    );
    if (diffusionClusterSigma == "0.05") {
      diffusionClusterSigmaRatio = 0.05f;
    } else if (diffusionClusterSigma == "0.10" || diffusionClusterSigma == "0.1") {
      diffusionClusterSigma = "0.10";
      diffusionClusterSigmaRatio = 0.10f;
    } else {
      diffusionClusterSigma = "off";
      diffusionClusterSigmaRatio = 0.0f;
    }
    dirTailBackend = envStringAny(
      {"SPEKTRAFILM_VULKAN_DIR_TAIL_BACKEND", "SPEKTRAFILM_DIR_TAIL_BACKEND"},
      "mps"
    ) == "mps" ? "mps" : "fused";
    threadgroupMode = envStringAny(
      {"SPEKTRAFILM_VULKAN_THREADGROUP", "SPEKTRAFILM_THREADGROUP"},
      "auto"
    );
    if (threadgroupMode != "auto" && threadgroupMode != "32x8") {
      threadgroupMode = "auto";
    }
    halationGroupedTail = envFlagEnabledOrDefaultAny(
      {"SPEKTRAFILM_VULKAN_HALATION_GROUPED_TAIL", "SPEKTRAFILM_HALATION_GROUPED_TAIL"},
      false
    );
    scannerMps = envFlagEnabledOrDefaultAny(
      {"SPEKTRAFILM_VULKAN_SCANNER_MPS", "SPEKTRAFILM_SCANNER_MPS"},
      false
    );
    registerTransientBudgetEntry();
    initialize();
  }

  ~Impl() {
    cleanup();
    unregisterTransientBudgetEntry();
  }

  bool renderCopyValidation(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    double time
  );

  bool renderCoreBootstrap(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  );
  bool computeHalationBoostMilestone(
    const ImageView &source,
    const RenderWindow &window,
    const RenderParams &params,
    uint32_t centerTileWidth,
    uint32_t centerTileHeight
  );

  bool renderTiledBootstrap(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  );

  bool createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memoryFlags,
    VkMemoryPropertyFlags preferredMemoryFlags,
    VkBuffer &buffer,
    VkDeviceMemory &memory,
    VkDeviceSize *allocatedBytes = nullptr,
    VkMemoryPropertyFlags *actualMemoryFlags = nullptr
  );

  bool ensureScratchBuffer(
    ScratchBuffer &scratch,
    VkDeviceSize size,
    const char *name,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags memoryFlags,
    VkMemoryPropertyFlags preferredMemoryFlags,
    bool persistentMap,
    bool countPrivate
  );
  bool flushMappedScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name);
  bool invalidateMappedScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name);
  bool ensureSharedScratchBuffer(
    ScratchBuffer &scratch,
    VkDeviceSize size,
    const char *name,
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
  );
  bool ensureUploadScratchBuffer(
    ScratchBuffer &scratch,
    VkDeviceSize size,
    const char *name,
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
  );
  bool ensureReadbackScratchBuffer(
    ScratchBuffer &scratch,
    VkDeviceSize size,
    const char *name,
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
  );
  bool ensurePrivateScratchBuffer(
    ScratchBuffer &scratch,
    VkDeviceSize size,
    const char *name,
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
  );
  bool ensureScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name);
  bool uploadScratchBuffer(ScratchBuffer &scratch, const void *data, VkDeviceSize size, const char *name);
  bool uploadStaticBuffer(ScratchBuffer &scratch, const void *data, VkDeviceSize size, const char *name);
  bool copyBufferImmediate(VkBuffer source, VkBuffer destination, VkDeviceSize size, const char *name);
  bool loadHanatosSpectraData();
  bool loadOutputGamutCompressionData();
  bool prepareStaticFilmResources(const RenderParams &params, bool includePrintScanResources, bool includeGrainResources);
  bool ensureCopyFrameResources();
  bool ensureCoreFrameResources();
  void destroyScratchBuffer(ScratchBuffer &scratch);
  void destroyStaticFilmResources();
  void destroyCopyFrameResources();
  void destroyCoreFrameResources();
  void releaseTransientResources();
  void releaseTransientResourcesNoLock();
  void registerTransientBudgetEntry();
  void unregisterTransientBudgetEntry();
  uint64_t transientAllocationBytes() const;
  void refreshTransientBudgetEntry();
  void enforceTransientBudget();
  void updateSharedDiagnostics();
  bool attachBackend();
  bool createCommandPool();
  void clearBackendAliases();
  void markBackendLost();
  void initialize();
  bool selectPhysicalDevice();
  bool createLogicalDevice();
  bool createCopyPipeline();
  bool createCorePipelines();
  void cleanup();
};

std::mutex VulkanRenderer::Impl::transientRegistryMutex;
std::vector<VulkanRenderer::Impl *> VulkanRenderer::Impl::transientRegistry;
uint64_t VulkanRenderer::Impl::transientUseClock = 0;

void VulkanRenderer::Impl::registerTransientBudgetEntry() {
  std::lock_guard<std::mutex> lock(transientRegistryMutex);
  if (std::find(transientRegistry.begin(), transientRegistry.end(), this) == transientRegistry.end()) {
    transientRegistry.push_back(this);
  }
}

void VulkanRenderer::Impl::unregisterTransientBudgetEntry() {
  std::lock_guard<std::mutex> lock(transientRegistryMutex);
  transientRegistry.erase(std::remove(transientRegistry.begin(), transientRegistry.end(), this), transientRegistry.end());
}

uint64_t VulkanRenderer::Impl::transientAllocationBytes() const {
  uint64_t bytes = 0;
  auto add = [&](const ScratchBuffer &scratch) {
    bytes += static_cast<uint64_t>(scratch.allocationSize);
  };

  add(copyFrame.source);
  add(copyFrame.destination);

  add(coreFrame.sourceStaging);
  add(coreFrame.source);
  add(coreFrame.filmRaw);
  add(coreFrame.filmDensity);
  add(coreFrame.destination);
  add(coreFrame.destinationHalf);
  add(coreFrame.destinationStaging);
  add(coreFrame.destinationHalfStaging);
  add(coreFrame.halationRawA);
  add(coreFrame.halationRawB);
  add(coreFrame.halationRawC);
  add(coreFrame.halationRawD);
  add(coreFrame.halationBoostedRaw);
  add(coreFrame.halationBoostChunks);
  add(coreFrame.halationBoostInfo);
  add(coreFrame.halationBoostInfoReadback);
  add(coreFrame.tiledHalationBoostInfo);
  add(coreFrame.halationLogRaw);
  add(coreFrame.diffusionTemp);
  add(coreFrame.diffusionAccum);
  add(coreFrame.diffusionDownsampleSource);
  add(coreFrame.diffusionDownsampleTemp);
  add(coreFrame.diffusionDownsampleBlur);
  add(coreFrame.cameraDiffusionRaw);
  add(coreFrame.printRaw);
  add(coreFrame.printDiffusionRaw);
  add(coreFrame.dirCorrectionA);
  add(coreFrame.dirCorrectionB);
  add(coreFrame.dirCorrectionC);
  add(coreFrame.dirDensity);
  add(coreFrame.cameraDiffusionInfo);
  add(coreFrame.cameraDiffusionComponents);
  add(coreFrame.printDiffusionInfo);
  add(coreFrame.printDiffusionComponents);
  add(coreFrame.dirFloats);
  add(coreFrame.dirCorrectedDensityCurves);
  add(coreFrame.scannerPostA);
  add(coreFrame.scannerPostB);
  add(coreFrame.scannerPostC);
  add(coreFrame.printGlareA);
  add(coreFrame.printGlareB);
  add(coreFrame.grainDensityA);
  add(coreFrame.grainDensityB);
  add(coreFrame.grainMicroA);
  add(coreFrame.grainMicroB);
  add(coreFrame.grainLayerA);
  add(coreFrame.grainLayerB);
  add(coreFrame.frameFloats);
  add(coreFrame.frameInts);
  return bytes;
}

void VulkanRenderer::Impl::refreshTransientBudgetEntry() {
  const uint64_t bytes = transientAllocationBytes();
  std::lock_guard<std::mutex> lock(transientRegistryMutex);
  transientCachedBytes = bytes;
  lastUseSerial = ++transientUseClock;
}

void VulkanRenderer::Impl::updateSharedDiagnostics() {
  diagnostics.sharedBackend = backend && backend->isAvailable();
  diagnostics.sharedBackendGeneration = backend ? backend->generation : 0u;
  diagnostics.sharedQueueCount = backend ? backend->queueCount() : 0u;
  diagnostics.transientCachedBytes = transientCachedBytes;
  diagnostics.transientBudgetBytes = backend ? backend->transientBudgetBytes : 0u;
}

void VulkanRenderer::Impl::enforceTransientBudget() {
  if (!backend || backend->transientBudgetBytes == 0u) {
    return;
  }

  struct Candidate {
    Impl *renderer = nullptr;
    uint64_t bytes = 0;
    uint64_t lastUse = 0;
  };

  const uint64_t budget = backend->transientBudgetBytes;
  uint64_t total = 0;
  std::vector<Candidate> candidates;
  {
    std::lock_guard<std::mutex> lock(transientRegistryMutex);
    for (Impl *renderer : transientRegistry) {
      if (!renderer) {
        continue;
      }
      total += renderer->transientCachedBytes;
      if (renderer != this && renderer->transientCachedBytes != 0u) {
        candidates.push_back({renderer, renderer->transientCachedBytes, renderer->lastUseSerial});
      }
    }
  }

  if (total <= budget || candidates.empty()) {
    return;
  }

  std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
    return a.lastUse < b.lastUse;
  });

  for (const Candidate &candidate : candidates) {
    if (total <= budget) {
      break;
    }
    std::unique_lock<std::recursive_mutex> lock(candidate.renderer->renderMutex, std::try_to_lock);
    if (!lock.owns_lock()) {
      continue;
    }
    const uint64_t releasedBytes = candidate.renderer->transientCachedBytes;
    candidate.renderer->releaseTransientResourcesNoLock();
    total = releasedBytes > total ? 0u : total - releasedBytes;
  }
}

bool VulkanRenderer::Impl::createCommandPool() {
  VkCommandPoolCreateInfo poolInfo{};
  poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = computeQueueFamily;
  const VkResult result = vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkCreateCommandPool", result);
    return false;
  }
  return true;
}

void VulkanRenderer::Impl::clearBackendAliases() {
  instance = VK_NULL_HANDLE;
  physicalDevice = VK_NULL_HANDLE;
  physicalDeviceProperties = {};
  device = VK_NULL_HANDLE;
  computeQueue = VK_NULL_HANDLE;
  computeQueueFamily = 0;
  copyShaderModule = VK_NULL_HANDLE;
  formatConvertShaderModule = VK_NULL_HANDLE;
  copyDescriptorSetLayout = VK_NULL_HANDLE;
  copyPipelineLayout = VK_NULL_HANDLE;
  copyPipeline = VK_NULL_HANDLE;
  formatConvertPipeline = VK_NULL_HANDLE;
  filmExposureShaderModule = VK_NULL_HANDLE;
  curveDevelopShaderModule = VK_NULL_HANDLE;
  printScanShaderModule = VK_NULL_HANDLE;
  halationShaderModule = VK_NULL_HANDLE;
  diffusionShaderModule = VK_NULL_HANDLE;
  dirShaderModule = VK_NULL_HANDLE;
  scannerPostShaderModule = VK_NULL_HANDLE;
  grainShaderModule = VK_NULL_HANDLE;
  coreDescriptorSetLayout = VK_NULL_HANDLE;
  corePipelineLayout = VK_NULL_HANDLE;
  filmExposurePipeline = VK_NULL_HANDLE;
  curveDevelopPipeline = VK_NULL_HANDLE;
  printScanPipeline = VK_NULL_HANDLE;
  halationPipeline = VK_NULL_HANDLE;
  diffusionPipeline = VK_NULL_HANDLE;
  dirPipeline = VK_NULL_HANDLE;
  scannerPostPipeline = VK_NULL_HANDLE;
  grainPipeline = VK_NULL_HANDLE;
  queueIndex = 0u;
}

bool VulkanRenderer::Impl::attachBackend() {
  backend = acquireVulkanSharedBackend(lastError);
  if (!backend) {
    return false;
  }

  queueIndex = backend->assignQueueIndex();
  instance = backend->instance;
  physicalDevice = backend->physicalDevice;
  physicalDeviceProperties = backend->physicalDeviceProperties;
  device = backend->device;
  computeQueue = backend->queueForIndex(queueIndex);
  computeQueueFamily = backend->computeQueueFamily;
  copyShaderModule = backend->copyShaderModule;
  formatConvertShaderModule = backend->formatConvertShaderModule;
  copyDescriptorSetLayout = backend->copyDescriptorSetLayout;
  copyPipelineLayout = backend->copyPipelineLayout;
  copyPipeline = backend->copyPipeline;
  formatConvertPipeline = backend->formatConvertPipeline;
  filmExposureShaderModule = backend->filmExposureShaderModule;
  curveDevelopShaderModule = backend->curveDevelopShaderModule;
  printScanShaderModule = backend->printScanShaderModule;
  halationShaderModule = backend->halationShaderModule;
  diffusionShaderModule = backend->diffusionShaderModule;
  dirShaderModule = backend->dirShaderModule;
  scannerPostShaderModule = backend->scannerPostShaderModule;
  grainShaderModule = backend->grainShaderModule;
  coreDescriptorSetLayout = backend->coreDescriptorSetLayout;
  corePipelineLayout = backend->corePipelineLayout;
  filmExposurePipeline = backend->filmExposurePipeline;
  curveDevelopPipeline = backend->curveDevelopPipeline;
  printScanPipeline = backend->printScanPipeline;
  halationPipeline = backend->halationPipeline;
  diffusionPipeline = backend->diffusionPipeline;
  dirPipeline = backend->dirPipeline;
  scannerPostPipeline = backend->scannerPostPipeline;
  grainPipeline = backend->grainPipeline;
  return createCommandPool();
}

void VulkanRenderer::Impl::markBackendLost() {
  if (!backend) {
    return;
  }
  backend->markLost();
  clearVulkanSharedBackendIfSame(backend);
}

void VulkanRenderer::Impl::initialize() {
  diagnostics = {};
  lastError.clear();
  available = false;

  if (!attachBackend()) {
    cleanup();
    return;
  }

  refreshTransientBudgetEntry();
  updateSharedDiagnostics();
  available = true;
}

bool VulkanRenderer::Impl::selectPhysicalDevice() {
  return backend && physicalDevice != VK_NULL_HANDLE;
}

bool VulkanRenderer::Impl::createLogicalDevice() {
  if (!backend || device == VK_NULL_HANDLE) {
    lastError = "Shared Vulkan backend is not attached.";
    return false;
  }
  return commandPool != VK_NULL_HANDLE || createCommandPool();
}

bool VulkanRenderer::Impl::createCopyPipeline() {
  if (!backend || copyDescriptorSetLayout == VK_NULL_HANDLE ||
      copyPipelineLayout == VK_NULL_HANDLE ||
      copyPipeline == VK_NULL_HANDLE ||
      formatConvertPipeline == VK_NULL_HANDLE) {
    lastError = "Shared Vulkan copy pipeline is not available.";
    return false;
  }
  return true;
}

bool VulkanRenderer::Impl::createCorePipelines() {
  if (!backend || coreDescriptorSetLayout == VK_NULL_HANDLE ||
      corePipelineLayout == VK_NULL_HANDLE ||
      filmExposurePipeline == VK_NULL_HANDLE ||
      curveDevelopPipeline == VK_NULL_HANDLE ||
      printScanPipeline == VK_NULL_HANDLE ||
      halationPipeline == VK_NULL_HANDLE ||
      diffusionPipeline == VK_NULL_HANDLE ||
      dirPipeline == VK_NULL_HANDLE ||
      scannerPostPipeline == VK_NULL_HANDLE ||
      grainPipeline == VK_NULL_HANDLE) {
    lastError = "Shared Vulkan core pipelines are not available.";
    return false;
  }
  return true;
}

void VulkanRenderer::Impl::cleanup() {
  available = false;
  if (device != VK_NULL_HANDLE) {
    destroyCoreFrameResources();
    destroyCopyFrameResources();
    destroyStaticFilmResources();
    if (commandPool != VK_NULL_HANDLE) {
      vkDestroyCommandPool(device, commandPool, nullptr);
      commandPool = VK_NULL_HANDLE;
    }
  }
  refreshTransientBudgetEntry();
  updateSharedDiagnostics();
  clearBackendAliases();
  backend.reset();
}

bool VulkanRenderer::Impl::createBuffer(
  VkDeviceSize size,
  VkBufferUsageFlags usage,
  VkMemoryPropertyFlags memoryFlags,
  VkMemoryPropertyFlags preferredMemoryFlags,
  VkBuffer &buffer,
  VkDeviceMemory &memory,
  VkDeviceSize *allocatedBytes,
  VkMemoryPropertyFlags *actualMemoryFlags
) {
  VkBufferCreateInfo bufferInfo{};
  bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferInfo.size = size;
  bufferInfo.usage = usage;
  bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result = vkCreateBuffer(device, &bufferInfo, nullptr, &buffer);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkCreateBuffer", result);
    return false;
  }

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer, &requirements);
  const uint32_t memoryTypeIndex = findMemoryTypeIndex(
    physicalDevice,
    requirements.memoryTypeBits,
    memoryFlags,
    preferredMemoryFlags
  );
  if (memoryTypeIndex == std::numeric_limits<uint32_t>::max()) {
    lastError = "Unable to find a compatible Vulkan memory type.";
    return false;
  }

  VkMemoryAllocateInfo allocationInfo{};
  allocationInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocationInfo.allocationSize = requirements.size;
  allocationInfo.memoryTypeIndex = memoryTypeIndex;
  result = vkAllocateMemory(device, &allocationInfo, nullptr, &memory);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkAllocateMemory", result);
    return false;
  }

  result = vkBindBufferMemory(device, buffer, memory, 0);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkBindBufferMemory", result);
    return false;
  }
  if (allocatedBytes) {
    *allocatedBytes = requirements.size;
  }
  if (actualMemoryFlags) {
    VkPhysicalDeviceMemoryProperties memoryProperties{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    *actualMemoryFlags = memoryProperties.memoryTypes[memoryTypeIndex].propertyFlags;
  }
  return true;
}

void VulkanRenderer::Impl::destroyScratchBuffer(ScratchBuffer &scratch) {
  if (scratch.mapped) {
    vkUnmapMemory(device, scratch.memory);
    scratch.mapped = nullptr;
  }
  if (scratch.buffer != VK_NULL_HANDLE) {
    vkDestroyBuffer(device, scratch.buffer, nullptr);
    scratch.buffer = VK_NULL_HANDLE;
  }
  if (scratch.memory != VK_NULL_HANDLE) {
    vkFreeMemory(device, scratch.memory, nullptr);
    scratch.memory = VK_NULL_HANDLE;
  }
  scratch.capacity = 0;
  scratch.allocationSize = 0;
  scratch.memoryFlags = 0;
}

void VulkanRenderer::Impl::destroyStaticFilmResources() {
  destroyScratchBuffer(staticFilm.academyPrinterDensityData);
  destroyScratchBuffer(staticFilm.colorEncodeLuts);
  destroyScratchBuffer(staticFilm.paperScanToOutputRgb);
  destroyScratchBuffer(staticFilm.filmScanToOutputRgb);
  destroyScratchBuffer(staticFilm.standardObserverCmfs);
  destroyScratchBuffer(staticFilm.paperScanIlluminant);
  destroyScratchBuffer(staticFilm.filmScanIlluminant);
  destroyScratchBuffer(staticFilm.paperBaseDensity);
  destroyScratchBuffer(staticFilm.paperChannelDensity);
  destroyScratchBuffer(staticFilm.neutralPrintFilters);
  destroyScratchBuffer(staticFilm.customEnlargerFilters);
  destroyScratchBuffer(staticFilm.thKg3Illuminant);
  destroyScratchBuffer(staticFilm.paperLogSensitivity);
  destroyScratchBuffer(staticFilm.filmBaseDensity);
  destroyScratchBuffer(staticFilm.filmChannelDensity);
  destroyScratchBuffer(staticFilm.paperDensityCurves);
  destroyScratchBuffer(staticFilm.paperLogExposure);
  destroyScratchBuffer(staticFilm.hanatosRawResponse);
  destroyScratchBuffer(staticFilm.paperHanatosResponse);
  destroyScratchBuffer(staticFilm.preflashPaperHanatosResponse);
  destroyScratchBuffer(staticFilm.mallettRawMatrix);
  destroyScratchBuffer(staticFilm.colorTransferKinds);
  destroyScratchBuffer(staticFilm.colorDecodeLuts);
  destroyScratchBuffer(staticFilm.inputToSrgb);
  destroyScratchBuffer(staticFilm.inputToReferenceXyz);
  destroyScratchBuffer(staticFilm.densityCurveLayerMaxima);
  destroyScratchBuffer(staticFilm.densityCurveLayers);
  destroyScratchBuffer(staticFilm.densityCurves);
  destroyScratchBuffer(staticFilm.logExposure);
  staticFilm.film = -1;
  staticFilm.paper = -1;
  staticFilm.rgbToRawMethod = RgbToRawMethod::Hanatos2026;
  staticFilm.cameraUvFilterEnabled = false;
  staticFilm.cameraUvCutNm = 0.0f;
  staticFilm.cameraIrFilterEnabled = false;
  staticFilm.cameraIrCutNm = 0.0f;
  staticFilm.processNegativePrintTiming = PrintTimingMode::FilteredEnlarger;
  staticFilm.processNegativeFilterC = 0.0f;
  staticFilm.processNegativeFilterMShift = 0.0f;
  staticFilm.processNegativeFilterYShift = 0.0f;
  staticFilm.processNegativePreflashMFilterShift = 0.0f;
  staticFilm.processNegativePreflashYFilterShift = 0.0f;
  staticFilm.printScanResources = false;
  staticFilm.grainResources = false;
  staticFilm.curves = nullptr;
  staticFilm.paperCurves = nullptr;
  staticFilm.exposureCount = 0;
  staticFilm.paperExposureCount = 0;
  staticFilm.wavelengthCount = 0;
  staticFilm.filmPositive = 0;
  staticFilm.hanatosWidth = 0;
  staticFilm.hanatosHeight = 0;
  staticFilm.filmDensityMaximum = {0.0f, 0.0f, 0.0f};
  staticFilm.paperDensityMaximum = {0.0f, 0.0f, 0.0f};
}

bool VulkanRenderer::Impl::ensureScratchBuffer(
  ScratchBuffer &scratch,
  VkDeviceSize size,
  const char *name,
  VkBufferUsageFlags usage,
  VkMemoryPropertyFlags memoryFlags,
  VkMemoryPropertyFlags preferredMemoryFlags,
  bool persistentMap,
  bool countPrivate
) {
  if (scratch.buffer != VK_NULL_HANDLE &&
      scratch.memory != VK_NULL_HANDLE &&
      scratch.capacity >= size &&
      ((scratch.memoryFlags & memoryFlags) == memoryFlags) &&
      (!persistentMap || scratch.mapped)) {
    return true;
  }

  destroyScratchBuffer(scratch);
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize allocatedBytes = 0;
  VkMemoryPropertyFlags actualMemoryFlags = 0;
  if (!createBuffer(size, usage, memoryFlags, preferredMemoryFlags, buffer, memory, &allocatedBytes, &actualMemoryFlags)) {
    if (buffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, buffer, nullptr);
    }
    if (memory != VK_NULL_HANDLE) {
      vkFreeMemory(device, memory, nullptr);
    }
    if (lastError.empty()) {
      lastError = std::string("Unable to allocate Vulkan scratch buffer ") + name + ".";
    }
    return false;
  }

  void *mapped = nullptr;
  if (persistentMap) {
    const VkResult mapResult = vkMapMemory(device, memory, 0, allocatedBytes, 0, &mapped);
    if (mapResult != VK_SUCCESS) {
      vkDestroyBuffer(device, buffer, nullptr);
      vkFreeMemory(device, memory, nullptr);
      lastError = vkError("vkMapMemory(scratch)", mapResult);
      return false;
    }
  }

  scratch.buffer = buffer;
  scratch.memory = memory;
  scratch.capacity = size;
  scratch.allocationSize = allocatedBytes;
  scratch.memoryFlags = actualMemoryFlags;
  scratch.mapped = mapped;
  diagnostics.scratchAllocationBytes += static_cast<uint64_t>(allocatedBytes);
  diagnostics.scratchAllocationCount += 1u;
  if (countPrivate) {
    diagnostics.privateScratchAllocationBytes += static_cast<uint64_t>(allocatedBytes);
    diagnostics.privateScratchAllocationCount += 1u;
  } else {
    diagnostics.sharedScratchAllocationBytes += static_cast<uint64_t>(allocatedBytes);
    diagnostics.sharedScratchAllocationCount += 1u;
  }
  return true;
}

bool VulkanRenderer::Impl::flushMappedScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name) {
  if ((scratch.memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u || size == 0) {
    return true;
  }
  if (scratch.memory == VK_NULL_HANDLE) {
    lastError = std::string("Unable to flush unmapped Vulkan buffer ") + name + ".";
    return false;
  }

  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = scratch.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  const VkResult result = vkFlushMappedMemoryRanges(device, 1, &range);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkFlushMappedMemoryRanges", result);
    return false;
  }
  return true;
}

bool VulkanRenderer::Impl::invalidateMappedScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name) {
  if ((scratch.memoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0u || size == 0) {
    return true;
  }
  if (scratch.memory == VK_NULL_HANDLE) {
    lastError = std::string("Unable to invalidate unmapped Vulkan buffer ") + name + ".";
    return false;
  }

  VkMappedMemoryRange range{};
  range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  range.memory = scratch.memory;
  range.offset = 0;
  range.size = VK_WHOLE_SIZE;
  const VkResult result = vkInvalidateMappedMemoryRanges(device, 1, &range);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkInvalidateMappedMemoryRanges", result);
    return false;
  }
  return true;
}

bool VulkanRenderer::Impl::ensureSharedScratchBuffer(
  ScratchBuffer &scratch,
  VkDeviceSize size,
  const char *name,
  VkBufferUsageFlags usage
) {
  return ensureUploadScratchBuffer(scratch, size, name, usage);
}

bool VulkanRenderer::Impl::ensureUploadScratchBuffer(
  ScratchBuffer &scratch,
  VkDeviceSize size,
  const char *name,
  VkBufferUsageFlags usage
) {
  constexpr VkMemoryPropertyFlags hostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  constexpr VkMemoryPropertyFlags preferredHostMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  return ensureScratchBuffer(scratch, size, name, usage, hostMemory, preferredHostMemory, true, false);
}

bool VulkanRenderer::Impl::ensureReadbackScratchBuffer(
  ScratchBuffer &scratch,
  VkDeviceSize size,
  const char *name,
  VkBufferUsageFlags usage
) {
  constexpr VkMemoryPropertyFlags hostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  constexpr VkMemoryPropertyFlags preferredHostMemory = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  return ensureScratchBuffer(scratch, size, name, usage, hostMemory, preferredHostMemory, true, false);
}

bool VulkanRenderer::Impl::ensurePrivateScratchBuffer(
  ScratchBuffer &scratch,
  VkDeviceSize size,
  const char *name,
  VkBufferUsageFlags usage
) {
  if (scratch.buffer != VK_NULL_HANDLE &&
      scratch.memory != VK_NULL_HANDLE &&
      scratch.capacity >= size &&
      ((scratch.memoryFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0u)) {
    return true;
  }
  if (preferPrivateScratch &&
      ensureScratchBuffer(scratch, size, name, usage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0, false, true)) {
    return true;
  }
  preferPrivateScratch = false;
  lastError.clear();
  return ensureSharedScratchBuffer(scratch, size, name, usage);
}

bool VulkanRenderer::Impl::ensureScratchBuffer(ScratchBuffer &scratch, VkDeviceSize size, const char *name) {
  return ensureSharedScratchBuffer(scratch, size, name);
}

bool VulkanRenderer::Impl::uploadScratchBuffer(ScratchBuffer &scratch, const void *data, VkDeviceSize size, const char *name) {
  if (!data || size == 0) {
    lastError = std::string("Unable to upload empty Vulkan buffer ") + name + ".";
    return false;
  }
  if (!ensureSharedScratchBuffer(scratch, size, name)) {
    return false;
  }

  if (!scratch.mapped) {
    lastError = std::string("Vulkan shared scratch buffer is not mapped for ") + name + ".";
    return false;
  }
  std::memcpy(scratch.mapped, data, static_cast<size_t>(size));
  if (!flushMappedScratchBuffer(scratch, size, name)) {
    return false;
  }
  diagnostics.uploadBytes += static_cast<uint64_t>(size);
  return true;
}

bool VulkanRenderer::Impl::copyBufferImmediate(VkBuffer source, VkBuffer destination, VkDeviceSize size, const char *name) {
  VkCommandBufferAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocateInfo.commandPool = commandPool;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocateInfo.commandBufferCount = 1;

  VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
  VkResult result = vkAllocateCommandBuffers(device, &allocateInfo, &commandBuffer);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkAllocateCommandBuffers(upload)", result);
    return false;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    lastError = vkError("vkBeginCommandBuffer(upload)", result);
    return false;
  }

  VkMemoryBarrier hostInputBarrier{};
  hostInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostInputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  hostInputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    0,
    1,
    &hostInputBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  VkBufferCopy copyRegion{};
  copyRegion.size = size;
  vkCmdCopyBuffer(commandBuffer, source, destination, 1, &copyRegion);

  VkMemoryBarrier uploadBarrier{};
  uploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  uploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  uploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_TRANSFER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0,
    1,
    &uploadBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    lastError = vkError("vkEndCommandBuffer(upload)", result);
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  VkFenceCreateInfo fenceInfo{};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  result = vkCreateFence(device, &fenceInfo, nullptr, &fence);
  if (result != VK_SUCCESS) {
    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    lastError = vkError("vkCreateFence(upload)", result);
    return false;
  }

  result = backend ? backend->submit(queueIndex, submitInfo, fence) : VK_ERROR_INITIALIZATION_FAILED;
  if (result == VK_SUCCESS) {
    result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  }
  vkDestroyFence(device, fence, nullptr);
  vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
  if (result != VK_SUCCESS) {
    lastError = vkError(name, result);
    return false;
  }
  return true;
}

bool VulkanRenderer::Impl::uploadStaticBuffer(ScratchBuffer &scratch, const void *data, VkDeviceSize size, const char *name) {
  if (!data || size == 0) {
    lastError = std::string("Unable to upload empty Vulkan buffer ") + name + ".";
    return false;
  }

  constexpr VkBufferUsageFlags storageUsage =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (!ensurePrivateScratchBuffer(scratch, size, name, storageUsage)) {
    return false;
  }

  VkBuffer stagingBuffer = VK_NULL_HANDLE;
  VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
  VkDeviceSize stagingAllocatedBytes = 0;
  VkMemoryPropertyFlags stagingMemoryFlags = 0;
  constexpr VkMemoryPropertyFlags hostMemory = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  constexpr VkMemoryPropertyFlags preferredHostMemory = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!createBuffer(
        size,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        hostMemory,
        preferredHostMemory,
        stagingBuffer,
        stagingMemory,
        &stagingAllocatedBytes,
        &stagingMemoryFlags
      )) {
    if (stagingBuffer != VK_NULL_HANDLE) {
      vkDestroyBuffer(device, stagingBuffer, nullptr);
    }
    if (stagingMemory != VK_NULL_HANDLE) {
      vkFreeMemory(device, stagingMemory, nullptr);
    }
    return false;
  }

  void *mapped = nullptr;
  VkResult result = vkMapMemory(device, stagingMemory, 0, stagingAllocatedBytes, 0, &mapped);
  if (result != VK_SUCCESS) {
    vkDestroyBuffer(device, stagingBuffer, nullptr);
    vkFreeMemory(device, stagingMemory, nullptr);
    lastError = vkError("vkMapMemory(static staging)", result);
    return false;
  }
  std::memcpy(mapped, data, static_cast<size_t>(size));
  if ((stagingMemoryFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0u) {
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = stagingMemory;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    result = vkFlushMappedMemoryRanges(device, 1, &range);
    if (result != VK_SUCCESS) {
      vkUnmapMemory(device, stagingMemory);
      vkDestroyBuffer(device, stagingBuffer, nullptr);
      vkFreeMemory(device, stagingMemory, nullptr);
      lastError = vkError("vkFlushMappedMemoryRanges(static staging)", result);
      return false;
    }
  }
  vkUnmapMemory(device, stagingMemory);

  const bool ok = copyBufferImmediate(stagingBuffer, scratch.buffer, size, "vkQueueSubmit(static upload)");
  vkDestroyBuffer(device, stagingBuffer, nullptr);
  vkFreeMemory(device, stagingMemory, nullptr);
  if (!ok) {
    return false;
  }

  diagnostics.uploadBytes += static_cast<uint64_t>(size);
  return true;
}

bool VulkanRenderer::Impl::loadHanatosSpectraData() {
  const HanatosSpectraLutInfo &hanatos = hanatosSpectraLutInfo();
  if (hanatos.elementCount == 0u || hanatos.width == 0u || hanatos.height == 0u || hanatos.wavelengthCount == 0u) {
    lastError = "Generated Hanatos spectra LUT metadata is unavailable.";
    return false;
  }
  if (hanatosSpectraData.size() == hanatos.elementCount) {
    return true;
  }

  const std::filesystem::path lutPath = findResourcePath("SpektraHanatos2025Spectra.f32");
  if (lutPath.empty()) {
    lastError = "Unable to locate SpektraHanatos2025Spectra.f32 for Vulkan Hanatos RGB-to-raw.";
    return false;
  }

  std::string resourceError;
  std::vector<float> data = readFloatResourceFile(lutPath, hanatos.elementCount, resourceError);
  if (data.empty()) {
    lastError = resourceError;
    return false;
  }
  hanatosSpectraData = std::move(data);
  return true;
}

bool VulkanRenderer::Impl::loadOutputGamutCompressionData() {
  if (outputGamutCompressionData.size() == kSpektraOutputGamutCompressionElementCount) {
    return true;
  }

  const std::filesystem::path dataPath = findResourcePath("SpektraOutputGamutCompression.f32");
  if (dataPath.empty()) {
    lastError = "Unable to locate SpektraOutputGamutCompression.f32 for Vulkan output gamut compression.";
    return false;
  }

  std::string resourceError;
  std::vector<float> data = readFloatResourceFile(
    dataPath,
    kSpektraOutputGamutCompressionElementCount,
    resourceError
  );
  if (data.empty()) {
    lastError = resourceError;
    return false;
  }
  outputGamutCompressionData = std::move(data);
  return true;
}

bool VulkanRenderer::Impl::prepareStaticFilmResources(
  const RenderParams &params,
  bool includePrintScanResources,
  bool includeGrainResources
) {
  const bool processNegativeResponseCached =
    params.process != ProcessMode::ProcessNegative ||
    (staticFilm.processNegativePrintTiming == params.printTiming &&
     staticFilm.processNegativeFilterC == params.filterC &&
     staticFilm.processNegativeFilterMShift == params.filterMShift &&
     staticFilm.processNegativeFilterYShift == params.filterYShift &&
     staticFilm.processNegativePreflashMFilterShift == params.preflashMFilterShift &&
     staticFilm.processNegativePreflashYFilterShift == params.preflashYFilterShift);
  const bool coreResourcesCached =
    staticFilm.film == params.film &&
    staticFilm.rgbToRawMethod == params.rgbToRawMethod &&
    staticFilm.cameraUvFilterEnabled == params.cameraUvFilterEnabled &&
    staticFilm.cameraUvCutNm == params.cameraUvCutNm &&
    staticFilm.cameraIrFilterEnabled == params.cameraIrFilterEnabled &&
    staticFilm.cameraIrCutNm == params.cameraIrCutNm &&
    processNegativeResponseCached &&
    staticFilm.curves &&
    staticFilm.exposureCount > 0u &&
    staticFilm.wavelengthCount > 0u &&
    staticFilm.logExposure.buffer != VK_NULL_HANDLE &&
    staticFilm.densityCurves.buffer != VK_NULL_HANDLE &&
    staticFilm.inputToReferenceXyz.buffer != VK_NULL_HANDLE &&
    staticFilm.inputToSrgb.buffer != VK_NULL_HANDLE &&
    staticFilm.colorDecodeLuts.buffer != VK_NULL_HANDLE &&
    staticFilm.colorTransferKinds.buffer != VK_NULL_HANDLE &&
    staticFilm.mallettRawMatrix.buffer != VK_NULL_HANDLE &&
    staticFilm.hanatosRawResponse.buffer != VK_NULL_HANDLE &&
    staticFilm.paperHanatosResponse.buffer != VK_NULL_HANDLE &&
    staticFilm.preflashPaperHanatosResponse.buffer != VK_NULL_HANDLE;
  const bool printScanResourcesCached =
    !includePrintScanResources ||
    (staticFilm.printScanResources &&
     staticFilm.paper == params.paper &&
     staticFilm.paperCurves &&
     staticFilm.paperExposureCount > 0u &&
     staticFilm.paperLogExposure.buffer != VK_NULL_HANDLE &&
     staticFilm.paperDensityCurves.buffer != VK_NULL_HANDLE &&
     staticFilm.filmChannelDensity.buffer != VK_NULL_HANDLE &&
     staticFilm.filmBaseDensity.buffer != VK_NULL_HANDLE &&
     staticFilm.paperLogSensitivity.buffer != VK_NULL_HANDLE &&
     staticFilm.thKg3Illuminant.buffer != VK_NULL_HANDLE &&
     staticFilm.customEnlargerFilters.buffer != VK_NULL_HANDLE &&
     staticFilm.neutralPrintFilters.buffer != VK_NULL_HANDLE &&
     staticFilm.paperChannelDensity.buffer != VK_NULL_HANDLE &&
     staticFilm.paperBaseDensity.buffer != VK_NULL_HANDLE &&
     staticFilm.filmScanIlluminant.buffer != VK_NULL_HANDLE &&
     staticFilm.paperScanIlluminant.buffer != VK_NULL_HANDLE &&
     staticFilm.standardObserverCmfs.buffer != VK_NULL_HANDLE &&
     staticFilm.filmScanToOutputRgb.buffer != VK_NULL_HANDLE &&
     staticFilm.paperScanToOutputRgb.buffer != VK_NULL_HANDLE &&
     staticFilm.colorEncodeLuts.buffer != VK_NULL_HANDLE &&
     staticFilm.academyPrinterDensityData.buffer != VK_NULL_HANDLE);
  const bool grainResourcesCached =
    !includeGrainResources ||
    (staticFilm.grainResources &&
     staticFilm.densityCurveLayers.buffer != VK_NULL_HANDLE &&
     staticFilm.densityCurveLayerMaxima.buffer != VK_NULL_HANDLE);
  if (coreResourcesCached && printScanResourcesCached && grainResourcesCached) {
    return true;
  }

  const ProfileCurveSet *curves = filmProfileCurves(params.film);
  if (!curves) {
    curves = filmProfileCurves(static_cast<int32_t>(kSpektraDefaultFilmIndex));
  }
  const ProfileCurveSet *paperCurves = nullptr;
  if (includePrintScanResources) {
    paperCurves = paperProfileCurves(params.paper);
    if (!paperCurves) {
      paperCurves = paperProfileCurves(static_cast<int32_t>(kSpektraDefaultPaperIndex));
    }
  }
  if (!curves || curves->exposureCount == 0u || !curves->logExposure || !curves->densityCurves) {
    lastError = "Unable to locate generated film density curves.";
    return false;
  }
  if (includeGrainResources && (!curves->densityCurveLayers || !curves->densityCurveLayerMaxima)) {
    lastError = "Unable to locate generated film grain layer data for the Vulkan grain pass.";
    return false;
  }
  if (includePrintScanResources &&
      (!paperCurves || paperCurves->exposureCount == 0u || !paperCurves->logExposure || !paperCurves->densityCurves)) {
    lastError = "Unable to locate generated paper density curves.";
    return false;
  }
  if (!curves->inputToReferenceXyz || !curves->inputToSrgb || !colorDecodeLuts() || !colorTransferKinds()) {
    lastError = "Unable to locate generated Vulkan input color transform data.";
    return false;
  }
  if (includePrintScanResources && (!colorEncodeLuts() || !colorTransferParams())) {
    lastError = "Unable to locate generated Vulkan output color transform data.";
    return false;
  }
  if (curves->wavelengthCount == 0u || !curves->logSensitivity ||
      !curves->mallettBasisIlluminant || curves->mallettRawMidgrayGreen <= 0.0f) {
    lastError = "Unable to locate generated Vulkan Mallett raw matrix data.";
    return false;
  }
  if (includePrintScanResources &&
      (paperCurves->wavelengthCount != curves->wavelengthCount || !paperCurves->logSensitivity)) {
    lastError = "Unable to locate generated Vulkan paper spectral sensitivity data.";
    return false;
  }
  if (includePrintScanResources &&
      (!curves->channelDensity || !curves->baseDensity || !curves->densityCurveMinimum ||
       !paperCurves->channelDensity || !paperCurves->baseDensity)) {
    lastError = "Unable to locate generated Vulkan spectral density data.";
    return false;
  }
  if (includePrintScanResources &&
      (!curves->scanIlluminant || !paperCurves->scanIlluminant ||
       !curves->scanToOutputRgb || !paperCurves->scanToOutputRgb || !standardObserverCmfs())) {
    lastError = "Unable to locate generated Vulkan scan conversion data.";
    return false;
  }
  if (includePrintScanResources &&
      (!thKg3Illuminant() || !customEnlargerFilters() || !neutralPrintFilters() || !academyPrinterDensityData())) {
    lastError = "Unable to locate generated Vulkan print exposure data.";
    return false;
  }
  if (params.rgbToRawMethod == RgbToRawMethod::Hanatos2025 && !curves->bandpassHanatos2025) {
    lastError = "Unable to locate archived Hanatos 2025 film bandpass data.";
    return false;
  }
  if (params.rgbToRawMethod == RgbToRawMethod::Hanatos2026 &&
      (!curves->hanatos2026WindowParams || !curves->referenceIlluminantSpectrum || !curves->wavelengths)) {
    lastError = "Unable to locate generated Hanatos 2026 film adaptation data.";
    return false;
  }
  const bool needsHanatos = params.rgbToRawMethod != RgbToRawMethod::Mallett2019;
  if (needsHanatos && !loadHanatosSpectraData()) {
    return false;
  }
  if (includePrintScanResources && !loadOutputGamutCompressionData()) {
    return false;
  }

  const std::vector<float> packedFilmCurveExposure =
    makePackedCurveExposure(curves->logExposure, curves->exposureCount);
  const VkDeviceSize logExposureBytes = static_cast<VkDeviceSize>(packedFilmCurveExposure.size()) * sizeof(float);
  const VkDeviceSize densityCurveBytes = static_cast<VkDeviceSize>(curves->exposureCount) * 3u * sizeof(float);
  const VkDeviceSize densityCurveLayerBytes = static_cast<VkDeviceSize>(curves->exposureCount) * 9u * sizeof(float);
  const VkDeviceSize densityCurveLayerMaximaBytes = 9u * sizeof(float);
  const VkDeviceSize inputMatrixBytes = static_cast<VkDeviceSize>(kSpektraColorSpaceCount) * 9u * sizeof(float);
  const VkDeviceSize transferLutBytes =
    static_cast<VkDeviceSize>(kSpektraColorSpaceCount) *
    static_cast<VkDeviceSize>(kSpektraColorTransferLutSize) *
    sizeof(float);
  std::vector<float> colorEncodeAndGamutData;
  if (includePrintScanResources) {
    colorEncodeAndGamutData.reserve(
      static_cast<size_t>(kSpektraColorSpaceCount) * static_cast<size_t>(kSpektraColorTransferLutSize) +
      outputGamutCompressionData.size() +
      static_cast<size_t>(kSpektraColorSpaceCount)
    );
    colorEncodeAndGamutData.insert(
      colorEncodeAndGamutData.end(),
      colorEncodeLuts(),
      colorEncodeLuts() + static_cast<size_t>(kSpektraColorSpaceCount) * static_cast<size_t>(kSpektraColorTransferLutSize)
    );
    colorEncodeAndGamutData.insert(
      colorEncodeAndGamutData.end(),
      outputGamutCompressionData.begin(),
      outputGamutCompressionData.end()
    );
    colorEncodeAndGamutData.insert(
      colorEncodeAndGamutData.end(),
      colorTransferParams(),
      colorTransferParams() + static_cast<size_t>(kSpektraColorSpaceCount)
    );
  }
  const VkDeviceSize transferKindBytes = static_cast<VkDeviceSize>(kSpektraColorSpaceCount) * sizeof(uint32_t);
  const VkDeviceSize mallettRawMatrixBytes = 9u * sizeof(float);
  const std::vector<float> baseFilmSensitivityLinear = makeLinearSensitivity(curves->logSensitivity, curves->wavelengthCount);
  const std::vector<float> filmSensitivityLinear = applyCameraBandPass(*curves, baseFilmSensitivityLinear, params);
  const std::array<float, 9> mallettRawMatrix = makeMallettRawMatrix(*curves, filmSensitivityLinear);
  const HanatosSpectraLutInfo &hanatos = hanatosSpectraLutInfo();
  std::vector<float> hanatosRawResponse = needsHanatos
    ? makeHanatosRawResponsePair(*curves, filmSensitivityLinear, hanatosSpectraData, hanatos, params.rgbToRawMethod)
    : std::vector<float>(8u, 0.0f);
  if (hanatosRawResponse.empty()) {
    hanatosRawResponse.assign(8u, 0.0f);
  }
  std::vector<float> paperHanatosResponse(8u, 0.0f);
  std::vector<float> preflashPaperHanatosResponse(8u, 0.0f);
  if (includePrintScanResources) {
    const std::vector<float> paperSensitivityLinear = makeLinearSensitivity(paperCurves->logSensitivity, paperCurves->wavelengthCount);
    const std::vector<float> processNegativePaperWeights = makeProcessNegativePaperWeights(
      *paperCurves,
      paperSensitivityLinear,
      params,
      false
    );
    const std::vector<float> processNegativePreflashWeights = makeProcessNegativePaperWeights(
      *paperCurves,
      paperSensitivityLinear,
      params,
      true
    );
    paperHanatosResponse = makeHanatosResponsePairFromWeights(*curves, processNegativePaperWeights, hanatosSpectraData, hanatos);
    preflashPaperHanatosResponse = makeHanatosResponsePairFromWeights(*curves, processNegativePreflashWeights, hanatosSpectraData, hanatos);
  }
  const VkDeviceSize hanatosRawResponseBytes =
    static_cast<VkDeviceSize>(hanatosRawResponse.size()) * sizeof(float);
  const VkDeviceSize paperHanatosResponseBytes =
    static_cast<VkDeviceSize>(paperHanatosResponse.size()) * sizeof(float);
  const VkDeviceSize preflashPaperHanatosResponseBytes =
    static_cast<VkDeviceSize>(preflashPaperHanatosResponse.size()) * sizeof(float);
  const uint64_t scratchBytesBefore = diagnostics.scratchAllocationBytes;
  const uint64_t scratchCountBefore = diagnostics.scratchAllocationCount;
  const uint64_t sharedBytesBefore = diagnostics.sharedScratchAllocationBytes;
  const uint64_t sharedCountBefore = diagnostics.sharedScratchAllocationCount;
  const uint64_t privateBytesBefore = diagnostics.privateScratchAllocationBytes;
  const uint64_t privateCountBefore = diagnostics.privateScratchAllocationCount;

  if (!uploadStaticBuffer(staticFilm.logExposure, packedFilmCurveExposure.data(), logExposureBytes, "packed film curve exposure") ||
      !uploadStaticBuffer(staticFilm.densityCurves, curves->densityCurves, densityCurveBytes, "film density curves") ||
      !uploadStaticBuffer(staticFilm.inputToReferenceXyz, curves->inputToReferenceXyz, inputMatrixBytes, "input to reference XYZ matrices") ||
      !uploadStaticBuffer(staticFilm.inputToSrgb, curves->inputToSrgb, inputMatrixBytes, "input to sRGB matrices") ||
      !uploadStaticBuffer(staticFilm.colorDecodeLuts, colorDecodeLuts(), transferLutBytes, "color decode LUTs") ||
      !uploadStaticBuffer(staticFilm.colorTransferKinds, colorTransferKinds(), transferKindBytes, "color transfer kinds") ||
      !uploadStaticBuffer(staticFilm.mallettRawMatrix, mallettRawMatrix.data(), mallettRawMatrixBytes, "Mallett raw matrix") ||
      !uploadStaticBuffer(staticFilm.hanatosRawResponse, hanatosRawResponse.data(), hanatosRawResponseBytes, "Hanatos raw response") ||
      !uploadStaticBuffer(staticFilm.paperHanatosResponse, paperHanatosResponse.data(), paperHanatosResponseBytes, "paper Hanatos response") ||
      !uploadStaticBuffer(staticFilm.preflashPaperHanatosResponse, preflashPaperHanatosResponse.data(), preflashPaperHanatosResponseBytes, "preflash paper Hanatos response")) {
    return false;
  }
  if (includeGrainResources &&
      (!uploadStaticBuffer(staticFilm.densityCurveLayers, curves->densityCurveLayers, densityCurveLayerBytes, "film density curve layers") ||
       !uploadStaticBuffer(
         staticFilm.densityCurveLayerMaxima,
         curves->densityCurveLayerMaxima,
         densityCurveLayerMaximaBytes,
         "film density curve layer maxima"
       ))) {
    return false;
  }
  if (includePrintScanResources) {
    const std::vector<float> packedPaperCurveExposure =
      makePackedCurveExposure(paperCurves->logExposure, paperCurves->exposureCount);
    const VkDeviceSize paperLogExposureBytes =
      static_cast<VkDeviceSize>(packedPaperCurveExposure.size()) * sizeof(float);
    const VkDeviceSize paperDensityCurveBytes = static_cast<VkDeviceSize>(paperCurves->exposureCount) * 3u * sizeof(float);
    const VkDeviceSize wavelengthBytes = static_cast<VkDeviceSize>(curves->wavelengthCount) * sizeof(float);
    const VkDeviceSize spectralTripletBytes = static_cast<VkDeviceSize>(curves->wavelengthCount) * 3u * sizeof(float);
    const VkDeviceSize neutralPrintFilterBytes =
      static_cast<VkDeviceSize>(kSpektraPaperCount) * static_cast<VkDeviceSize>(kSpektraFilmCount) * 3u * sizeof(float);
    const VkDeviceSize academyPrinterDensityBytes =
      (static_cast<VkDeviceSize>(curves->wavelengthCount) * 3u +
       static_cast<VkDeviceSize>(kSpektraPaperCount) * static_cast<VkDeviceSize>(kSpektraFilmCount) * 3u) *
      sizeof(float);
    const std::vector<float> paperSensitivityLinear =
      makeLinearSensitivity(paperCurves->logSensitivity, paperCurves->wavelengthCount);
    const std::vector<float> packedFilmSpectralDensity =
      makePackedSpectralDensity(curves->channelDensity, curves->baseDensity, curves->wavelengthCount);
    const std::vector<float> packedPaperSpectralDensity =
      makePackedSpectralDensity(paperCurves->channelDensity, paperCurves->baseDensity, paperCurves->wavelengthCount);
    const std::vector<float> scanProducts = makeScanProducts(
      curves->scanIlluminant,
      paperCurves->scanIlluminant,
      curves->baseDensity,
      paperCurves->baseDensity,
      standardObserverCmfs(),
      curves->wavelengthCount
    );
    if (!uploadStaticBuffer(staticFilm.paperLogExposure, packedPaperCurveExposure.data(), paperLogExposureBytes, "packed paper curve exposure") ||
        !uploadStaticBuffer(staticFilm.paperDensityCurves, paperCurves->densityCurves, paperDensityCurveBytes, "paper density curves") ||
        !uploadStaticBuffer(staticFilm.filmChannelDensity, packedFilmSpectralDensity.data(), packedFilmSpectralDensity.size() * sizeof(float), "packed film spectral density") ||
        !uploadStaticBuffer(staticFilm.filmBaseDensity, curves->baseDensity, wavelengthBytes, "film base density") ||
        !uploadStaticBuffer(staticFilm.paperLogSensitivity, paperSensitivityLinear.data(), spectralTripletBytes, "paper linear sensitivity") ||
        !uploadStaticBuffer(staticFilm.thKg3Illuminant, thKg3Illuminant(), wavelengthBytes, "TH-KG3 illuminant") ||
        !uploadStaticBuffer(staticFilm.customEnlargerFilters, customEnlargerFilters(), spectralTripletBytes, "custom enlarger filters") ||
        !uploadStaticBuffer(staticFilm.neutralPrintFilters, neutralPrintFilters(), neutralPrintFilterBytes, "neutral print filters") ||
        !uploadStaticBuffer(staticFilm.paperChannelDensity, packedPaperSpectralDensity.data(), packedPaperSpectralDensity.size() * sizeof(float), "packed paper spectral density") ||
        !uploadStaticBuffer(staticFilm.paperBaseDensity, paperCurves->baseDensity, wavelengthBytes, "paper base density") ||
        !uploadStaticBuffer(staticFilm.filmScanIlluminant, scanProducts.data(), scanProducts.size() * sizeof(float), "packed scan products") ||
        !uploadStaticBuffer(staticFilm.paperScanIlluminant, paperCurves->scanIlluminant, wavelengthBytes, "paper scan illuminant") ||
        !uploadStaticBuffer(staticFilm.standardObserverCmfs, standardObserverCmfs(), spectralTripletBytes, "standard observer CMFs") ||
        !uploadStaticBuffer(staticFilm.filmScanToOutputRgb, curves->scanToOutputRgb, inputMatrixBytes, "film scan to output RGB") ||
        !uploadStaticBuffer(staticFilm.paperScanToOutputRgb, paperCurves->scanToOutputRgb, inputMatrixBytes, "paper scan to output RGB") ||
        !uploadStaticBuffer(
          staticFilm.colorEncodeLuts,
          colorEncodeAndGamutData.data(),
          static_cast<VkDeviceSize>(colorEncodeAndGamutData.size()) * sizeof(float),
          "color encode LUTs and output gamut compression data"
        ) ||
        !uploadStaticBuffer(staticFilm.academyPrinterDensityData, academyPrinterDensityData(), academyPrinterDensityBytes, "Academy printer density data")) {
      return false;
    }
  }

  diagnostics.staticAllocationBytes += diagnostics.scratchAllocationBytes - scratchBytesBefore;
  diagnostics.staticAllocationCount += diagnostics.scratchAllocationCount - scratchCountBefore;
  diagnostics.scratchAllocationBytes = scratchBytesBefore;
  diagnostics.scratchAllocationCount = scratchCountBefore;
  diagnostics.sharedScratchAllocationBytes = sharedBytesBefore;
  diagnostics.sharedScratchAllocationCount = sharedCountBefore;
  diagnostics.privateScratchAllocationBytes = privateBytesBefore;
  diagnostics.privateScratchAllocationCount = privateCountBefore;

  staticFilm.film = params.film;
  staticFilm.paper = params.paper;
  staticFilm.rgbToRawMethod = params.rgbToRawMethod;
  staticFilm.cameraUvFilterEnabled = params.cameraUvFilterEnabled;
  staticFilm.cameraUvCutNm = params.cameraUvCutNm;
  staticFilm.cameraIrFilterEnabled = params.cameraIrFilterEnabled;
  staticFilm.cameraIrCutNm = params.cameraIrCutNm;
  staticFilm.processNegativePrintTiming = params.printTiming;
  staticFilm.processNegativeFilterC = params.filterC;
  staticFilm.processNegativeFilterMShift = params.filterMShift;
  staticFilm.processNegativeFilterYShift = params.filterYShift;
  staticFilm.processNegativePreflashMFilterShift = params.preflashMFilterShift;
  staticFilm.processNegativePreflashYFilterShift = params.preflashYFilterShift;
  staticFilm.printScanResources = includePrintScanResources;
  staticFilm.grainResources = includeGrainResources;
  staticFilm.curves = curves;
  staticFilm.exposureCount = curves->exposureCount;
  staticFilm.wavelengthCount = curves->wavelengthCount;
  staticFilm.filmPositive = curves->type && std::strcmp(curves->type, "positive") == 0 ? 1u : 0u;
  staticFilm.hanatosWidth = needsHanatos ? hanatos.width : 0u;
  staticFilm.hanatosHeight = needsHanatos ? hanatos.height : 0u;
  staticFilm.filmDensityMaximum = densityCurveMaximums(*curves);
  if (includePrintScanResources) {
    staticFilm.paperCurves = paperCurves;
    staticFilm.paperExposureCount = paperCurves->exposureCount;
    staticFilm.paperDensityMaximum = densityCurveMaximums(*paperCurves);
  } else {
    staticFilm.paperCurves = nullptr;
    staticFilm.paperExposureCount = 0u;
    staticFilm.paperDensityMaximum = {0.0f, 0.0f, 0.0f};
  }
  return true;
}

bool VulkanRenderer::Impl::ensureCopyFrameResources() {
  if (copyFrame.descriptorPool == VK_NULL_HANDLE) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &copyFrame.descriptorPool);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDescriptorPool", result);
      return false;
    }

    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = copyFrame.descriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &copyDescriptorSetLayout;
    result = vkAllocateDescriptorSets(device, &setInfo, &copyFrame.descriptorSet);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkAllocateDescriptorSets", result);
      destroyCopyFrameResources();
      return false;
    }
  }

  if (copyFrame.commandBuffer == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo commandInfo{};
    commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandInfo.commandPool = commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(device, &commandInfo, &copyFrame.commandBuffer);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkAllocateCommandBuffers", result);
      return false;
    }
  }

  if (copyFrame.fence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &copyFrame.fence);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateFence", result);
      return false;
    }
  }

  return true;
}

bool VulkanRenderer::Impl::ensureCoreFrameResources() {
  if (coreFrame.descriptorPool == VK_NULL_HANDLE) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 31u * kCoreDescriptorSetCount;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = kCoreDescriptorSetCount;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &coreFrame.descriptorPool);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDescriptorPool", result);
      return false;
    }

    std::array<VkDescriptorSetLayout, kCoreDescriptorSetCount> setLayouts{};
    setLayouts.fill(coreDescriptorSetLayout);
    std::array<VkDescriptorSet, kCoreDescriptorSetCount> descriptorSets{};
    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = coreFrame.descriptorPool;
    setInfo.descriptorSetCount = kCoreDescriptorSetCount;
    setInfo.pSetLayouts = setLayouts.data();
    result = vkAllocateDescriptorSets(device, &setInfo, descriptorSets.data());
    if (result != VK_SUCCESS) {
      lastError = vkError("vkAllocateDescriptorSets", result);
      destroyCoreFrameResources();
      return false;
    }
    coreFrame.exposureDescriptorSet = descriptorSets[0];
    coreFrame.developDescriptorSet = descriptorSets[1];
    coreFrame.finalDescriptorSet = descriptorSets[2];
    for (uint32_t setIndex = 0; setIndex < kCoreHalationSetCount; ++setIndex) {
      coreFrame.halationDescriptorSets[setIndex] = descriptorSets[3u + setIndex];
    }
    uint32_t descriptorSetOffset = 3u + kCoreHalationSetCount;
    for (uint32_t setIndex = 0; setIndex < kCoreDiffusionSetCount; ++setIndex) {
      coreFrame.diffusionDescriptorSets[setIndex] = descriptorSets[descriptorSetOffset + setIndex];
    }
    descriptorSetOffset += kCoreDiffusionSetCount;
    for (uint32_t setIndex = 0; setIndex < kCorePrintDiffusionSetCount; ++setIndex) {
      coreFrame.printDiffusionDescriptorSets[setIndex] = descriptorSets[descriptorSetOffset + setIndex];
    }
    descriptorSetOffset += kCorePrintDiffusionSetCount;
    for (uint32_t setIndex = 0; setIndex < kCoreDirSetCount; ++setIndex) {
      coreFrame.dirDescriptorSets[setIndex] = descriptorSets[descriptorSetOffset + setIndex];
    }
    descriptorSetOffset += kCoreDirSetCount;
    for (uint32_t setIndex = 0; setIndex < kCoreScannerPostSetCount; ++setIndex) {
      coreFrame.scannerPostDescriptorSets[setIndex] = descriptorSets[descriptorSetOffset + setIndex];
    }
    descriptorSetOffset += kCoreScannerPostSetCount;
    for (uint32_t setIndex = 0; setIndex < kCoreGrainSetCount; ++setIndex) {
      coreFrame.grainDescriptorSets[setIndex] = descriptorSets[descriptorSetOffset + setIndex];
    }
  }

  if (coreFrame.formatDescriptorPool == VK_NULL_HANDLE) {
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VkResult result = vkCreateDescriptorPool(device, &poolInfo, nullptr, &coreFrame.formatDescriptorPool);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateDescriptorPool(format)", result);
      return false;
    }

    VkDescriptorSetAllocateInfo setInfo{};
    setInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    setInfo.descriptorPool = coreFrame.formatDescriptorPool;
    setInfo.descriptorSetCount = 1;
    setInfo.pSetLayouts = &copyDescriptorSetLayout;
    result = vkAllocateDescriptorSets(device, &setInfo, &coreFrame.destinationFormatDescriptorSet);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkAllocateDescriptorSets(format)", result);
      destroyCoreFrameResources();
      return false;
    }
  }

  if (coreFrame.commandBuffer == VK_NULL_HANDLE) {
    VkCommandBufferAllocateInfo commandInfo{};
    commandInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandInfo.commandPool = commandPool;
    commandInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandInfo.commandBufferCount = 1;
    VkResult result = vkAllocateCommandBuffers(device, &commandInfo, &coreFrame.commandBuffer);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkAllocateCommandBuffers", result);
      return false;
    }
  }

  if (coreFrame.fence == VK_NULL_HANDLE) {
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkResult result = vkCreateFence(device, &fenceInfo, nullptr, &coreFrame.fence);
    if (result != VK_SUCCESS) {
      lastError = vkError("vkCreateFence", result);
      return false;
    }
  }

  return true;
}

void VulkanRenderer::Impl::destroyCopyFrameResources() {
  if (copyFrame.fence != VK_NULL_HANDLE) {
    vkDestroyFence(device, copyFrame.fence, nullptr);
    copyFrame.fence = VK_NULL_HANDLE;
  }
  if (copyFrame.commandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device, commandPool, 1, &copyFrame.commandBuffer);
    copyFrame.commandBuffer = VK_NULL_HANDLE;
  }
  if (copyFrame.descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, copyFrame.descriptorPool, nullptr);
    copyFrame.descriptorPool = VK_NULL_HANDLE;
    copyFrame.descriptorSet = VK_NULL_HANDLE;
  }
  destroyScratchBuffer(copyFrame.destination);
  destroyScratchBuffer(copyFrame.source);
}

void VulkanRenderer::Impl::destroyCoreFrameResources() {
  if (coreFrame.fence != VK_NULL_HANDLE) {
    vkDestroyFence(device, coreFrame.fence, nullptr);
    coreFrame.fence = VK_NULL_HANDLE;
  }
  if (coreFrame.commandBuffer != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(device, commandPool, 1, &coreFrame.commandBuffer);
    coreFrame.commandBuffer = VK_NULL_HANDLE;
  }
  if (coreFrame.descriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, coreFrame.descriptorPool, nullptr);
    coreFrame.descriptorPool = VK_NULL_HANDLE;
    coreFrame.exposureDescriptorSet = VK_NULL_HANDLE;
    coreFrame.developDescriptorSet = VK_NULL_HANDLE;
    coreFrame.finalDescriptorSet = VK_NULL_HANDLE;
    coreFrame.halationDescriptorSets.fill(VK_NULL_HANDLE);
    coreFrame.diffusionDescriptorSets.fill(VK_NULL_HANDLE);
    coreFrame.printDiffusionDescriptorSets.fill(VK_NULL_HANDLE);
    coreFrame.dirDescriptorSets.fill(VK_NULL_HANDLE);
    coreFrame.scannerPostDescriptorSets.fill(VK_NULL_HANDLE);
    coreFrame.grainDescriptorSets.fill(VK_NULL_HANDLE);
  }
  if (coreFrame.formatDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device, coreFrame.formatDescriptorPool, nullptr);
    coreFrame.formatDescriptorPool = VK_NULL_HANDLE;
    coreFrame.destinationFormatDescriptorSet = VK_NULL_HANDLE;
  }
  destroyScratchBuffer(coreFrame.grainLayerB);
  destroyScratchBuffer(coreFrame.grainLayerA);
  destroyScratchBuffer(coreFrame.grainMicroB);
  destroyScratchBuffer(coreFrame.grainMicroA);
  destroyScratchBuffer(coreFrame.grainDensityB);
  destroyScratchBuffer(coreFrame.grainDensityA);
  destroyScratchBuffer(coreFrame.printGlareB);
  destroyScratchBuffer(coreFrame.printGlareA);
  destroyScratchBuffer(coreFrame.scannerPostC);
  destroyScratchBuffer(coreFrame.scannerPostB);
  destroyScratchBuffer(coreFrame.scannerPostA);
  destroyScratchBuffer(coreFrame.dirCorrectedDensityCurves);
  destroyScratchBuffer(coreFrame.dirFloats);
  destroyScratchBuffer(coreFrame.printDiffusionComponents);
  destroyScratchBuffer(coreFrame.printDiffusionInfo);
  destroyScratchBuffer(coreFrame.cameraDiffusionComponents);
  destroyScratchBuffer(coreFrame.cameraDiffusionInfo);
  destroyScratchBuffer(coreFrame.frameInts);
  destroyScratchBuffer(coreFrame.frameFloats);
  destroyScratchBuffer(coreFrame.filteredEnlargerResponse);
  destroyScratchBuffer(coreFrame.printDiffusionRaw);
  destroyScratchBuffer(coreFrame.printRaw);
  destroyScratchBuffer(coreFrame.dirDensity);
  destroyScratchBuffer(coreFrame.dirCorrectionC);
  destroyScratchBuffer(coreFrame.dirCorrectionB);
  destroyScratchBuffer(coreFrame.dirCorrectionA);
  destroyScratchBuffer(coreFrame.cameraDiffusionRaw);
  destroyScratchBuffer(coreFrame.diffusionDownsampleBlur);
  destroyScratchBuffer(coreFrame.diffusionDownsampleTemp);
  destroyScratchBuffer(coreFrame.diffusionDownsampleSource);
  destroyScratchBuffer(coreFrame.diffusionAccum);
  destroyScratchBuffer(coreFrame.diffusionTemp);
  destroyScratchBuffer(coreFrame.halationLogRaw);
  destroyScratchBuffer(coreFrame.tiledHalationBoostInfo);
  destroyScratchBuffer(coreFrame.halationBoostInfoReadback);
  destroyScratchBuffer(coreFrame.halationBoostInfo);
  destroyScratchBuffer(coreFrame.halationBoostChunks);
  destroyScratchBuffer(coreFrame.halationBoostedRaw);
  destroyScratchBuffer(coreFrame.halationRawD);
  destroyScratchBuffer(coreFrame.halationRawC);
  destroyScratchBuffer(coreFrame.halationRawB);
  destroyScratchBuffer(coreFrame.halationRawA);
  destroyScratchBuffer(coreFrame.destinationHalfStaging);
  destroyScratchBuffer(coreFrame.destinationStaging);
  destroyScratchBuffer(coreFrame.destinationHalf);
  destroyScratchBuffer(coreFrame.destination);
  destroyScratchBuffer(coreFrame.filmDensity);
  destroyScratchBuffer(coreFrame.filmRaw);
  destroyScratchBuffer(coreFrame.source);
  destroyScratchBuffer(coreFrame.sourceStaging);
}

void VulkanRenderer::Impl::releaseTransientResources() {
  std::lock_guard<std::recursive_mutex> lock(renderMutex);
  releaseTransientResourcesNoLock();
}

void VulkanRenderer::Impl::releaseTransientResourcesNoLock() {
  if (device == VK_NULL_HANDLE) {
    return;
  }
  destroyCoreFrameResources();
  destroyCopyFrameResources();
  refreshTransientBudgetEntry();
  updateSharedDiagnostics();
}

bool VulkanRenderer::Impl::renderCopyValidation(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  double
) {
  std::lock_guard<std::recursive_mutex> lock(renderMutex);
  diagnostics = {};
  lastError.clear();
  updateSharedDiagnostics();

  const PerfClock::time_point setupStart = PerfClock::now();
  const int32_t width = window.x2 - window.x1;
  const int32_t height = window.y2 - window.y1;
  if (width <= 0 || height <= 0) {
    return true;
  }
  if (!isSupportedRgba(source, destination)) {
    lastError = "Only RGBA 16-bit half and 32-bit float images are supported by the Windows Vulkan path.";
    return false;
  }
  if (source.bytesPerComponent != destination.bytesPerComponent) {
    lastError = "The Windows Vulkan copy-validation path requires matching source and destination pixel depths.";
    return false;
  }

  const int32_t pixelBytes = source.components * source.bytesPerComponent;
  const VkDeviceSize byteCount = static_cast<VkDeviceSize>(
    static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * static_cast<uint64_t>(pixelBytes)
  );
  const VkDeviceSize paddedByteCount = (byteCount + 3u) & ~VkDeviceSize(3u);
  if (paddedByteCount == 0) {
    return true;
  }

  if (byteCount > static_cast<VkDeviceSize>(std::numeric_limits<uint32_t>::max())) {
    lastError = "The Vulkan copy-validation render window is too large.";
    return false;
  }
  const uint32_t wordCount = static_cast<uint32_t>(paddedByteCount / sizeof(uint32_t));
  if (static_cast<VkDeviceSize>(wordCount) * sizeof(uint32_t) != paddedByteCount) {
    lastError = "The Vulkan copy-validation render window is too large.";
    return false;
  }

  diagnostics.uploadBytes = static_cast<uint64_t>(byteCount);
  diagnostics.threadgroupMode = "vulkan-256x1";
  diagnostics.passTimingMode = "cpu-fence";
  diagnostics.passCount = 1;
  diagnostics.privateScratchEnabled = false;

  if (!ensureCopyFrameResources() ||
      !ensureUploadScratchBuffer(copyFrame.source, paddedByteCount, "copy source") ||
      !ensureReadbackScratchBuffer(copyFrame.destination, paddedByteCount, "copy destination")) {
    return false;
  }

  if (!copyFrame.source.mapped) {
    lastError = "Vulkan copy source buffer is not mapped.";
    return false;
  }
  const PerfClock::time_point sourceCopyStart = PerfClock::now();
  if (!copyWindowToMappedBytes(source, window, width, height, copyFrame.source.mapped, static_cast<size_t>(byteCount))) {
    lastError = "The requested render window does not fit inside the Vulkan source image view.";
    return false;
  }
  if (paddedByteCount > byteCount) {
    std::memset(static_cast<uint8_t *>(copyFrame.source.mapped) + byteCount, 0, static_cast<size_t>(paddedByteCount - byteCount));
  }
  if (!flushMappedScratchBuffer(copyFrame.source, paddedByteCount, "copy source")) {
    return false;
  }
  const PerfClock::time_point sourceCopyEnd = PerfClock::now();
  diagnostics.sourceCopyMs = elapsedMilliseconds(sourceCopyStart, sourceCopyEnd);
  VkResult result = VK_SUCCESS;

  VkDescriptorBufferInfo sourceBufferInfo{};
  sourceBufferInfo.buffer = copyFrame.source.buffer;
  sourceBufferInfo.offset = 0;
  sourceBufferInfo.range = paddedByteCount;
  VkDescriptorBufferInfo destinationBufferInfo{};
  destinationBufferInfo.buffer = copyFrame.destination.buffer;
  destinationBufferInfo.offset = 0;
  destinationBufferInfo.range = paddedByteCount;

  VkWriteDescriptorSet writes[2]{};
  writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[0].dstSet = copyFrame.descriptorSet;
  writes[0].dstBinding = 0;
  writes[0].descriptorCount = 1;
  writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[0].pBufferInfo = &sourceBufferInfo;
  writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  writes[1].dstSet = copyFrame.descriptorSet;
  writes[1].dstBinding = 1;
  writes[1].descriptorCount = 1;
  writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  writes[1].pBufferInfo = &destinationBufferInfo;
  vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

  VkCommandBuffer commandBuffer = copyFrame.commandBuffer;
  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkResetCommandBuffer", result);
    return false;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkBeginCommandBuffer", result);
    return false;
  }

  VkMemoryBarrier hostInputBarrier{};
  hostInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostInputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  hostInputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0,
    1,
    &hostInputBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  const uint32_t pushConstants[2] = {wordCount, static_cast<uint32_t>(byteCount)};
  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, copyPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, copyPipelineLayout, 0, 1, &copyFrame.descriptorSet, 0, nullptr);
  vkCmdPushConstants(commandBuffer, copyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), pushConstants);
  vkCmdDispatch(commandBuffer, (wordCount + 255u) / 256u, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_HOST_BIT,
    0,
    1,
    &memoryBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkEndCommandBuffer", result);
    return false;
  }

  VkFence fence = copyFrame.fence;
  result = vkResetFences(device, 1, &fence);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkResetFences", result);
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  const PerfClock::time_point commandStart = PerfClock::now();
  diagnostics.cpuSetupMs = std::max(0.0, elapsedMilliseconds(setupStart, commandStart) - diagnostics.sourceCopyMs);
  result = backend ? backend->submit(queueIndex, submitInfo, fence) : VK_ERROR_INITIALIZATION_FAILED;
  if (result != VK_SUCCESS) {
    lastError = vkError("vkQueueSubmit", result);
    return false;
  }
  result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  const PerfClock::time_point commandEnd = PerfClock::now();
  if (result != VK_SUCCESS) {
    lastError = vkError("vkWaitForFences", result);
    return false;
  }
  diagnostics.commandBufferMs = elapsedMilliseconds(commandStart, commandEnd);

  const PerfClock::time_point outputCopyStart = PerfClock::now();
  if (!copyFrame.destination.mapped) {
    lastError = "Vulkan copy destination buffer is not mapped.";
    return false;
  }
  if (!invalidateMappedScratchBuffer(copyFrame.destination, paddedByteCount, "copy destination")) {
    return false;
  }
  if (!copyMappedBytesToWindow(copyFrame.destination.mapped, static_cast<size_t>(byteCount), destination, window, width, height)) {
    lastError = "The requested render window does not fit inside the Vulkan destination image view.";
    return false;
  }
  const PerfClock::time_point outputCopyEnd = PerfClock::now();
  diagnostics.outputCopyMs = elapsedMilliseconds(outputCopyStart, outputCopyEnd);

  RendererPassDiagnostics pass{};
  pass.name = "copyValidation";
  pass.width = static_cast<uint32_t>(width);
  pass.height = static_cast<uint32_t>(height);
  pass.threadgroupWidth = 256;
  pass.threadgroupHeight = 1;
  pass.estimatedBytes = static_cast<uint64_t>(byteCount) * 2u;
  diagnostics.passes.push_back(pass);

  refreshTransientBudgetEntry();
  enforceTransientBudget();
  updateSharedDiagnostics();
  return true;
}

bool VulkanRenderer::Impl::renderCoreBootstrap(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  const RenderParams &params,
  double time
) {
  std::lock_guard<std::recursive_mutex> lock(renderMutex);
  diagnostics = {};
  diagnostics.renderSerialized = true;
  lastError.clear();
  updateSharedDiagnostics();

  const PerfClock::time_point setupStart = PerfClock::now();
  const int32_t width = window.x2 - window.x1;
  const int32_t height = window.y2 - window.y1;
  if (width <= 0 || height <= 0) {
    return true;
  }
  if (!isSupportedRgba(source, destination)) {
    lastError = "Only RGBA 16-bit half and 32-bit float images are supported by the Windows Vulkan path.";
    return false;
  }

  const uint64_t pixelCount = static_cast<uint64_t>(width) * static_cast<uint64_t>(height);
  if (pixelCount > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    lastError = "The Vulkan core render window is too large.";
    return false;
  }
  const uint64_t byteCount64 = pixelCount * 4u * sizeof(float);
  const uint64_t halfByteCount64 = pixelCount * 4u * sizeof(uint16_t);
  if (pixelCount == 0 || byteCount64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max())) {
    lastError = "The Vulkan core render window is too large.";
    return false;
  }
  const uint64_t grainLayerByteCount64 = pixelCount * 9u * sizeof(float);
  if (grainLayerByteCount64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max())) {
    lastError = "The Vulkan grain render window is too large.";
    return false;
  }
  const VkDeviceSize byteCount = static_cast<VkDeviceSize>(byteCount64);
  const VkDeviceSize halfByteCount = static_cast<VkDeviceSize>(halfByteCount64);
  const VkDeviceSize grainLayerByteCount = static_cast<VkDeviceSize>(grainLayerByteCount64);
  const bool destinationIsHalf = destination.bytesPerComponent == 2;
  const uint32_t coordinateWidth =
    activeTileContext.enabled ? std::max(activeTileContext.fullWidth, 1u) : static_cast<uint32_t>(width);
  const uint32_t coordinateHeight =
    activeTileContext.enabled ? std::max(activeTileContext.fullHeight, 1u) : static_cast<uint32_t>(height);
  const bool fullFrameSourcePath = activeTileContext.enabled && activeTileContext.fullFrameSource;
  const uint64_t sourcePixelCount = fullFrameSourcePath
    ? static_cast<uint64_t>(coordinateWidth) * static_cast<uint64_t>(coordinateHeight)
    : pixelCount;
  const uint64_t sourceByteCount64 = sourcePixelCount * 4u * sizeof(float);
  if (sourcePixelCount == 0u || sourceByteCount64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max())) {
    lastError = "The Vulkan source render window is too large.";
    return false;
  }
  const VkDeviceSize sourceByteCount = static_cast<VkDeviceSize>(sourceByteCount64);
  const uint32_t halationBoostMaxChunkCount = static_cast<uint32_t>(
    (pixelCount + kHalationBoostMaxChunkPixels - 1u) / kHalationBoostMaxChunkPixels
  );
  const VkDeviceSize halationBoostChunkByteCount = static_cast<VkDeviceSize>(
    std::max<uint64_t>(halationBoostMaxChunkCount, 1u) * 4u * sizeof(float)
  );
  const float filmPixelSizeUm = filmFormatLongEdgeMm(params.filmFormat) * 1000.0f /
    static_cast<float>(std::max(coordinateWidth, coordinateHeight)) /
    resolvedEnlargerScale(params);
  const bool finalOutput = params.renderOutput == RenderOutputMode::FinalPreview;
  const bool rcmOutput = finalOutput && params.outputRole == OutputRole::Rcm;
  const bool finalPrintSimulation =
    finalOutput && params.process == ProcessMode::PrintSimulation;
  const bool finalScanNegative =
    finalOutput && params.process == ProcessMode::ScanNegative;
  const bool finalProcessNegative =
    finalOutput && params.process == ProcessMode::ProcessNegative;
  const bool printLikeFinalPath = finalPrintSimulation || finalProcessNegative;
  const bool finalPostProcessPath = finalPrintSimulation || finalScanNegative || finalProcessNegative;
  const bool printScanPassEnabled = envFlagEnabledOrDefault(
    "SPEKTRAFILM_VULKAN_PRINT_SCAN_PASS",
    finalPostProcessPath
  );
  const bool scannerPostFeatureEnabled =
    envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_SCANNER_POST_PASS", true);
  const bool printGlarePath =
    scannerPostFeatureEnabled &&
    printScanPassEnabled &&
    printLikeFinalPath &&
    !rcmOutput &&
    params.scannerEnabled &&
    params.glarePercent > 0.0f;
  const bool printGlareBlurPath = printGlarePath && params.glareBlur > 0.0f;
  const bool scannerBlurPath =
    scannerPostFeatureEnabled &&
    printScanPassEnabled &&
    finalPostProcessPath &&
    !rcmOutput &&
    params.scannerEnabled &&
    params.scannerMtf50LpMm > 0.0f;
  const bool scannerUnsharpPath =
    scannerPostFeatureEnabled &&
    printScanPassEnabled &&
    finalPostProcessPath &&
    !rcmOutput &&
    params.scannerEnabled &&
    params.scannerUnsharpRadiusUm > 0.0f &&
    params.scannerUnsharpAmount > 0.0f;
  const bool scannerPostPath = printGlarePath || scannerBlurPath || scannerUnsharpPath;
  const bool grainFeatureEnabled =
    envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_GRAIN_PASS", true) && params.grainEnabled && !finalProcessNegative;
  const bool grainSynthesisRequested =
    grainFeatureEnabled &&
    params.grainModel == GrainModel::GrainSynthesis;
  const bool previewGrainPath =
    grainFeatureEnabled &&
    params.grainModel == GrainModel::Preview;
  const bool productionGrainPath =
    grainFeatureEnabled &&
    params.grainModel == GrainModel::Production;
  const bool grainSynthesisPath = grainSynthesisRequested;
  const bool grainPath = previewGrainPath || productionGrainPath || grainSynthesisPath;
  const bool diffusionFeatureEnabled =
    envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_DIFFUSION_PASS", true);
  VulkanDiffusionInfo cameraDiffusionInfo{};
  std::vector<VulkanDiffusionComponent> cameraDiffusionComponents;
  if (diffusionFeatureEnabled &&
      !finalProcessNegative &&
      params.cameraDiffusionEnabled &&
      params.cameraDiffusionStrength > 0.0f &&
      params.cameraDiffusionSpatialScale > 0.0f) {
    cameraDiffusionComponents = makeDiffusionComponents(
      {params.cameraDiffusionFamily, params.cameraDiffusionStrength, params.cameraDiffusionSpatialScale,
       params.cameraDiffusionHaloWarmth, params.cameraDiffusionCoreIntensity, params.cameraDiffusionCoreSize,
       params.cameraDiffusionHaloIntensity, params.cameraDiffusionHaloSize, params.cameraDiffusionBloomIntensity,
       params.cameraDiffusionBloomSize},
      filmPixelSizeUm,
      cameraDiffusionInfo,
      diffusionClusterSigmaRatio
    );
  }
  const bool cameraDiffusionPath =
    cameraDiffusionInfo.componentCount > 0u &&
    !cameraDiffusionComponents.empty();
  VulkanDiffusionInfo printDiffusionInfo{};
  std::vector<VulkanDiffusionComponent> printDiffusionComponents;
  if (diffusionFeatureEnabled &&
      printScanPassEnabled &&
      printLikeFinalPath &&
      params.printDiffusionEnabled &&
      params.printDiffusionStrength > 0.0f &&
      params.printDiffusionSpatialScale > 0.0f) {
    printDiffusionComponents = makeDiffusionComponents(
      {params.printDiffusionFamily, params.printDiffusionStrength, params.printDiffusionSpatialScale,
       params.printDiffusionHaloWarmth, params.printDiffusionCoreIntensity, params.printDiffusionCoreSize,
       params.printDiffusionHaloIntensity, params.printDiffusionHaloSize, params.printDiffusionBloomIntensity,
       params.printDiffusionBloomSize},
      filmPixelSizeUm,
      printDiffusionInfo,
      diffusionClusterSigmaRatio
    );
  }
  const bool printDiffusionPath =
    printDiffusionInfo.componentCount > 0u &&
    !printDiffusionComponents.empty();
  const bool diffusionDownsamplePath = blurDownsample != "off" &&
    (anyDiffusionComponentDownsamples(cameraDiffusionComponents, blurDownsample) ||
     anyDiffusionComponentDownsamples(printDiffusionComponents, blurDownsample));
  const bool dirFeatureEnabled = envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_DIR_PASS", true);
  const bool dirPath = dirFeatureEnabled && !finalProcessNegative && params.dirCouplersAmount > 0.0f;
  const bool dirBlurPath = dirPath && params.dirCouplersDiffusionUm > 0.0f;
  const bool dirTailPath =
    dirBlurPath &&
    params.dirCouplersDiffusionTailUm > 0.0f &&
    params.dirCouplersDiffusionTailWeight > 0.0f;
  const VkDeviceSize pixelStorageByteCount = byteCount;
  const VkDeviceSize diffusionTempByteCount =
    pixelStorageByteCount * static_cast<VkDeviceSize>(std::max(diffusionGroupSize, 1u));
  const uint32_t maxDiffusionDownsampleWidth = alignedReducedDimension(
    static_cast<uint32_t>(width),
    activeTileContext.enabled ? activeTileContext.tileOriginX : 0u,
    coordinateWidth,
    2u
  );
  const uint32_t maxDiffusionDownsampleHeight = alignedReducedDimension(
    static_cast<uint32_t>(height),
    activeTileContext.enabled ? activeTileContext.tileOriginY : 0u,
    coordinateHeight,
    2u
  );
  const uint64_t diffusionDownsamplePixelCount =
    static_cast<uint64_t>(maxDiffusionDownsampleWidth) * static_cast<uint64_t>(maxDiffusionDownsampleHeight);
  const VkDeviceSize diffusionDownsampleByteCount =
    static_cast<VkDeviceSize>(diffusionDownsamplePixelCount * 4u * sizeof(float));
  const VkDeviceSize diffusionDownsampleGroupByteCount =
    diffusionDownsampleByteCount * static_cast<VkDeviceSize>(std::max(diffusionGroupSize, 1u));
  const bool destinationFormatConvert = destinationIsHalf;
  const bool halationFeatureEnabled =
    envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_HALATION_PASS", true) && params.halationEnabled && !finalProcessNegative;
  const bool halationScatterEnabled =
    halationFeatureEnabled &&
    params.scatterAmount > 0.0f &&
    params.scatterScale > 0.0f;
  const bool halationBounceEnabled =
    halationFeatureEnabled &&
    params.halationAmount > 0.0f &&
    params.halationScale > 0.0f &&
    (params.halationStrengthR > 0.0f || params.halationStrengthG > 0.0f || params.halationStrengthB > 0.0f);
  const bool halationBoostEnabled =
    halationFeatureEnabled &&
    params.halationBoostEv > 0.0f;
  const bool halationBoostMilestoneEnabled =
    halationBoostEnabled &&
    activeTileContext.enabled &&
    activeTileContext.halationBoostMilestoneEnabled;
  const bool halationBoostLocalReductionEnabled =
    halationBoostEnabled && !halationBoostMilestoneEnabled;
  const bool halationPassEnabled = halationBoostEnabled || halationScatterEnabled || halationBounceEnabled;
  const uint32_t halationBoostPassCount = halationBoostEnabled
    ? (halationBoostMilestoneEnabled ? 1u : 3u)
    : 0u;
  const uint32_t halationExtraPassCount = halationPassEnabled
    ? halationBoostPassCount + (halationScatterEnabled ? 10u : 0u) +
        (halationBounceEnabled ? 8u : 1u)
    : 0u;
  auto diffusionBlurPassCount = [&](const std::vector<VulkanDiffusionComponent> &components) {
    uint32_t passCount = 0u;
    for (uint32_t component = 0u; component < static_cast<uint32_t>(components.size());) {
      const uint32_t downsampleScale = diffusionDownsamplePath
        ? diffusionDownsampleScaleForSigma(blurDownsample, components[component].sigmaPx)
        : 1u;
      uint32_t groupCount = 1u;
      while (component + groupCount < static_cast<uint32_t>(components.size()) &&
             groupCount < diffusionGroupSize &&
             (!diffusionDownsamplePath ||
              diffusionDownsampleScaleForSigma(blurDownsample, components[component + groupCount].sigmaPx) == downsampleScale)) {
        ++groupCount;
      }
      passCount += downsampleScale > 1u ? 4u : 2u;
      component += groupCount;
    }
    return passCount;
  };
  const uint32_t cameraDiffusionExtraPassCount = cameraDiffusionPath
    ? diffusionBlurPassCount(cameraDiffusionComponents) + 2u + (halationPassEnabled ? 0u : 1u)
    : 0u;
  const uint32_t printDiffusionExtraPassCount = printDiffusionPath
    ? diffusionBlurPassCount(printDiffusionComponents) + 4u
    : 0u;
  const uint32_t dirExtraPassCount = dirPath
    ? 2u + (dirBlurPath ? 2u : 0u) + (dirTailPath ? 7u : 0u)
    : 0u;
  const uint32_t scannerPostExtraPassCount = scannerPostPath
    ? (printGlarePath ? 2u + (printGlareBlurPath ? 2u : 0u) : 0u) +
      (scannerBlurPath ? 2u : 0u) +
      (scannerUnsharpPath ? 2u : 0u) +
      1u
    : 0u;
  const uint32_t grainExtraPassCount =
    previewGrainPath ? 1u : ((productionGrainPath || grainSynthesisPath) ? 10u : 0u);
  const uint32_t cameraDiffusionRadius = cameraDiffusionPath ? kVulkanSpatialEffectRadiusPx : 0u;
  const uint32_t halationRadius =
    (halationScatterEnabled || halationBounceEnabled) ? kVulkanSpatialEffectRadiusPx : 0u;
  const uint32_t dirRadius = dirBlurPath ? kVulkanSpatialEffectRadiusPx : 0u;
  const uint32_t grainRadius =
    (productionGrainPath || grainSynthesisPath) ? kVulkanGrainSpatialRadiusPx : 0u;
  const uint32_t printDiffusionRadius = printDiffusionPath ? kVulkanSpatialEffectRadiusPx : 0u;
  const uint32_t scannerPostRadius =
    (printGlareBlurPath || scannerBlurPath || scannerUnsharpPath) ? kVulkanSpatialEffectRadiusPx : 0u;
  const bool activeRectShrinkEnabled = activeTileContext.enabled;
  const bool frameParamsEnabled = printScanPassEnabled || halationPassEnabled || scannerPostPath || grainPath;
  const bool preExposureRawPath = cameraDiffusionPath || halationPassEnabled;
  const bool filmDensityIntermediateEnabled = printScanPassEnabled || dirPath || grainPath;

  const float effectiveFilmGamma = params.filmGamma *
    (params.filmPushPullMode == PushPullMode::Standard ? filmPushPullGamma(params.filmPushPullStops) : 1.0f);
  const float effectivePrintGamma = params.printGamma * printPushPullGamma(params.printPushPullStops);

  diagnostics.uploadBytes = fullFrameSourcePath ? 0u : static_cast<uint64_t>(byteCount);
  diagnostics.diffusionGroupSize = diffusionGroupSize;
  diagnostics.threadgroupMode = threadgroupMode;
  diagnostics.passTimingMode = "cpu-fence";
  diagnostics.blurBackend = blurBackend;
  diagnostics.blurDownsample = blurDownsample;
  diagnostics.intermediatePrecision = intermediatePrecision;
  diagnostics.diffusionClusterSigma = diffusionClusterSigma;
  diagnostics.dirTailBackend = dirTailBackend;
  diagnostics.halationGroupedTail = halationGroupedTail;
  diagnostics.scannerMps = scannerMps;
  diagnostics.grainBlurRecurrence = grainBlurRecurrence;
  diagnostics.passCount =
    2u +
    cameraDiffusionExtraPassCount +
    halationExtraPassCount +
    dirExtraPassCount +
    grainExtraPassCount +
    (printScanPassEnabled ? 1u + (printDiffusionPath ? printDiffusionExtraPassCount : 1u) : 0u) +
    scannerPostExtraPassCount;
  diagnostics.privateScratchEnabled = false;
  diagnostics.halationPath = halationPassEnabled;
  diagnostics.cameraDiffusionPath = cameraDiffusionPath;
  diagnostics.printDiffusionPath = printDiffusionPath;
  diagnostics.dirPath = dirPath;
  diagnostics.productionGrainPath = productionGrainPath;
  diagnostics.grainSynthesisPath = grainSynthesisPath;
  diagnostics.finalPostProcessPath = printScanPassEnabled;

  if (!prepareStaticFilmResources(params, printScanPassEnabled, productionGrainPath || grainSynthesisPath) ||
      !ensureCoreFrameResources() ||
      !ensurePrivateScratchBuffer(coreFrame.source, sourceByteCount, fullFrameSourcePath ? "core full-frame source" : "core source") ||
      !ensurePrivateScratchBuffer(coreFrame.filmRaw, pixelStorageByteCount, "core film raw") ||
      !ensurePrivateScratchBuffer(coreFrame.destination, pixelStorageByteCount, "core destination")) {
    return false;
  }
  const uint32_t formatGroups = static_cast<uint32_t>((pixelCount + 255u) / 256u);
  if (!fullFrameSourcePath &&
      !ensureUploadScratchBuffer(
        coreFrame.sourceStaging,
        byteCount,
        "core source staging",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
      )) {
    return false;
  }
  if (destinationIsHalf) {
    if (!ensurePrivateScratchBuffer(coreFrame.destinationHalf, halfByteCount, "core destination half") ||
        !ensureReadbackScratchBuffer(
          coreFrame.destinationHalfStaging,
          halfByteCount,
          "core destination half staging",
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )) {
      return false;
    }
  } else if (!ensureReadbackScratchBuffer(
               coreFrame.destinationStaging,
               byteCount,
               "core destination staging",
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
             )) {
    return false;
  }
  if (filmDensityIntermediateEnabled &&
      !ensurePrivateScratchBuffer(coreFrame.filmDensity, pixelStorageByteCount, "core film density")) {
    return false;
  }
  if (halationPassEnabled &&
      (!ensurePrivateScratchBuffer(coreFrame.halationRawA, pixelStorageByteCount, "core halation raw A") ||
       !ensurePrivateScratchBuffer(coreFrame.halationRawB, pixelStorageByteCount, "core halation raw B") ||
       !ensurePrivateScratchBuffer(coreFrame.halationRawC, pixelStorageByteCount, "core halation raw C") ||
       !ensurePrivateScratchBuffer(coreFrame.halationRawD, pixelStorageByteCount, "core halation raw D") ||
       !ensurePrivateScratchBuffer(coreFrame.halationLogRaw, pixelStorageByteCount, "core halation log raw"))) {
    return false;
  }
  if (halationBoostEnabled) {
    if (!ensurePrivateScratchBuffer(coreFrame.halationBoostedRaw, pixelStorageByteCount, "core halation boosted raw")) {
      return false;
    }
    if (halationBoostLocalReductionEnabled &&
        (!ensurePrivateScratchBuffer(coreFrame.halationBoostChunks, halationBoostChunkByteCount, "core halation boost chunks") ||
         !ensurePrivateScratchBuffer(coreFrame.halationBoostInfo, 4u * sizeof(float), "core halation boost info"))) {
      return false;
    }
    if (halationBoostMilestoneEnabled && coreFrame.tiledHalationBoostInfo.buffer == VK_NULL_HANDLE) {
      lastError = "Vulkan tiled halation boost milestone info is unavailable.";
      return false;
    }
  }
  if ((cameraDiffusionPath || printDiffusionPath) &&
      (!ensurePrivateScratchBuffer(coreFrame.diffusionTemp, diffusionTempByteCount, "core diffusion temp") ||
       !ensurePrivateScratchBuffer(coreFrame.diffusionAccum, pixelStorageByteCount, "core diffusion accumulator"))) {
    return false;
  }
  if (diffusionDownsamplePath &&
      (!ensurePrivateScratchBuffer(coreFrame.diffusionDownsampleSource, diffusionDownsampleByteCount, "core diffusion downsample source") ||
       !ensurePrivateScratchBuffer(coreFrame.diffusionDownsampleTemp, diffusionDownsampleGroupByteCount, "core diffusion downsample temp") ||
       !ensurePrivateScratchBuffer(coreFrame.diffusionDownsampleBlur, diffusionDownsampleGroupByteCount, "core diffusion downsample blur"))) {
    return false;
  }
  if (cameraDiffusionPath &&
      (!ensurePrivateScratchBuffer(coreFrame.cameraDiffusionRaw, pixelStorageByteCount, "core camera diffusion raw") ||
       (!halationPassEnabled && !ensurePrivateScratchBuffer(coreFrame.halationLogRaw, pixelStorageByteCount, "core diffusion log raw")))) {
    return false;
  }
  if (printDiffusionPath &&
      (!ensurePrivateScratchBuffer(coreFrame.printRaw, pixelStorageByteCount, "core print raw") ||
       !ensurePrivateScratchBuffer(coreFrame.printDiffusionRaw, pixelStorageByteCount, "core print diffusion raw"))) {
    return false;
  }
  if (dirPath &&
      (!ensurePrivateScratchBuffer(coreFrame.dirCorrectionA, pixelStorageByteCount, "core DIR correction A") ||
       !ensurePrivateScratchBuffer(coreFrame.dirCorrectionB, pixelStorageByteCount, "core DIR correction B") ||
       !ensurePrivateScratchBuffer(coreFrame.dirCorrectionC, pixelStorageByteCount, "core DIR correction C") ||
       !ensurePrivateScratchBuffer(coreFrame.dirDensity, pixelStorageByteCount, "core DIR density"))) {
    return false;
  }
  if (scannerPostPath &&
      (!ensurePrivateScratchBuffer(coreFrame.scannerPostA, pixelStorageByteCount, "core scanner post A") ||
       !ensurePrivateScratchBuffer(coreFrame.scannerPostB, pixelStorageByteCount, "core scanner post B") ||
       !ensurePrivateScratchBuffer(coreFrame.scannerPostC, pixelStorageByteCount, "core scanner post C"))) {
    return false;
  }
  if (printGlarePath &&
      (!ensurePrivateScratchBuffer(coreFrame.printGlareA, pixelStorageByteCount, "core print glare A") ||
       !ensurePrivateScratchBuffer(coreFrame.printGlareB, pixelStorageByteCount, "core print glare B"))) {
    return false;
  }
  if (grainPath &&
      (!ensurePrivateScratchBuffer(coreFrame.grainDensityA, pixelStorageByteCount, "core grain density A") ||
       !ensurePrivateScratchBuffer(coreFrame.grainDensityB, pixelStorageByteCount, "core grain density B"))) {
    return false;
  }
  if ((productionGrainPath || grainSynthesisPath) &&
      (!ensurePrivateScratchBuffer(coreFrame.grainMicroA, pixelStorageByteCount, "core grain micro A") ||
       !ensurePrivateScratchBuffer(coreFrame.grainMicroB, pixelStorageByteCount, "core grain micro B") ||
       !ensurePrivateScratchBuffer(coreFrame.grainLayerA, grainLayerByteCount, "core grain layer A") ||
       !ensurePrivateScratchBuffer(coreFrame.grainLayerB, grainLayerByteCount, "core grain layer B"))) {
    return false;
  }
  if (frameParamsEnabled &&
      (!ensureSharedScratchBuffer(coreFrame.frameFloats, kCoreFrameFloatCount * sizeof(float), "core frame float params") ||
       !ensureSharedScratchBuffer(coreFrame.frameInts, kCoreFrameIntCount * sizeof(uint32_t), "core frame int params"))) {
    return false;
  }
  if (printScanPassEnabled &&
      !ensurePrivateScratchBuffer(
        coreFrame.filteredEnlargerResponse,
        static_cast<VkDeviceSize>(staticFilm.wavelengthCount) * 8u * sizeof(float),
        "filtered enlarger response"
      )) {
    return false;
  }
  diagnostics.privateScratchEnabled = preferPrivateScratch;

  float autoExposureEv = 0.0f;
  const PerfClock::time_point sourceCopyStart = PerfClock::now();
  if (!fullFrameSourcePath) {
    if (!coreFrame.sourceStaging.mapped) {
      lastError = "Vulkan source staging buffer is not mapped.";
      return false;
    }
    if (params.autoExposure) {
      std::vector<float> sourcePixels(static_cast<size_t>(pixelCount) * 4u);
      if (!copySourceToFloatStaging(source, window, width, height, sourcePixels.data())) {
        lastError = "The requested render window does not fit inside the Vulkan source image view.";
        return false;
      }
      autoExposureEv = measureAutoExposureEv(sourcePixels.data(), width, height, params);
      std::memcpy(coreFrame.sourceStaging.mapped, sourcePixels.data(), static_cast<size_t>(byteCount));
    } else if (!copySourceToFloatStaging(
                 source,
                 window,
                 width,
                 height,
                 static_cast<float *>(coreFrame.sourceStaging.mapped)
               )) {
      lastError = "The requested render window does not fit inside the Vulkan source image view.";
      return false;
    }
    if (!flushMappedScratchBuffer(coreFrame.sourceStaging, byteCount, "core source staging")) {
      return false;
    }
  }
  const PerfClock::time_point sourceCopyEnd = PerfClock::now();
  diagnostics.sourceCopyMs = elapsedMilliseconds(sourceCopyStart, sourceCopyEnd);
  VkResult result = VK_SUCCESS;

  if (cameraDiffusionPath &&
      (!uploadScratchBuffer(coreFrame.cameraDiffusionInfo, &cameraDiffusionInfo, sizeof(cameraDiffusionInfo), "camera diffusion info") ||
       !uploadScratchBuffer(
         coreFrame.cameraDiffusionComponents,
         cameraDiffusionComponents.data(),
         static_cast<VkDeviceSize>(cameraDiffusionComponents.size() * sizeof(VulkanDiffusionComponent)),
         "camera diffusion components"
       ))) {
    return false;
  }
  if (printDiffusionPath &&
      (!uploadScratchBuffer(coreFrame.printDiffusionInfo, &printDiffusionInfo, sizeof(printDiffusionInfo), "print diffusion info") ||
       !uploadScratchBuffer(
         coreFrame.printDiffusionComponents,
         printDiffusionComponents.data(),
         static_cast<VkDeviceSize>(printDiffusionComponents.size() * sizeof(VulkanDiffusionComponent)),
         "print diffusion components"
       ))) {
    return false;
  }
  if (dirPath) {
    if (!staticFilm.curves || !staticFilm.curves->densityCurves || !staticFilm.curves->logExposure) {
      lastError = "Unable to locate generated film density curves for DIR correction.";
      return false;
    }
    const std::array<float, kDirFloatCount> dirFloats = makeDirFloatParams(*staticFilm.curves, params, filmPixelSizeUm);
    const std::vector<float> correctedDensityCurves = makeDirCorrectedDensityCurves(*staticFilm.curves, dirFloats);
    if (!uploadScratchBuffer(coreFrame.dirFloats, dirFloats.data(), dirFloats.size() * sizeof(float), "DIR params") ||
        !uploadScratchBuffer(
          coreFrame.dirCorrectedDensityCurves,
          correctedDensityCurves.data(),
          static_cast<VkDeviceSize>(correctedDensityCurves.size() * sizeof(float)),
          "DIR corrected density curves"
        )) {
      return false;
    }
  }

  if (frameParamsEnabled) {
    std::array<float, kCoreFrameFloatCount> frameFloats{};
    frameFloats[0] = params.filterC;
    frameFloats[1] = params.filterMShift;
    frameFloats[2] = params.filterYShift;
    frameFloats[3] = params.printExposureEv;
    frameFloats[4] = effectivePrintGamma;
    frameFloats[5] = params.printShadowShape;
    frameFloats[6] = params.printHighlightShape;
    frameFloats[7] = params.negativeBleachBypassAmount;
    frameFloats[8] = params.negativeLeucoCyanCoupling;
    frameFloats[9] = params.printBleachBypassAmount;
    frameFloats[10] = params.preflashExposure;
    frameFloats[11] = params.preflashMFilterShift;
    frameFloats[12] = params.preflashYFilterShift;
    frameFloats[13] = params.printerLightsR;
    frameFloats[14] = params.printerLightsG;
    frameFloats[15] = params.printerLightsB;
    frameFloats[16] = staticFilm.filmDensityMaximum[0];
    frameFloats[17] = staticFilm.filmDensityMaximum[1];
    frameFloats[18] = staticFilm.filmDensityMaximum[2];
    frameFloats[19] = staticFilm.paperDensityMaximum[0];
    frameFloats[20] = staticFilm.paperDensityMaximum[1];
    frameFloats[21] = staticFilm.paperDensityMaximum[2];
    frameFloats[22] = colorEncodeLutMin();
    frameFloats[23] = colorEncodeLutMax();
    frameFloats[24] = params.scannerWhiteLevel;
    frameFloats[25] = params.scannerBlackLevel;
    frameFloats[26] = staticFilm.curves && staticFilm.curves->densityCurveMinimum ? staticFilm.curves->densityCurveMinimum[0] : 0.0f;
    frameFloats[27] = staticFilm.curves && staticFilm.curves->densityCurveMinimum ? staticFilm.curves->densityCurveMinimum[1] : 0.0f;
    frameFloats[28] = staticFilm.curves && staticFilm.curves->densityCurveMinimum ? staticFilm.curves->densityCurveMinimum[2] : 0.0f;
    float halationStrengthR = params.halationStrengthR;
    float halationStrengthG = params.halationStrengthG;
    float halationStrengthB = params.halationStrengthB;
    if (staticFilm.curves && staticFilm.curves->halationStrength && isDefaultHalationStrength(params)) {
      halationStrengthR = staticFilm.curves->halationStrength[0];
      halationStrengthG = staticFilm.curves->halationStrength[1];
      halationStrengthB = staticFilm.curves->halationStrength[2];
    }
    float halationFirstSigmaR = params.halationFirstSigmaUmR;
    float halationFirstSigmaG = params.halationFirstSigmaUmG;
    float halationFirstSigmaB = params.halationFirstSigmaUmB;
    if (staticFilm.curves && staticFilm.curves->halationFirstSigmaUm) {
      halationFirstSigmaR = staticFilm.curves->halationFirstSigmaUm[0];
      halationFirstSigmaG = staticFilm.curves->halationFirstSigmaUm[1];
      halationFirstSigmaB = staticFilm.curves->halationFirstSigmaUm[2];
    }
    frameFloats[29] = filmPixelSizeUm;
    frameFloats[30] = params.scatterAmount;
    frameFloats[31] = params.scatterScale;
    frameFloats[32] = params.halationAmount;
    frameFloats[33] = params.halationScale;
    frameFloats[34] = halationStrengthR;
    frameFloats[35] = halationStrengthG;
    frameFloats[36] = halationStrengthB;
    frameFloats[37] = halationFirstSigmaR;
    frameFloats[38] = halationFirstSigmaG;
    frameFloats[39] = halationFirstSigmaB;
    frameFloats[40] = scannerSigmaUmFromMtf50(params.scannerMtf50LpMm) / std::max(filmPixelSizeUm, 1.0e-6f);
    frameFloats[41] = std::max(params.scannerUnsharpRadiusUm, 0.0f) / std::max(filmPixelSizeUm, 1.0e-6f);
    frameFloats[42] = params.scannerUnsharpAmount;
    frameFloats[43] = params.glarePercent;
    frameFloats[44] = params.glareRoughness;
    frameFloats[45] = params.glareBlur;
    const std::array<float, 3> glareRgb = scanIlluminantToOutputRgb(staticFilm.paperCurves, params);
    frameFloats[46] = glareRgb[0];
    frameFloats[47] = glareRgb[1];
    frameFloats[48] = glareRgb[2];
    frameFloats[49] = params.grainAmount;
    frameFloats[50] = params.grainSaturation;
    frameFloats[51] = params.grainParticleAreaUm2;
    frameFloats[52] = params.grainParticleScaleR;
    frameFloats[53] = params.grainParticleScaleG;
    frameFloats[54] = params.grainParticleScaleB;
    frameFloats[55] = params.grainParticleScaleLayer0;
    frameFloats[56] = params.grainParticleScaleLayer1;
    frameFloats[57] = params.grainParticleScaleLayer2;
    frameFloats[58] = params.grainDensityMinR;
    frameFloats[59] = params.grainDensityMinG;
    frameFloats[60] = params.grainDensityMinB;
    frameFloats[61] = params.grainUniformityR;
    frameFloats[62] = params.grainUniformityG;
    frameFloats[63] = params.grainUniformityB;
    const float grainFormatScale = std::pow(
      std::max(filmFormatLongEdgeMm(params.filmFormat) / 35.0f, 1.0e-6f),
      0.62f
    );
    frameFloats[64] = std::max(params.grainFinalBlurUm, 0.0f) * grainFormatScale /
      std::max(filmPixelSizeUm, 1.0e-6f);
    frameFloats[65] = params.grainBlurDyeCloudsUm;
    frameFloats[66] = params.grainMicroStructureScale;
    frameFloats[67] = params.grainMicroStructureSigmaNm * 0.001f / std::max(filmPixelSizeUm, 1.0e-6f);
    frameFloats[68] = resolvedEnlargerScale(params);
    frameFloats[69] = params.enlargerOffsetXPercent;
    frameFloats[70] = params.enlargerOffsetYPercent;
    frameFloats[71] = filmPixelSizeUm;
    frameFloats[72] = params.hdrReferenceWhiteNits;
    frameFloats[73] = params.hdrPeakNits;
    frameFloats[74] = params.hdrExposureEv;
    frameFloats[75] = params.halationBoostEv;
    frameFloats[76] = params.halationBoostRange;
    frameFloats[77] = params.halationProtectEv;
    const float grainSynthesisQuality = std::clamp(params.grainSynthesisQuality, 0.25f, 4.0f);
    const float grainSynthesisSize = std::clamp(params.grainSynthesisSize, 0.25f, 4.0f);
    const float grainSynthesisSharpness = std::max(params.grainSynthesisSharpness, 0.25f);
    frameFloats[78] = std::clamp(params.grainSynthesisAmount, 0.0f, 3.0f);
    frameFloats[79] = params.grainSynthesisMeanRadiusUm * grainSynthesisSize;
    frameFloats[80] = params.grainSynthesisRadiusStdDevRatio;
    frameFloats[81] = params.grainSynthesisObservationSigmaUm / grainSynthesisSharpness;
    frameFloats[82] = params.grainSynthesisCellSizeRatio;
    frameFloats[83] = params.grainSynthesisMaxRadiusQuantile;
    frameFloats[84] = params.grainSynthesisCoverageEpsilon;
    frameFloats[85] = params.grainSynthesisRadiusScaleR;
    frameFloats[86] = params.grainSynthesisRadiusScaleG;
    frameFloats[87] = params.grainSynthesisRadiusScaleB;
    frameFloats[88] = params.grainSynthesisLayerScale0;
    frameFloats[89] = params.grainSynthesisLayerScale1;
    frameFloats[90] = params.grainSynthesisLayerScale2;

    std::array<uint32_t, kCoreFrameIntCount> frameInts{};
    frameInts[0] = static_cast<uint32_t>(params.process);
    frameInts[1] = static_cast<uint32_t>(params.outputColorSpace);
    frameInts[2] = static_cast<uint32_t>(params.outputRole);
    frameInts[3] = static_cast<uint32_t>(params.printTiming);
    frameInts[4] = static_cast<uint32_t>(std::max(params.film, 0));
    frameInts[5] = static_cast<uint32_t>(std::max(params.paper, 0));
    frameInts[6] = kSpektraFilmCount;
    frameInts[7] = kSpektraPaperCount;
    frameInts[8] = staticFilm.wavelengthCount;
    frameInts[9] = staticFilm.paperExposureCount;
    frameInts[10] = staticFilm.filmPositive;
    frameInts[11] = params.printerLightsGang ? 1u : 0u;
    frameInts[12] = params.printerLightCalibration ? 1u : 0u;
    frameInts[13] = params.scannerEnabled ? 1u : 0u;
    frameInts[14] = params.scannerWhiteCorrection ? 1u : 0u;
    frameInts[15] = params.scannerBlackCorrection ? 1u : 0u;
    frameInts[16] = params.grainSeed;
    frameInts[17] = params.grainSublayersEnabled ? 1u : 0u;
    frameInts[18] = static_cast<uint32_t>(std::max(params.grainSubLayerCount, 1));
    frameInts[19] = params.grainAnimate
      ? static_cast<uint32_t>(std::max(0.0, std::floor(time * 24.0 + 0.5)))
      : 0u;
    frameInts[20] = static_cast<uint32_t>(params.hdrTransfer);
    frameInts[21] = static_cast<uint32_t>(params.hdrToneMapping);
    frameInts[22] = static_cast<uint32_t>(std::clamp(
      static_cast<int32_t>(std::lround(static_cast<float>(params.grainSynthesisSamples) * grainSynthesisQuality)),
      1,
      1024
    ));
    frameInts[23] = static_cast<uint32_t>(std::clamp(params.grainSynthesisMaxGrainsPerCell, 1, 128));
    frameInts[24] = params.grainSynthesisLayered ? 1u : 0u;
    frameInts[25] = grainSynthesisPath ? 1u : 0u;
    frameInts[26] = grainBlurRecurrence ? 1u : 0u;
    frameInts[27] = colorAdaptationFlags(params);
    frameInts[28] = params.scanNegativeInvert ? 1u : 0u;

    if (!coreFrame.frameFloats.mapped || !coreFrame.frameInts.mapped) {
      lastError = "Vulkan frame parameter buffers are not mapped.";
      return false;
    }
    std::memcpy(coreFrame.frameFloats.mapped, frameFloats.data(), frameFloats.size() * sizeof(float));
    std::memcpy(coreFrame.frameInts.mapped, frameInts.data(), frameInts.size() * sizeof(uint32_t));
    if (!flushMappedScratchBuffer(coreFrame.frameFloats, frameFloats.size() * sizeof(float), "core frame float params") ||
        !flushMappedScratchBuffer(coreFrame.frameInts, frameInts.size() * sizeof(uint32_t), "core frame int params")) {
      return false;
    }
  }

  VkDescriptorBufferInfo sourceBufferInfo{};
  sourceBufferInfo.buffer = coreFrame.source.buffer;
  sourceBufferInfo.offset = 0;
  sourceBufferInfo.range = sourceByteCount;
  VkDescriptorBufferInfo filmRawBufferInfo{};
  filmRawBufferInfo.buffer = coreFrame.filmRaw.buffer;
  filmRawBufferInfo.offset = 0;
  filmRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo destinationBufferInfo{};
  destinationBufferInfo.buffer = coreFrame.destination.buffer;
  destinationBufferInfo.offset = 0;
  destinationBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo destinationHalfBufferInfo{};
  destinationHalfBufferInfo.buffer = coreFrame.destinationHalf.buffer;
  destinationHalfBufferInfo.offset = 0;
  destinationHalfBufferInfo.range = halfByteCount;
  VkDescriptorBufferInfo filmDensityBufferInfo{};
  filmDensityBufferInfo.buffer = coreFrame.filmDensity.buffer;
  filmDensityBufferInfo.offset = 0;
  filmDensityBufferInfo.range = pixelStorageByteCount;
  const VkDescriptorBufferInfo &developOutputBufferInfo = filmDensityIntermediateEnabled ? filmDensityBufferInfo : destinationBufferInfo;
  VkDescriptorBufferInfo logExposureBufferInfo{};
  logExposureBufferInfo.buffer = staticFilm.logExposure.buffer;
  logExposureBufferInfo.offset = 0;
  logExposureBufferInfo.range = staticFilm.logExposure.capacity;
  VkDescriptorBufferInfo densityCurvesBufferInfo{};
  densityCurvesBufferInfo.buffer = staticFilm.densityCurves.buffer;
  densityCurvesBufferInfo.offset = 0;
  densityCurvesBufferInfo.range = staticFilm.densityCurves.capacity;
  VkDescriptorBufferInfo densityCurveLayersBufferInfo{};
  densityCurveLayersBufferInfo.buffer = staticFilm.densityCurveLayers.buffer != VK_NULL_HANDLE
    ? staticFilm.densityCurveLayers.buffer
    : staticFilm.densityCurves.buffer;
  densityCurveLayersBufferInfo.offset = 0;
  densityCurveLayersBufferInfo.range = staticFilm.densityCurveLayers.buffer != VK_NULL_HANDLE
    ? staticFilm.densityCurveLayers.capacity
    : staticFilm.densityCurves.capacity;
  VkDescriptorBufferInfo densityCurveLayerMaximaBufferInfo{};
  densityCurveLayerMaximaBufferInfo.buffer = staticFilm.densityCurveLayerMaxima.buffer != VK_NULL_HANDLE
    ? staticFilm.densityCurveLayerMaxima.buffer
    : staticFilm.densityCurves.buffer;
  densityCurveLayerMaximaBufferInfo.offset = 0;
  densityCurveLayerMaximaBufferInfo.range = staticFilm.densityCurveLayerMaxima.buffer != VK_NULL_HANDLE
    ? staticFilm.densityCurveLayerMaxima.capacity
    : staticFilm.densityCurves.capacity;
  VkDescriptorBufferInfo inputToReferenceXyzBufferInfo{};
  inputToReferenceXyzBufferInfo.buffer = staticFilm.inputToReferenceXyz.buffer;
  inputToReferenceXyzBufferInfo.offset = 0;
  inputToReferenceXyzBufferInfo.range = staticFilm.inputToReferenceXyz.capacity;
  VkDescriptorBufferInfo inputToSrgbBufferInfo{};
  inputToSrgbBufferInfo.buffer = staticFilm.inputToSrgb.buffer;
  inputToSrgbBufferInfo.offset = 0;
  inputToSrgbBufferInfo.range = staticFilm.inputToSrgb.capacity;
  VkDescriptorBufferInfo colorDecodeLutsBufferInfo{};
  colorDecodeLutsBufferInfo.buffer = staticFilm.colorDecodeLuts.buffer;
  colorDecodeLutsBufferInfo.offset = 0;
  colorDecodeLutsBufferInfo.range = staticFilm.colorDecodeLuts.capacity;
  VkDescriptorBufferInfo colorTransferKindsBufferInfo{};
  colorTransferKindsBufferInfo.buffer = staticFilm.colorTransferKinds.buffer;
  colorTransferKindsBufferInfo.offset = 0;
  colorTransferKindsBufferInfo.range = staticFilm.colorTransferKinds.capacity;
  VkDescriptorBufferInfo mallettRawMatrixBufferInfo{};
  mallettRawMatrixBufferInfo.buffer = staticFilm.mallettRawMatrix.buffer;
  mallettRawMatrixBufferInfo.offset = 0;
  mallettRawMatrixBufferInfo.range = staticFilm.mallettRawMatrix.capacity;
  VkDescriptorBufferInfo hanatosRawResponseBufferInfo{};
  hanatosRawResponseBufferInfo.buffer = staticFilm.hanatosRawResponse.buffer;
  hanatosRawResponseBufferInfo.offset = 0;
  hanatosRawResponseBufferInfo.range = staticFilm.hanatosRawResponse.capacity;
  VkDescriptorBufferInfo paperHanatosResponseBufferInfo{};
  paperHanatosResponseBufferInfo.buffer = staticFilm.paperHanatosResponse.buffer;
  paperHanatosResponseBufferInfo.offset = 0;
  paperHanatosResponseBufferInfo.range = staticFilm.paperHanatosResponse.capacity;
  VkDescriptorBufferInfo preflashPaperHanatosResponseBufferInfo{};
  preflashPaperHanatosResponseBufferInfo.buffer = staticFilm.preflashPaperHanatosResponse.buffer;
  preflashPaperHanatosResponseBufferInfo.offset = 0;
  preflashPaperHanatosResponseBufferInfo.range = staticFilm.preflashPaperHanatosResponse.capacity;
  VkDescriptorBufferInfo paperLogExposureBufferInfo{};
  paperLogExposureBufferInfo.buffer = staticFilm.paperLogExposure.buffer;
  paperLogExposureBufferInfo.offset = 0;
  paperLogExposureBufferInfo.range = staticFilm.paperLogExposure.capacity;
  VkDescriptorBufferInfo paperDensityCurvesBufferInfo{};
  paperDensityCurvesBufferInfo.buffer = staticFilm.paperDensityCurves.buffer;
  paperDensityCurvesBufferInfo.offset = 0;
  paperDensityCurvesBufferInfo.range = staticFilm.paperDensityCurves.capacity;
  VkDescriptorBufferInfo filmChannelDensityBufferInfo{};
  filmChannelDensityBufferInfo.buffer = staticFilm.filmChannelDensity.buffer;
  filmChannelDensityBufferInfo.offset = 0;
  filmChannelDensityBufferInfo.range = staticFilm.filmChannelDensity.capacity;
  VkDescriptorBufferInfo filmBaseDensityBufferInfo{};
  filmBaseDensityBufferInfo.buffer = staticFilm.filmBaseDensity.buffer;
  filmBaseDensityBufferInfo.offset = 0;
  filmBaseDensityBufferInfo.range = staticFilm.filmBaseDensity.capacity;
  VkDescriptorBufferInfo paperLogSensitivityBufferInfo{};
  paperLogSensitivityBufferInfo.buffer = staticFilm.paperLogSensitivity.buffer;
  paperLogSensitivityBufferInfo.offset = 0;
  paperLogSensitivityBufferInfo.range = staticFilm.paperLogSensitivity.capacity;
  VkDescriptorBufferInfo filteredEnlargerResponseBufferInfo{};
  filteredEnlargerResponseBufferInfo.buffer = coreFrame.filteredEnlargerResponse.buffer;
  filteredEnlargerResponseBufferInfo.offset = 0;
  filteredEnlargerResponseBufferInfo.range = coreFrame.filteredEnlargerResponse.capacity;
  VkDescriptorBufferInfo thKg3IlluminantBufferInfo{};
  thKg3IlluminantBufferInfo.buffer = staticFilm.thKg3Illuminant.buffer;
  thKg3IlluminantBufferInfo.offset = 0;
  thKg3IlluminantBufferInfo.range = staticFilm.thKg3Illuminant.capacity;
  VkDescriptorBufferInfo customEnlargerFiltersBufferInfo{};
  customEnlargerFiltersBufferInfo.buffer = staticFilm.customEnlargerFilters.buffer;
  customEnlargerFiltersBufferInfo.offset = 0;
  customEnlargerFiltersBufferInfo.range = staticFilm.customEnlargerFilters.capacity;
  VkDescriptorBufferInfo neutralPrintFiltersBufferInfo{};
  neutralPrintFiltersBufferInfo.buffer = staticFilm.neutralPrintFilters.buffer;
  neutralPrintFiltersBufferInfo.offset = 0;
  neutralPrintFiltersBufferInfo.range = staticFilm.neutralPrintFilters.capacity;
  VkDescriptorBufferInfo paperChannelDensityBufferInfo{};
  paperChannelDensityBufferInfo.buffer = staticFilm.paperChannelDensity.buffer;
  paperChannelDensityBufferInfo.offset = 0;
  paperChannelDensityBufferInfo.range = staticFilm.paperChannelDensity.capacity;
  VkDescriptorBufferInfo paperBaseDensityBufferInfo{};
  paperBaseDensityBufferInfo.buffer = staticFilm.paperBaseDensity.buffer;
  paperBaseDensityBufferInfo.offset = 0;
  paperBaseDensityBufferInfo.range = staticFilm.paperBaseDensity.capacity;
  VkDescriptorBufferInfo filmScanIlluminantBufferInfo{};
  filmScanIlluminantBufferInfo.buffer = staticFilm.filmScanIlluminant.buffer;
  filmScanIlluminantBufferInfo.offset = 0;
  filmScanIlluminantBufferInfo.range = staticFilm.filmScanIlluminant.capacity;
  VkDescriptorBufferInfo paperScanIlluminantBufferInfo{};
  paperScanIlluminantBufferInfo.buffer = staticFilm.paperScanIlluminant.buffer;
  paperScanIlluminantBufferInfo.offset = 0;
  paperScanIlluminantBufferInfo.range = staticFilm.paperScanIlluminant.capacity;
  VkDescriptorBufferInfo standardObserverCmfsBufferInfo{};
  standardObserverCmfsBufferInfo.buffer = staticFilm.standardObserverCmfs.buffer;
  standardObserverCmfsBufferInfo.offset = 0;
  standardObserverCmfsBufferInfo.range = staticFilm.standardObserverCmfs.capacity;
  VkDescriptorBufferInfo filmScanToOutputRgbBufferInfo{};
  filmScanToOutputRgbBufferInfo.buffer = staticFilm.filmScanToOutputRgb.buffer;
  filmScanToOutputRgbBufferInfo.offset = 0;
  filmScanToOutputRgbBufferInfo.range = staticFilm.filmScanToOutputRgb.capacity;
  VkDescriptorBufferInfo paperScanToOutputRgbBufferInfo{};
  paperScanToOutputRgbBufferInfo.buffer = staticFilm.paperScanToOutputRgb.buffer;
  paperScanToOutputRgbBufferInfo.offset = 0;
  paperScanToOutputRgbBufferInfo.range = staticFilm.paperScanToOutputRgb.capacity;
  VkDescriptorBufferInfo colorEncodeLutsBufferInfo{};
  colorEncodeLutsBufferInfo.buffer = staticFilm.colorEncodeLuts.buffer;
  colorEncodeLutsBufferInfo.offset = 0;
  colorEncodeLutsBufferInfo.range = staticFilm.colorEncodeLuts.capacity;
  VkDescriptorBufferInfo academyPrinterDensityDataBufferInfo{};
  academyPrinterDensityDataBufferInfo.buffer = staticFilm.academyPrinterDensityData.buffer;
  academyPrinterDensityDataBufferInfo.offset = 0;
  academyPrinterDensityDataBufferInfo.range = staticFilm.academyPrinterDensityData.capacity;
  VkDescriptorBufferInfo halationRawABufferInfo{};
  halationRawABufferInfo.buffer = coreFrame.halationRawA.buffer;
  halationRawABufferInfo.offset = 0;
  halationRawABufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo halationRawBBufferInfo{};
  halationRawBBufferInfo.buffer = coreFrame.halationRawB.buffer;
  halationRawBBufferInfo.offset = 0;
  halationRawBBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo halationRawCBufferInfo{};
  halationRawCBufferInfo.buffer = coreFrame.halationRawC.buffer;
  halationRawCBufferInfo.offset = 0;
  halationRawCBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo halationRawDBufferInfo{};
  halationRawDBufferInfo.buffer = coreFrame.halationRawD.buffer;
  halationRawDBufferInfo.offset = 0;
  halationRawDBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo halationBoostedRawBufferInfo{};
  halationBoostedRawBufferInfo.buffer = coreFrame.halationBoostedRaw.buffer;
  halationBoostedRawBufferInfo.offset = 0;
  halationBoostedRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo halationBoostChunksBufferInfo{};
  halationBoostChunksBufferInfo.buffer = coreFrame.halationBoostChunks.buffer;
  halationBoostChunksBufferInfo.offset = 0;
  halationBoostChunksBufferInfo.range = coreFrame.halationBoostChunks.capacity;
  VkDescriptorBufferInfo halationBoostInfoBufferInfo{};
  halationBoostInfoBufferInfo.buffer = coreFrame.halationBoostInfo.buffer;
  halationBoostInfoBufferInfo.offset = 0;
  halationBoostInfoBufferInfo.range = coreFrame.halationBoostInfo.capacity;
  VkDescriptorBufferInfo tiledHalationBoostInfoBufferInfo{};
  tiledHalationBoostInfoBufferInfo.buffer = coreFrame.tiledHalationBoostInfo.buffer;
  tiledHalationBoostInfoBufferInfo.offset = 0;
  tiledHalationBoostInfoBufferInfo.range = coreFrame.tiledHalationBoostInfo.capacity;
  VkDescriptorBufferInfo halationLogRawBufferInfo{};
  halationLogRawBufferInfo.buffer = coreFrame.halationLogRaw.buffer;
  halationLogRawBufferInfo.offset = 0;
  halationLogRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo diffusionTempBufferInfo{};
  diffusionTempBufferInfo.buffer = coreFrame.diffusionTemp.buffer;
  diffusionTempBufferInfo.offset = 0;
  diffusionTempBufferInfo.range = coreFrame.diffusionTemp.capacity;
  VkDescriptorBufferInfo diffusionAccumBufferInfo{};
  diffusionAccumBufferInfo.buffer = coreFrame.diffusionAccum.buffer;
  diffusionAccumBufferInfo.offset = 0;
  diffusionAccumBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo diffusionDownsampleSourceBufferInfo{};
  diffusionDownsampleSourceBufferInfo.buffer = coreFrame.diffusionDownsampleSource.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleSource.buffer
    : coreFrame.diffusionTemp.buffer;
  diffusionDownsampleSourceBufferInfo.offset = 0;
  diffusionDownsampleSourceBufferInfo.range = coreFrame.diffusionDownsampleSource.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleSource.capacity
    : coreFrame.diffusionTemp.capacity;
  VkDescriptorBufferInfo diffusionDownsampleTempBufferInfo{};
  diffusionDownsampleTempBufferInfo.buffer = coreFrame.diffusionDownsampleTemp.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleTemp.buffer
    : coreFrame.diffusionTemp.buffer;
  diffusionDownsampleTempBufferInfo.offset = 0;
  diffusionDownsampleTempBufferInfo.range = coreFrame.diffusionDownsampleTemp.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleTemp.capacity
    : coreFrame.diffusionTemp.capacity;
  VkDescriptorBufferInfo diffusionDownsampleBlurBufferInfo{};
  diffusionDownsampleBlurBufferInfo.buffer = coreFrame.diffusionDownsampleBlur.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleBlur.buffer
    : coreFrame.diffusionTemp.buffer;
  diffusionDownsampleBlurBufferInfo.offset = 0;
  diffusionDownsampleBlurBufferInfo.range = coreFrame.diffusionDownsampleBlur.buffer != VK_NULL_HANDLE
    ? coreFrame.diffusionDownsampleBlur.capacity
    : coreFrame.diffusionTemp.capacity;
  VkDescriptorBufferInfo cameraDiffusionRawBufferInfo{};
  cameraDiffusionRawBufferInfo.buffer = coreFrame.cameraDiffusionRaw.buffer;
  cameraDiffusionRawBufferInfo.offset = 0;
  cameraDiffusionRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo printRawBufferInfo{};
  printRawBufferInfo.buffer = coreFrame.printRaw.buffer;
  printRawBufferInfo.offset = 0;
  printRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo printDiffusionRawBufferInfo{};
  printDiffusionRawBufferInfo.buffer = coreFrame.printDiffusionRaw.buffer;
  printDiffusionRawBufferInfo.offset = 0;
  printDiffusionRawBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo cameraDiffusionInfoBufferInfo{};
  cameraDiffusionInfoBufferInfo.buffer = coreFrame.cameraDiffusionInfo.buffer;
  cameraDiffusionInfoBufferInfo.offset = 0;
  cameraDiffusionInfoBufferInfo.range = coreFrame.cameraDiffusionInfo.capacity;
  VkDescriptorBufferInfo cameraDiffusionComponentsBufferInfo{};
  cameraDiffusionComponentsBufferInfo.buffer = coreFrame.cameraDiffusionComponents.buffer;
  cameraDiffusionComponentsBufferInfo.offset = 0;
  cameraDiffusionComponentsBufferInfo.range = coreFrame.cameraDiffusionComponents.capacity;
  VkDescriptorBufferInfo printDiffusionInfoBufferInfo{};
  printDiffusionInfoBufferInfo.buffer = coreFrame.printDiffusionInfo.buffer;
  printDiffusionInfoBufferInfo.offset = 0;
  printDiffusionInfoBufferInfo.range = coreFrame.printDiffusionInfo.capacity;
  VkDescriptorBufferInfo printDiffusionComponentsBufferInfo{};
  printDiffusionComponentsBufferInfo.buffer = coreFrame.printDiffusionComponents.buffer;
  printDiffusionComponentsBufferInfo.offset = 0;
  printDiffusionComponentsBufferInfo.range = coreFrame.printDiffusionComponents.capacity;
  VkDescriptorBufferInfo dirCorrectionABufferInfo{};
  dirCorrectionABufferInfo.buffer = coreFrame.dirCorrectionA.buffer;
  dirCorrectionABufferInfo.offset = 0;
  dirCorrectionABufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo dirCorrectionBBufferInfo{};
  dirCorrectionBBufferInfo.buffer = coreFrame.dirCorrectionB.buffer;
  dirCorrectionBBufferInfo.offset = 0;
  dirCorrectionBBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo dirCorrectionCBufferInfo{};
  dirCorrectionCBufferInfo.buffer = coreFrame.dirCorrectionC.buffer;
  dirCorrectionCBufferInfo.offset = 0;
  dirCorrectionCBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo dirDensityBufferInfo{};
  dirDensityBufferInfo.buffer = coreFrame.dirDensity.buffer;
  dirDensityBufferInfo.offset = 0;
  dirDensityBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo dirFloatsBufferInfo{};
  dirFloatsBufferInfo.buffer = coreFrame.dirFloats.buffer;
  dirFloatsBufferInfo.offset = 0;
  dirFloatsBufferInfo.range = coreFrame.dirFloats.capacity;
  VkDescriptorBufferInfo dirCorrectedDensityCurvesBufferInfo{};
  dirCorrectedDensityCurvesBufferInfo.buffer = coreFrame.dirCorrectedDensityCurves.buffer;
  dirCorrectedDensityCurvesBufferInfo.offset = 0;
  dirCorrectedDensityCurvesBufferInfo.range = coreFrame.dirCorrectedDensityCurves.capacity;
  VkDescriptorBufferInfo scannerPostABufferInfo{};
  scannerPostABufferInfo.buffer = coreFrame.scannerPostA.buffer;
  scannerPostABufferInfo.offset = 0;
  scannerPostABufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo scannerPostBBufferInfo{};
  scannerPostBBufferInfo.buffer = coreFrame.scannerPostB.buffer;
  scannerPostBBufferInfo.offset = 0;
  scannerPostBBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo scannerPostCBufferInfo{};
  scannerPostCBufferInfo.buffer = coreFrame.scannerPostC.buffer;
  scannerPostCBufferInfo.offset = 0;
  scannerPostCBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo printGlareABufferInfo{};
  printGlareABufferInfo.buffer = coreFrame.printGlareA.buffer;
  printGlareABufferInfo.offset = 0;
  printGlareABufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo printGlareBBufferInfo{};
  printGlareBBufferInfo.buffer = coreFrame.printGlareB.buffer;
  printGlareBBufferInfo.offset = 0;
  printGlareBBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo grainDensityABufferInfo{};
  grainDensityABufferInfo.buffer = coreFrame.grainDensityA.buffer;
  grainDensityABufferInfo.offset = 0;
  grainDensityABufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo grainDensityBBufferInfo{};
  grainDensityBBufferInfo.buffer = coreFrame.grainDensityB.buffer;
  grainDensityBBufferInfo.offset = 0;
  grainDensityBBufferInfo.range = pixelStorageByteCount;
  VkDescriptorBufferInfo grainMicroABufferInfo{};
  grainMicroABufferInfo.buffer = coreFrame.grainMicroA.buffer != VK_NULL_HANDLE
    ? coreFrame.grainMicroA.buffer
    : coreFrame.grainDensityA.buffer;
  grainMicroABufferInfo.offset = 0;
  grainMicroABufferInfo.range = coreFrame.grainMicroA.buffer != VK_NULL_HANDLE ? pixelStorageByteCount : coreFrame.grainDensityA.capacity;
  VkDescriptorBufferInfo grainMicroBBufferInfo{};
  grainMicroBBufferInfo.buffer = coreFrame.grainMicroB.buffer != VK_NULL_HANDLE
    ? coreFrame.grainMicroB.buffer
    : coreFrame.grainDensityB.buffer;
  grainMicroBBufferInfo.offset = 0;
  grainMicroBBufferInfo.range = coreFrame.grainMicroB.buffer != VK_NULL_HANDLE ? pixelStorageByteCount : coreFrame.grainDensityB.capacity;
  VkDescriptorBufferInfo grainLayerABufferInfo{};
  grainLayerABufferInfo.buffer = coreFrame.grainLayerA.buffer != VK_NULL_HANDLE
    ? coreFrame.grainLayerA.buffer
    : coreFrame.grainDensityA.buffer;
  grainLayerABufferInfo.offset = 0;
  grainLayerABufferInfo.range = coreFrame.grainLayerA.buffer != VK_NULL_HANDLE ? grainLayerByteCount : coreFrame.grainDensityA.capacity;
  VkDescriptorBufferInfo grainLayerBBufferInfo{};
  grainLayerBBufferInfo.buffer = coreFrame.grainLayerB.buffer != VK_NULL_HANDLE
    ? coreFrame.grainLayerB.buffer
    : coreFrame.grainDensityB.buffer;
  grainLayerBBufferInfo.offset = 0;
  grainLayerBBufferInfo.range = coreFrame.grainLayerB.buffer != VK_NULL_HANDLE ? grainLayerByteCount : coreFrame.grainDensityB.capacity;
  const VkDescriptorBufferInfo &dirCorrectionFinalBufferInfo = dirPath
    ? (dirTailPath ? dirCorrectionBBufferInfo : (dirBlurPath ? dirCorrectionCBufferInfo : dirCorrectionABufferInfo))
    : dirCorrectionABufferInfo;
  const VkDescriptorBufferInfo &dirDensityOutputBufferInfo =
    (printScanPassEnabled || grainPath) ? dirDensityBufferInfo : destinationBufferInfo;
  const VkDescriptorBufferInfo &preGrainFilmDensityBufferInfo = dirPath ? dirDensityBufferInfo : filmDensityBufferInfo;
  const VkDescriptorBufferInfo &grainFinalDensityBufferInfo = productionGrainPath || grainSynthesisPath
    ? (printScanPassEnabled ? grainDensityBBufferInfo : destinationBufferInfo)
    : (previewGrainPath ? (printScanPassEnabled ? grainDensityABufferInfo : destinationBufferInfo) : preGrainFilmDensityBufferInfo);
  const VkDescriptorBufferInfo &finalFilmDensityBufferInfo = grainPath
    ? grainFinalDensityBufferInfo
    : (dirPath ? (printScanPassEnabled ? dirDensityBufferInfo : destinationBufferInfo) : filmDensityBufferInfo);
  const VkDescriptorBufferInfo &developInputBufferInfo = preExposureRawPath ? halationLogRawBufferInfo : filmRawBufferInfo;
  VkDescriptorBufferInfo frameFloatsBufferInfo{};
  frameFloatsBufferInfo.buffer = coreFrame.frameFloats.buffer;
  frameFloatsBufferInfo.offset = 0;
  frameFloatsBufferInfo.range = coreFrame.frameFloats.capacity;
  VkDescriptorBufferInfo frameIntsBufferInfo{};
  frameIntsBufferInfo.buffer = coreFrame.frameInts.buffer;
  frameIntsBufferInfo.offset = 0;
  frameIntsBufferInfo.range = coreFrame.frameInts.capacity;

  std::array<VkWriteDescriptorSet, 460> writes{};
  uint32_t writeCount = 0;
  auto writeStorageBuffer = [&](VkDescriptorSet set, uint32_t binding, const VkDescriptorBufferInfo &bufferInfo) {
    VkWriteDescriptorSet &write = writes[writeCount++];
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &bufferInfo;
  };
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 0, sourceBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 1, filmRawBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 2, logExposureBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 3, densityCurvesBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 4, inputToSrgbBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 5, colorDecodeLutsBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 6, colorTransferKindsBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 7, mallettRawMatrixBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 8, inputToReferenceXyzBufferInfo);
  writeStorageBuffer(coreFrame.exposureDescriptorSet, 9, hanatosRawResponseBufferInfo);
  writeStorageBuffer(coreFrame.developDescriptorSet, 0, developInputBufferInfo);
  writeStorageBuffer(coreFrame.developDescriptorSet, 1, developOutputBufferInfo);
  writeStorageBuffer(coreFrame.developDescriptorSet, 2, logExposureBufferInfo);
  writeStorageBuffer(coreFrame.developDescriptorSet, 3, densityCurvesBufferInfo);
  auto writePrintScanSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &inputBufferInfo,
    const VkDescriptorBufferInfo &outputBufferInfo
  ) {
    writeStorageBuffer(set, 0, inputBufferInfo);
    writeStorageBuffer(set, 1, outputBufferInfo);
    writeStorageBuffer(set, 2, logExposureBufferInfo);
    writeStorageBuffer(set, 3, densityCurvesBufferInfo);
    writeStorageBuffer(set, 5, colorDecodeLutsBufferInfo);
    writeStorageBuffer(set, 6, colorTransferKindsBufferInfo);
    writeStorageBuffer(set, 7, mallettRawMatrixBufferInfo);
    writeStorageBuffer(set, 8, inputToReferenceXyzBufferInfo);
    writeStorageBuffer(set, 9, hanatosRawResponseBufferInfo);
    writeStorageBuffer(set, 10, paperLogExposureBufferInfo);
    writeStorageBuffer(set, 11, paperDensityCurvesBufferInfo);
    writeStorageBuffer(set, 12, filmChannelDensityBufferInfo);
    writeStorageBuffer(set, 13, filmBaseDensityBufferInfo);
    writeStorageBuffer(set, 14, filteredEnlargerResponseBufferInfo);
    writeStorageBuffer(set, 15, thKg3IlluminantBufferInfo);
    writeStorageBuffer(set, 16, customEnlargerFiltersBufferInfo);
    writeStorageBuffer(set, 17, neutralPrintFiltersBufferInfo);
    writeStorageBuffer(set, 18, paperChannelDensityBufferInfo);
    writeStorageBuffer(set, 19, paperBaseDensityBufferInfo);
    writeStorageBuffer(set, 20, filmScanIlluminantBufferInfo);
    writeStorageBuffer(set, 21, paperLogSensitivityBufferInfo);
    writeStorageBuffer(set, 22, standardObserverCmfsBufferInfo);
    writeStorageBuffer(set, 23, filmScanToOutputRgbBufferInfo);
    writeStorageBuffer(set, 24, paperScanToOutputRgbBufferInfo);
    writeStorageBuffer(set, 25, colorEncodeLutsBufferInfo);
    writeStorageBuffer(set, 26, academyPrinterDensityDataBufferInfo);
    writeStorageBuffer(set, 27, frameFloatsBufferInfo);
    writeStorageBuffer(set, 28, frameIntsBufferInfo);
    writeStorageBuffer(set, 29, paperHanatosResponseBufferInfo);
    writeStorageBuffer(set, 30, preflashPaperHanatosResponseBufferInfo);
  };
  if (printScanPassEnabled) {
    const VkDescriptorBufferInfo &printScanInputBufferInfo = finalProcessNegative
      ? sourceBufferInfo
      : finalFilmDensityBufferInfo;
    if (printDiffusionPath) {
      writePrintScanSet(coreFrame.printDiffusionDescriptorSets[0], printScanInputBufferInfo, printRawBufferInfo);
      writePrintScanSet(
        coreFrame.printDiffusionDescriptorSets[1],
        printDiffusionRawBufferInfo,
        scannerPostPath ? scannerPostABufferInfo : destinationBufferInfo
      );
    } else {
      writePrintScanSet(
        coreFrame.finalDescriptorSet,
        printScanInputBufferInfo,
        scannerPostPath ? scannerPostABufferInfo : destinationBufferInfo
      );
    }
  }
  auto writeHalationSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &binding0,
    const VkDescriptorBufferInfo &binding1,
    const VkDescriptorBufferInfo &binding2,
    const VkDescriptorBufferInfo &binding3
  ) {
    writeStorageBuffer(set, 0, binding0);
    writeStorageBuffer(set, 1, binding1);
    writeStorageBuffer(set, 2, binding2);
    writeStorageBuffer(set, 3, binding3);
    writeStorageBuffer(set, 27, frameFloatsBufferInfo);
  };
  auto writeDiffusionSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &source,
    const VkDescriptorBufferInfo &temp,
    const VkDescriptorBufferInfo &accum,
    const VkDescriptorBufferInfo &destination,
    const VkDescriptorBufferInfo &info,
    const VkDescriptorBufferInfo &components
  ) {
    writeStorageBuffer(set, 0, source);
    writeStorageBuffer(set, 1, temp);
    writeStorageBuffer(set, 2, accum);
    writeStorageBuffer(set, 3, destination);
    writeStorageBuffer(set, 4, diffusionDownsampleSourceBufferInfo);
    writeStorageBuffer(set, 5, diffusionDownsampleTempBufferInfo);
    writeStorageBuffer(set, 6, diffusionDownsampleBlurBufferInfo);
    writeStorageBuffer(set, 27, info);
    writeStorageBuffer(set, 28, components);
  };
  if (cameraDiffusionPath) {
    const VkDescriptorBufferInfo &cameraDiffusionSourceBufferInfo =
      halationBoostEnabled ? halationBoostedRawBufferInfo : filmRawBufferInfo;
    writeDiffusionSet(
      coreFrame.diffusionDescriptorSets[0],
      cameraDiffusionSourceBufferInfo,
      diffusionTempBufferInfo,
      diffusionAccumBufferInfo,
      cameraDiffusionRawBufferInfo,
      cameraDiffusionInfoBufferInfo,
      cameraDiffusionComponentsBufferInfo
    );
    if (!halationPassEnabled) {
      writeDiffusionSet(
        coreFrame.diffusionDescriptorSets[1],
        cameraDiffusionRawBufferInfo,
        diffusionTempBufferInfo,
        diffusionAccumBufferInfo,
        halationLogRawBufferInfo,
        cameraDiffusionInfoBufferInfo,
        cameraDiffusionComponentsBufferInfo
      );
    }
  }
  if (printDiffusionPath) {
    writeDiffusionSet(
      coreFrame.diffusionDescriptorSets[2],
      printRawBufferInfo,
      diffusionTempBufferInfo,
      diffusionAccumBufferInfo,
      printDiffusionRawBufferInfo,
      printDiffusionInfoBufferInfo,
      printDiffusionComponentsBufferInfo
    );
  }
  auto writeDirSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &binding0,
    const VkDescriptorBufferInfo &binding1,
    const VkDescriptorBufferInfo &binding2,
    const VkDescriptorBufferInfo &binding3,
    const VkDescriptorBufferInfo &curves
  ) {
    writeStorageBuffer(set, 0, binding0);
    writeStorageBuffer(set, 1, binding1);
    writeStorageBuffer(set, 2, binding2);
    writeStorageBuffer(set, 3, binding3);
    writeStorageBuffer(set, 4, logExposureBufferInfo);
    writeStorageBuffer(set, 5, curves);
    writeStorageBuffer(set, 27, dirFloatsBufferInfo);
  };
  if (dirPath) {
    writeDirSet(coreFrame.dirDescriptorSets[0], filmDensityBufferInfo, dirCorrectionABufferInfo, dirCorrectionABufferInfo, dirCorrectionABufferInfo, dirCorrectedDensityCurvesBufferInfo);
    writeDirSet(coreFrame.dirDescriptorSets[1], dirCorrectionABufferInfo, dirCorrectionBBufferInfo, dirCorrectionABufferInfo, dirCorrectionABufferInfo, dirCorrectedDensityCurvesBufferInfo);
    writeDirSet(coreFrame.dirDescriptorSets[2], dirCorrectionBBufferInfo, dirCorrectionCBufferInfo, dirCorrectionABufferInfo, dirCorrectionCBufferInfo, dirCorrectedDensityCurvesBufferInfo);
    const VkDescriptorBufferInfo &tailBaseBufferInfo = dirBlurPath ? dirCorrectionCBufferInfo : dirCorrectionABufferInfo;
    writeDirSet(coreFrame.dirDescriptorSets[3], tailBaseBufferInfo, dirCorrectionBBufferInfo, dirCorrectionABufferInfo, dirCorrectionBBufferInfo, dirCorrectedDensityCurvesBufferInfo);
    writeDirSet(coreFrame.dirDescriptorSets[4], dirCorrectionABufferInfo, dirCorrectionCBufferInfo, dirCorrectionABufferInfo, dirCorrectionCBufferInfo, dirCorrectedDensityCurvesBufferInfo);
    writeDirSet(coreFrame.dirDescriptorSets[5], dirCorrectionCBufferInfo, dirCorrectionBBufferInfo, dirCorrectionABufferInfo, dirCorrectionBBufferInfo, dirCorrectedDensityCurvesBufferInfo);
    writeDirSet(coreFrame.dirDescriptorSets[6], developInputBufferInfo, dirCorrectionABufferInfo, dirCorrectionFinalBufferInfo, dirDensityOutputBufferInfo, dirCorrectedDensityCurvesBufferInfo);
  }
  auto writeScannerPostSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &source,
    const VkDescriptorBufferInfo &temp,
    const VkDescriptorBufferInfo &unsharp,
    const VkDescriptorBufferInfo &destination
  ) {
    writeStorageBuffer(set, 0, source);
    writeStorageBuffer(set, 1, temp);
    writeStorageBuffer(set, 2, unsharp);
    writeStorageBuffer(set, 3, destination);
    writeStorageBuffer(set, 6, colorTransferKindsBufferInfo);
    writeStorageBuffer(set, 25, colorEncodeLutsBufferInfo);
    writeStorageBuffer(set, 27, frameFloatsBufferInfo);
    writeStorageBuffer(set, 28, frameIntsBufferInfo);
  };
  if (scannerPostPath) {
    const VkDescriptorBufferInfo &postGlareBufferInfo = printGlarePath ? scannerPostBBufferInfo : scannerPostABufferInfo;
    const VkDescriptorBufferInfo &postBlurBufferInfo = scannerBlurPath ? scannerPostABufferInfo : postGlareBufferInfo;
    const bool postBlurBufferIsA = scannerBlurPath || !printGlarePath;
    const VkDescriptorBufferInfo &unsharpBlurBufferInfo = postBlurBufferIsA ? scannerPostBBufferInfo : scannerPostABufferInfo;
    if (printGlarePath) {
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[0], scannerPostABufferInfo, printGlareABufferInfo, scannerPostABufferInfo, printGlareABufferInfo);
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[1], printGlareABufferInfo, printGlareBBufferInfo, printGlareABufferInfo, printGlareBBufferInfo);
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[2], printGlareBBufferInfo, printGlareABufferInfo, printGlareBBufferInfo, printGlareABufferInfo);
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[3], scannerPostABufferInfo, printGlareABufferInfo, scannerPostABufferInfo, scannerPostBBufferInfo);
    }
    if (scannerBlurPath) {
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[4], postGlareBufferInfo, scannerPostCBufferInfo, postGlareBufferInfo, scannerPostCBufferInfo);
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[5], scannerPostCBufferInfo, scannerPostABufferInfo, scannerPostCBufferInfo, scannerPostABufferInfo);
    }
    if (scannerUnsharpPath) {
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[6], postBlurBufferInfo, scannerPostCBufferInfo, postBlurBufferInfo, scannerPostCBufferInfo);
      writeScannerPostSet(coreFrame.scannerPostDescriptorSets[7], scannerPostCBufferInfo, unsharpBlurBufferInfo, scannerPostCBufferInfo, unsharpBlurBufferInfo);
    }
    writeScannerPostSet(
      coreFrame.scannerPostDescriptorSets[8],
      postBlurBufferInfo,
      scannerPostCBufferInfo,
      scannerUnsharpPath ? unsharpBlurBufferInfo : postBlurBufferInfo,
      destinationBufferInfo
    );
  }
  auto writeGrainSet = [&](
    VkDescriptorSet set,
    const VkDescriptorBufferInfo &source,
    const VkDescriptorBufferInfo &destination,
    const VkDescriptorBufferInfo &auxA,
    const VkDescriptorBufferInfo &auxB
  ) {
    writeStorageBuffer(set, 0, source);
    writeStorageBuffer(set, 1, destination);
    writeStorageBuffer(set, 2, auxA);
    writeStorageBuffer(set, 3, auxB);
    writeStorageBuffer(set, 4, grainMicroABufferInfo);
    writeStorageBuffer(set, 5, grainMicroBBufferInfo);
    writeStorageBuffer(set, 6, grainLayerABufferInfo);
    writeStorageBuffer(set, 7, grainLayerBBufferInfo);
    writeStorageBuffer(set, 8, densityCurvesBufferInfo);
    writeStorageBuffer(set, 9, densityCurveLayersBufferInfo);
    writeStorageBuffer(set, 10, densityCurveLayerMaximaBufferInfo);
    writeStorageBuffer(set, 27, frameFloatsBufferInfo);
    writeStorageBuffer(set, 28, frameIntsBufferInfo);
  };
  if (previewGrainPath) {
    writeGrainSet(
      coreFrame.grainDescriptorSets[0],
      preGrainFilmDensityBufferInfo,
      grainFinalDensityBufferInfo,
      grainDensityBBufferInfo,
      grainDensityBBufferInfo
    );
  }
  if (productionGrainPath || grainSynthesisPath) {
    writeGrainSet(
      coreFrame.grainDescriptorSets[1],
      preGrainFilmDensityBufferInfo,
      grainDensityABufferInfo,
      grainDensityBBufferInfo,
      grainDensityBBufferInfo
    );
    writeGrainSet(
      coreFrame.grainDescriptorSets[2],
      grainDensityABufferInfo,
      grainDensityBBufferInfo,
      grainDensityBBufferInfo,
      grainDensityABufferInfo
    );
    writeGrainSet(
      coreFrame.grainDescriptorSets[3],
      grainDensityBBufferInfo,
      grainDensityABufferInfo,
      grainDensityBBufferInfo,
      grainDensityABufferInfo
    );
    writeGrainSet(
      coreFrame.grainDescriptorSets[4],
      preGrainFilmDensityBufferInfo,
      grainDensityABufferInfo,
      grainFinalDensityBufferInfo,
      grainDensityBBufferInfo
    );
  }
  if (halationPassEnabled) {
    if (halationBoostEnabled) {
      if (halationBoostLocalReductionEnabled) {
        writeHalationSet(
          coreFrame.halationDescriptorSets[10],
          filmRawBufferInfo,
          halationBoostChunksBufferInfo,
          halationBoostInfoBufferInfo,
          halationBoostedRawBufferInfo
        );
        writeHalationSet(
          coreFrame.halationDescriptorSets[11],
          halationBoostChunksBufferInfo,
          halationBoostInfoBufferInfo,
          halationBoostInfoBufferInfo,
          halationBoostedRawBufferInfo
        );
      }
      const VkDescriptorBufferInfo &boostApplyInfoBufferInfo = halationBoostMilestoneEnabled
        ? tiledHalationBoostInfoBufferInfo
        : halationBoostInfoBufferInfo;
      writeHalationSet(
        coreFrame.halationDescriptorSets[12],
        filmRawBufferInfo,
        halationBoostedRawBufferInfo,
        boostApplyInfoBufferInfo,
        halationBoostedRawBufferInfo
      );
    }
    const VkDescriptorBufferInfo &scatterSourceBufferInfo = cameraDiffusionPath
      ? cameraDiffusionRawBufferInfo
      : (halationBoostEnabled ? halationBoostedRawBufferInfo : filmRawBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[0], scatterSourceBufferInfo, halationRawABufferInfo, halationRawABufferInfo, halationRawABufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[1], halationRawABufferInfo, halationRawBBufferInfo, halationRawABufferInfo, halationRawBBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[2], scatterSourceBufferInfo, halationRawCBufferInfo, halationRawABufferInfo, halationRawCBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[3], scatterSourceBufferInfo, halationRawABufferInfo, halationRawABufferInfo, halationRawABufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[4], halationRawABufferInfo, halationRawCBufferInfo, halationRawABufferInfo, halationRawCBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[5], scatterSourceBufferInfo, halationRawBBufferInfo, halationRawCBufferInfo, halationRawDBufferInfo);
    const VkDescriptorBufferInfo &bounceSourceBufferInfo = halationScatterEnabled ? halationRawDBufferInfo : scatterSourceBufferInfo;
    writeHalationSet(coreFrame.halationDescriptorSets[6], bounceSourceBufferInfo, halationRawCBufferInfo, halationRawABufferInfo, halationRawCBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[7], bounceSourceBufferInfo, halationRawABufferInfo, halationRawABufferInfo, halationRawABufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[8], halationRawABufferInfo, halationRawCBufferInfo, halationRawABufferInfo, halationRawCBufferInfo);
    writeHalationSet(coreFrame.halationDescriptorSets[9], bounceSourceBufferInfo, halationRawCBufferInfo, halationRawABufferInfo, halationLogRawBufferInfo);
  }
  vkUpdateDescriptorSets(device, writeCount, writes.data(), 0, nullptr);
  auto updateFormatDescriptorSet = [&](
    VkDescriptorSet descriptorSet,
    const VkDescriptorBufferInfo &sourceInfo,
    const VkDescriptorBufferInfo &destinationInfo
  ) {
    VkWriteDescriptorSet formatWrites[2]{};
    formatWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    formatWrites[0].dstSet = descriptorSet;
    formatWrites[0].dstBinding = 0;
    formatWrites[0].descriptorCount = 1;
    formatWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    formatWrites[0].pBufferInfo = &sourceInfo;
    formatWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    formatWrites[1].dstSet = descriptorSet;
    formatWrites[1].dstBinding = 1;
    formatWrites[1].descriptorCount = 1;
    formatWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    formatWrites[1].pBufferInfo = &destinationInfo;
    vkUpdateDescriptorSets(device, 2, formatWrites, 0, nullptr);
  };
  if (destinationFormatConvert) {
    updateFormatDescriptorSet(
      coreFrame.destinationFormatDescriptorSet,
      destinationBufferInfo,
      destinationHalfBufferInfo
    );
  }

  VkCommandBuffer commandBuffer = coreFrame.commandBuffer;
  result = vkResetCommandBuffer(commandBuffer, 0);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkResetCommandBuffer", result);
    return false;
  }

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkBeginCommandBuffer", result);
    return false;
  }

  VkMemoryBarrier hostInputBarrier{};
  hostInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostInputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
  hostInputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    VK_PIPELINE_STAGE_HOST_BIT,
    VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0,
    1,
    &hostInputBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  if (!fullFrameSourcePath) {
    VkBufferCopy sourceUploadRegion{};
    sourceUploadRegion.size = pixelStorageByteCount;
    vkCmdCopyBuffer(commandBuffer, coreFrame.sourceStaging.buffer, coreFrame.source.buffer, 1, &sourceUploadRegion);

    VkMemoryBarrier sourceUploadBarrier{};
    sourceUploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    sourceUploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    sourceUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &sourceUploadBarrier,
      0,
      nullptr,
      0,
      nullptr
    );
  }

  VulkanCorePushConstants pushConstants{};
  pushConstants.width = static_cast<uint32_t>(width);
  pushConstants.height = static_cast<uint32_t>(height);
  pushConstants.filmExposureEv = params.filmExposureEv + autoExposureEv;
  pushConstants.filmGamma = effectiveFilmGamma;
  pushConstants.exposureCount = staticFilm.exposureCount;
  pushConstants.inputColorSpace = static_cast<int32_t>(params.inputColorSpace);
  pushConstants.rgbToRawMethod = static_cast<int32_t>(params.rgbToRawMethod);
  pushConstants.colorSpaceCount = kSpektraColorSpaceCount;
  pushConstants.transferLutSize = kSpektraColorTransferLutSize;
  pushConstants.colorDecodeMin = colorDecodeLutMin();
  pushConstants.colorDecodeMax = colorDecodeLutMax();
  pushConstants.hanatosWidth = staticFilm.hanatosWidth;
  pushConstants.hanatosHeight = staticFilm.hanatosHeight;
  pushConstants.filmPushPullMode = static_cast<int32_t>(params.filmPushPullMode);
  pushConstants.filmPushPullStops = params.filmPushPullStops;
  pushConstants.fullWidth = coordinateWidth;
  pushConstants.fullHeight = coordinateHeight;
  pushConstants.tileOriginX = activeTileContext.enabled ? activeTileContext.tileOriginX : 0u;
  pushConstants.tileOriginY = activeTileContext.enabled ? activeTileContext.tileOriginY : 0u;

  struct ActiveRect {
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
  };
  const ActiveRect centerRect = activeTileContext.enabled
    ? ActiveRect{
        std::min(activeTileContext.centerOriginX, pushConstants.width),
        std::min(activeTileContext.centerOriginY, pushConstants.height),
        std::min(activeTileContext.centerWidth, pushConstants.width - std::min(activeTileContext.centerOriginX, pushConstants.width)),
        std::min(activeTileContext.centerHeight, pushConstants.height - std::min(activeTileContext.centerOriginY, pushConstants.height))
      }
    : ActiveRect{0u, 0u, pushConstants.width, pushConstants.height};
  auto inflatedCenterRect = [&](uint32_t radius) {
    const uint32_t x0 = centerRect.x > radius ? centerRect.x - radius : 0u;
    const uint32_t y0 = centerRect.y > radius ? centerRect.y - radius : 0u;
    const uint32_t x1 = std::min<uint32_t>(
      pushConstants.width,
      centerRect.x + centerRect.width > std::numeric_limits<uint32_t>::max() - radius
        ? pushConstants.width
        : centerRect.x + centerRect.width + radius
    );
    const uint32_t y1 = std::min<uint32_t>(
      pushConstants.height,
      centerRect.y + centerRect.height > std::numeric_limits<uint32_t>::max() - radius
        ? pushConstants.height
        : centerRect.y + centerRect.height + radius
    );
    return ActiveRect{x0, y0, x1 > x0 ? x1 - x0 : 0u, y1 > y0 ? y1 - y0 : 0u};
  };
  auto setActiveRect = [&](const ActiveRect &rect) {
    pushConstants.activeOriginX = std::min(rect.x, pushConstants.width);
    pushConstants.activeOriginY = std::min(rect.y, pushConstants.height);
    pushConstants.activeWidth = std::min(rect.width, pushConstants.width - pushConstants.activeOriginX);
    pushConstants.activeHeight = std::min(rect.height, pushConstants.height - pushConstants.activeOriginY);
    if (pushConstants.activeWidth == 0u || pushConstants.activeHeight == 0u) {
      pushConstants.activeOriginX = 0u;
      pushConstants.activeOriginY = 0u;
      pushConstants.activeWidth = pushConstants.width;
      pushConstants.activeHeight = pushConstants.height;
    }
  };
  uint32_t remainingSpatialRadius =
    cameraDiffusionRadius +
    halationRadius +
    dirRadius +
    grainRadius +
    printDiffusionRadius +
    scannerPostRadius;
  auto setActiveForRemainingRadius = [&]() {
    if (!activeRectShrinkEnabled) {
      setActiveRect(ActiveRect{0u, 0u, pushConstants.width, pushConstants.height});
      return;
    }
    setActiveRect(inflatedCenterRect(remainingSpatialRadius));
  };
  auto consumeSpatialRadius = [&](uint32_t radius) {
    if (!activeRectShrinkEnabled) {
      return;
    }
    remainingSpatialRadius = remainingSpatialRadius > radius ? remainingSpatialRadius - radius : 0u;
    setActiveForRemainingRadius();
  };
  setActiveForRemainingRadius();

  auto activeGroupsX = [&]() {
    return (std::max(pushConstants.activeWidth, 1u) + 31u) / 32u;
  };
  auto activeGroupsY = [&]() {
    return (std::max(pushConstants.activeHeight, 1u) + 7u) / 8u;
  };
  auto insertComputeBarrier = [&]() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &barrier,
      0,
      nullptr,
      0,
      nullptr
    );
  };
  auto dispatchHalation = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t sigmaMode, uint32_t component) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = sigmaMode;
    pushConstants._pad2 = component;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, halationPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();
  };
  auto dispatchHalation1D = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t value, uint32_t itemCount) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = value;
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, halationPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, (std::max(itemCount, 1u) + 31u) / 32u, 1u, 1u);
    insertComputeBarrier();
  };
  auto dispatchDiffusion = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t component) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = component;
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, diffusionPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();
  };
  auto dispatchDiffusionActiveSized = [&](
    VkDescriptorSet descriptorSet,
    uint32_t operation,
    uint32_t value1,
    uint32_t value2
  ) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = value1;
    pushConstants._pad2 = value2;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, diffusionPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();
  };
  auto dispatchDiffusionSized = [&](
    VkDescriptorSet descriptorSet,
    uint32_t operation,
    uint32_t value1,
    uint32_t value2,
    uint32_t dispatchWidth,
    uint32_t dispatchHeight
  ) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = value1;
    pushConstants._pad2 = value2;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, diffusionPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, (std::max(dispatchWidth, 1u) + 31u) / 32u, (std::max(dispatchHeight, 1u) + 7u) / 8u, 1);
    insertComputeBarrier();
  };
  auto packDiffusionGroup = [](uint32_t groupCount, uint32_t downsampleScale) {
    return (std::min(downsampleScale, 0xffffu) << 16u) | std::min(groupCount, 0xffffu);
  };
  auto dispatchDiffusionSequence = [&](
    VkDescriptorSet descriptorSet,
    const std::vector<VulkanDiffusionComponent> &components
  ) {
    dispatchDiffusion(descriptorSet, kDiffusionOpClear, 0u);
    const uint32_t componentCount = static_cast<uint32_t>(components.size());
    for (uint32_t component = 0u; component < componentCount;) {
      const uint32_t downsampleScale = diffusionDownsamplePath
        ? diffusionDownsampleScaleForSigma(blurDownsample, components[component].sigmaPx)
        : 1u;
      uint32_t groupCount = 1u;
      while (component + groupCount < componentCount &&
             groupCount < diffusionGroupSize &&
             (!diffusionDownsamplePath ||
              diffusionDownsampleScaleForSigma(blurDownsample, components[component + groupCount].sigmaPx) == downsampleScale)) {
        ++groupCount;
      }
      if (downsampleScale <= 1u) {
        if (groupCount <= 1u) {
          dispatchDiffusion(descriptorSet, kDiffusionOpBlurX, component);
          dispatchDiffusion(descriptorSet, kDiffusionOpBlurYAccumulate, component);
        } else {
          dispatchDiffusionActiveSized(descriptorSet, kDiffusionOpGroupBlurX, component, groupCount);
          dispatchDiffusionActiveSized(descriptorSet, kDiffusionOpGroupBlurYAccumulate, component, groupCount);
        }
        component += groupCount;
        continue;
      }

      const uint32_t reducedWidth = alignedReducedDimension(
        pushConstants.width,
        pushConstants.tileOriginX,
        pushConstants.fullWidth,
        downsampleScale
      );
      const uint32_t reducedHeight = alignedReducedDimension(
        pushConstants.height,
        pushConstants.tileOriginY,
        pushConstants.fullHeight,
        downsampleScale
      );
      dispatchDiffusionSized(descriptorSet, kDiffusionOpDownsample, downsampleScale, 0u, reducedWidth, reducedHeight);
      if (groupCount <= 1u) {
        const uint32_t packed = packDiffusionGroup(1u, downsampleScale);
        dispatchDiffusionSized(descriptorSet, kDiffusionOpDownsampleBlurX, component, packed, reducedWidth, reducedHeight);
        dispatchDiffusionSized(descriptorSet, kDiffusionOpDownsampleBlurY, component, packed, reducedWidth, reducedHeight);
        dispatchDiffusionActiveSized(descriptorSet, kDiffusionOpDownsampleUpsampleAccumulate, component, packed);
      } else {
        const uint32_t packed = packDiffusionGroup(groupCount, downsampleScale);
        dispatchDiffusionSized(descriptorSet, kDiffusionOpDownsampleGroupBlurX, component, packed, reducedWidth, reducedHeight);
        dispatchDiffusionSized(descriptorSet, kDiffusionOpDownsampleGroupBlurY, component, packed, reducedWidth, reducedHeight);
        dispatchDiffusionActiveSized(descriptorSet, kDiffusionOpDownsampleGroupUpsampleAccumulate, component, packed);
      }
      component += groupCount;
    }
    dispatchDiffusion(descriptorSet, kDiffusionOpResolve, 0u);
  };
  auto dispatchDir = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t component) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = component;
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dirPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();
  };
  auto dispatchGrain = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t depth) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = 0u;
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, grainPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), depth);
    insertComputeBarrier();
  };

  if (!finalProcessNegative) {
    pushConstants._pad0 = preExposureRawPath ? 1u : 0u;
    pushConstants._pad1 = colorAdaptationFlags(params);
    pushConstants._pad2 = fullFrameSourcePath ? 1u : 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, filmExposurePipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.exposureDescriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();

    if (halationBoostEnabled) {
      if (halationBoostLocalReductionEnabled) {
        dispatchHalation1D(
          coreFrame.halationDescriptorSets[10],
          kHalationOpBoostMax,
          kHalationBoostMaxChunkPixels,
          halationBoostMaxChunkCount
        );
        dispatchHalation1D(
          coreFrame.halationDescriptorSets[11],
          kHalationOpBoostReduceMax,
          halationBoostMaxChunkCount,
          1u
        );
      }
      dispatchHalation(coreFrame.halationDescriptorSets[12], kHalationOpBoostApply, 0u, 0u);
    }

    if (cameraDiffusionPath) {
      dispatchDiffusionSequence(coreFrame.diffusionDescriptorSets[0], cameraDiffusionComponents);
      consumeSpatialRadius(cameraDiffusionRadius);
      if (!halationPassEnabled) {
        dispatchDiffusion(coreFrame.diffusionDescriptorSets[1], kDiffusionOpRawToLog, 0u);
      }
    }

    if (halationPassEnabled) {
      if (halationScatterEnabled) {
        dispatchHalation(coreFrame.halationDescriptorSets[0], kHalationOpBlurX, kHalationSigmaScatterCore, 0u);
        dispatchHalation(coreFrame.halationDescriptorSets[1], kHalationOpBlurYStore, kHalationSigmaScatterCore, 0u);
        dispatchHalation(coreFrame.halationDescriptorSets[2], kHalationOpClear, 0u, 0u);
        for (uint32_t component = 0u; component < 3u; ++component) {
          dispatchHalation(coreFrame.halationDescriptorSets[3], kHalationOpBlurX, kHalationSigmaScatterTail, component);
          dispatchHalation(coreFrame.halationDescriptorSets[4], kHalationOpBlurYAccumulate, kHalationSigmaScatterTail, component);
        }
        dispatchHalation(coreFrame.halationDescriptorSets[5], kHalationOpScatterResolve, 0u, 0u);
      }
      if (halationBounceEnabled) {
        dispatchHalation(coreFrame.halationDescriptorSets[6], kHalationOpClear, 0u, 0u);
        for (uint32_t bounce = 0u; bounce < 3u; ++bounce) {
          dispatchHalation(coreFrame.halationDescriptorSets[7], kHalationOpBlurX, kHalationSigmaBounce, bounce);
          dispatchHalation(coreFrame.halationDescriptorSets[8], kHalationOpBlurYAccumulate, kHalationSigmaBounce, bounce);
        }
        dispatchHalation(coreFrame.halationDescriptorSets[9], kHalationOpBounceResolveLog, 0u, 0u);
      } else {
        dispatchHalation(coreFrame.halationDescriptorSets[9], kHalationOpRawToLog, 0u, 0u);
      }
      consumeSpatialRadius(halationRadius);
    }

    pushConstants._pad0 = 0u;
    pushConstants._pad1 = colorAdaptationFlags(params);
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, curveDevelopPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.developDescriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
  }

  if (dirPath) {
    insertComputeBarrier();
    dispatchDir(coreFrame.dirDescriptorSets[0], kDirOpCorrectionFromDensity, 0u);
    if (dirBlurPath) {
      dispatchDir(coreFrame.dirDescriptorSets[1], kDirOpBlurX, 0u);
      dispatchDir(coreFrame.dirDescriptorSets[2], kDirOpBlurYStore, 0u);
    }
    if (dirTailPath) {
      dispatchDir(coreFrame.dirDescriptorSets[3], kDirOpTailClear, 0u);
      for (uint32_t component = 0u; component < 3u; ++component) {
        dispatchDir(coreFrame.dirDescriptorSets[4], kDirOpTailBlurX, component);
        dispatchDir(coreFrame.dirDescriptorSets[5], kDirOpTailBlurYAccumulate, component);
      }
    }
    dispatchDir(coreFrame.dirDescriptorSets[6], kDirOpRedevelop, 0u);
    consumeSpatialRadius(dirRadius);
  }

  if (grainPath) {
    insertComputeBarrier();
    if (previewGrainPath) {
      dispatchGrain(coreFrame.grainDescriptorSets[0], kGrainOpPreview, 1u);
    } else if (productionGrainPath) {
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpProductionLayers, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpLayerBlurX, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpLayerBlurY, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroSource, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroBlurX, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroBlurY, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpResolveDensity, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[2], kGrainOpDensityBlurX, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[3], kGrainOpDensityBlurY, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[4], kGrainOpApplyControls, 1u);
      consumeSpatialRadius(grainRadius);
    } else if (grainSynthesisPath) {
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpSynthesisLayers, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpLayerBlurX, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpLayerBlurY, 9u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroSource, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroBlurX, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpMicroBlurY, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[1], kGrainOpSynthesisResolveDensity, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[2], kGrainOpDensityBlurX, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[3], kGrainOpDensityBlurY, 1u);
      dispatchGrain(coreFrame.grainDescriptorSets[4], kGrainOpCopyDensity, 1u);
      consumeSpatialRadius(grainRadius);
    }
  }

  if (printScanPassEnabled) {
    insertComputeBarrier();

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, printScanPipeline);
    pushConstants._pad0 = kPrintScanOpFrameConstants;
    pushConstants._pad1 = 0u;
    pushConstants._pad2 = 0u;
    const VkDescriptorSet frameConstantsSet = printDiffusionPath
      ? coreFrame.printDiffusionDescriptorSets[0]
      : coreFrame.finalDescriptorSet;
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &frameConstantsSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, 1u, 1u, 1u);
    insertComputeBarrier();

    if (printDiffusionPath) {
      pushConstants._pad0 = finalProcessNegative ? kPrintScanOpPrintRawFromNegativeLight : kPrintScanOpPrintRaw;
      pushConstants._pad1 = 0u;
      pushConstants._pad2 = 0u;
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.printDiffusionDescriptorSets[0], 0, nullptr);
      vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
      vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
      insertComputeBarrier();

      dispatchDiffusionSequence(coreFrame.diffusionDescriptorSets[2], printDiffusionComponents);
      consumeSpatialRadius(printDiffusionRadius);

      pushConstants._pad0 = kPrintScanOpFinalFromPrintRaw;
      pushConstants._pad1 = scannerPostPath ? 1u : 0u;
      pushConstants._pad2 = 0u;
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, printScanPipeline);
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.printDiffusionDescriptorSets[1], 0, nullptr);
    } else {
      pushConstants._pad0 = kPrintScanOpFull;
      pushConstants._pad1 = scannerPostPath ? 1u : 0u;
      pushConstants._pad2 = 0u;
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.finalDescriptorSet, 0, nullptr);
    }
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
  }

  auto dispatchScannerPost = [&](VkDescriptorSet descriptorSet, uint32_t operation) {
    pushConstants._pad0 = operation;
    pushConstants._pad1 = 0u;
    pushConstants._pad2 = 0u;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, scannerPostPipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, activeGroupsX(), activeGroupsY(), 1);
    insertComputeBarrier();
  };

  if (scannerPostPath) {
    insertComputeBarrier();
    if (printGlarePath) {
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[0], kScannerPostOpPrintGlareGenerate);
      if (printGlareBlurPath) {
        dispatchScannerPost(coreFrame.scannerPostDescriptorSets[1], kScannerPostOpPrintGlareBlurX);
        dispatchScannerPost(coreFrame.scannerPostDescriptorSets[2], kScannerPostOpPrintGlareBlurY);
      }
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[3], kScannerPostOpPrintGlareApply);
    }
    if (scannerBlurPath) {
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[4], kScannerPostOpScannerBlurX);
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[5], kScannerPostOpScannerBlurY);
    }
    if (scannerUnsharpPath) {
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[6], kScannerPostOpUnsharpBlurX);
      dispatchScannerPost(coreFrame.scannerPostDescriptorSets[7], kScannerPostOpUnsharpBlurY);
    }
    dispatchScannerPost(coreFrame.scannerPostDescriptorSets[8], kScannerPostOpFinalize);
    consumeSpatialRadius(scannerPostRadius);
  }

  VkPipelineStageFlags hostBarrierSourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkAccessFlags hostBarrierSourceAccess = VK_ACCESS_TRANSFER_WRITE_BIT;
  if (destinationFormatConvert) {
    VkMemoryBarrier destinationConvertInputBarrier{};
    destinationConvertInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    destinationConvertInputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    destinationConvertInputBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0,
      1,
      &destinationConvertInputBarrier,
      0,
      nullptr,
      0,
      nullptr
    );

    const uint32_t formatPushConstants[2] = {static_cast<uint32_t>(pixelCount), destinationIsHalf ? 1u : 0u};
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, formatConvertPipeline);
    vkCmdBindDescriptorSets(
      commandBuffer,
      VK_PIPELINE_BIND_POINT_COMPUTE,
      copyPipelineLayout,
      0,
      1,
      &coreFrame.destinationFormatDescriptorSet,
      0,
      nullptr
    );
    vkCmdPushConstants(commandBuffer, copyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(formatPushConstants), formatPushConstants);
    vkCmdDispatch(commandBuffer, formatGroups, 1, 1);

    VkMemoryBarrier readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      1,
      &readbackBarrier,
      0,
      nullptr,
      0,
      nullptr
    );

    VkBufferCopy destinationDownloadRegion{};
    destinationDownloadRegion.size = halfByteCount;
    vkCmdCopyBuffer(commandBuffer, coreFrame.destinationHalf.buffer, coreFrame.destinationHalfStaging.buffer, 1, &destinationDownloadRegion);
  } else {
    VkMemoryBarrier readbackBarrier{};
    readbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    readbackBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    readbackBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(
      commandBuffer,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      0,
      1,
      &readbackBarrier,
      0,
      nullptr,
      0,
      nullptr
    );

    VkBufferCopy destinationDownloadRegion{};
    destinationDownloadRegion.size = pixelStorageByteCount;
    vkCmdCopyBuffer(
      commandBuffer,
      coreFrame.destination.buffer,
      destinationIsHalf ? coreFrame.destinationHalfStaging.buffer : coreFrame.destinationStaging.buffer,
      1,
      &destinationDownloadRegion
    );
  }

  VkMemoryBarrier hostBarrier{};
  hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostBarrier.srcAccessMask = hostBarrierSourceAccess;
  hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(
    commandBuffer,
    hostBarrierSourceStage,
    VK_PIPELINE_STAGE_HOST_BIT,
    0,
    1,
    &hostBarrier,
    0,
    nullptr,
    0,
    nullptr
  );

  result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkEndCommandBuffer", result);
    return false;
  }

  VkFence fence = coreFrame.fence;
  result = vkResetFences(device, 1, &fence);
  if (result != VK_SUCCESS) {
    lastError = vkError("vkResetFences", result);
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  const PerfClock::time_point commandStart = PerfClock::now();
  diagnostics.cpuSetupMs = std::max(0.0, elapsedMilliseconds(setupStart, commandStart) - diagnostics.sourceCopyMs);
  result = backend ? backend->submit(queueIndex, submitInfo, fence) : VK_ERROR_INITIALIZATION_FAILED;
  if (result != VK_SUCCESS) {
    lastError = vkError("vkQueueSubmit", result);
    return false;
  }
  result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
  const PerfClock::time_point commandEnd = PerfClock::now();
  if (result != VK_SUCCESS) {
    lastError = vkError("vkWaitForFences", result);
    return false;
  }
  diagnostics.commandBufferMs = elapsedMilliseconds(commandStart, commandEnd);

  const PerfClock::time_point outputCopyStart = PerfClock::now();
  ScratchBuffer &readbackStaging = destinationIsHalf ? coreFrame.destinationHalfStaging : coreFrame.destinationStaging;
  const VkDeviceSize readbackByteCount = destinationIsHalf ? halfByteCount : byteCount;
  if (!readbackStaging.mapped) {
    lastError = "Vulkan destination staging buffer is not mapped.";
    return false;
  }
  if (!invalidateMappedScratchBuffer(readbackStaging, readbackByteCount, "core destination staging")) {
    return false;
  }
  if (!copyMappedBytesToWindow(
        readbackStaging.mapped,
        static_cast<size_t>(readbackByteCount),
        destination,
        window,
        width,
        height
      )) {
    lastError = "The requested render window does not fit inside the Vulkan destination image view.";
    return false;
  }
  const PerfClock::time_point outputCopyEnd = PerfClock::now();
  diagnostics.outputCopyMs = elapsedMilliseconds(outputCopyStart, outputCopyEnd);

  RendererPassDiagnostics exposurePass{};
  exposurePass.name = preExposureRawPath ? "spektrafilm_raw_exposure" : "spektrafilm_film_exposure";
  exposurePass.width = static_cast<uint32_t>(width);
  exposurePass.height = static_cast<uint32_t>(height);
  exposurePass.threadgroupWidth = 32;
  exposurePass.threadgroupHeight = 8;
  exposurePass.estimatedBytes = static_cast<uint64_t>(pixelStorageByteCount) * 2u;
  diagnostics.passes.push_back(exposurePass);

  auto addComputePassDepth = [&](const std::string &name, uint32_t depth) {
    RendererPassDiagnostics pass{};
    pass.name = name;
    pass.width = static_cast<uint32_t>(width);
    pass.height = static_cast<uint32_t>(height);
    pass.depth = depth;
    pass.threadgroupWidth = 32;
    pass.threadgroupHeight = 8;
    pass.estimatedBytes = depth == 9u
      ? static_cast<uint64_t>(grainLayerByteCount) * 2u
      : static_cast<uint64_t>(pixelStorageByteCount) * 2u;
    diagnostics.passes.push_back(pass);
  };
  auto addComputePass = [&](const std::string &name) {
    addComputePassDepth(name, 1u);
  };
  auto addComputePassSize = [&](
    const std::string &name,
    uint32_t passWidth,
    uint32_t passHeight,
    uint32_t depth,
    uint64_t estimatedBytes
  ) {
    RendererPassDiagnostics pass{};
    pass.name = name;
    pass.width = passWidth;
    pass.height = passHeight;
    pass.depth = depth;
    pass.threadgroupWidth = 32;
    pass.threadgroupHeight = 8;
    pass.estimatedBytes = estimatedBytes;
    diagnostics.passes.push_back(pass);
  };
  auto addDiffusionPasses = [&](const char *prefix, const std::vector<VulkanDiffusionComponent> &components) {
    const uint32_t localWidth = static_cast<uint32_t>(width);
    const uint32_t localHeight = static_cast<uint32_t>(height);
    const uint64_t fullBytes = static_cast<uint64_t>(pixelStorageByteCount) * 2u;
    addComputePass(std::string(prefix) + "_diffusion_clear");
    for (uint32_t component = 0u; component < static_cast<uint32_t>(components.size());) {
      const uint32_t downsampleScale = diffusionDownsamplePath
        ? diffusionDownsampleScaleForSigma(blurDownsample, components[component].sigmaPx)
        : 1u;
      uint32_t groupCount = 1u;
      while (component + groupCount < static_cast<uint32_t>(components.size()) &&
             groupCount < diffusionGroupSize &&
             (!diffusionDownsamplePath ||
              diffusionDownsampleScaleForSigma(blurDownsample, components[component + groupCount].sigmaPx) == downsampleScale)) {
        ++groupCount;
      }
      if (downsampleScale <= 1u) {
        if (groupCount <= 1u) {
          addComputePass(std::string(prefix) + "_diffusion_blur_x");
          addComputePass(std::string(prefix) + "_diffusion_blur_y_accumulate");
        } else {
          addComputePass(std::string(prefix) + "_diffusion_group_blur_x");
          addComputePass(std::string(prefix) + "_diffusion_group_blur_y_accumulate");
        }
        component += groupCount;
        continue;
      }
      const uint32_t reducedWidth = alignedReducedDimension(
        localWidth,
        activeTileContext.enabled ? activeTileContext.tileOriginX : 0u,
        coordinateWidth,
        downsampleScale
      );
      const uint32_t reducedHeight = alignedReducedDimension(
        localHeight,
        activeTileContext.enabled ? activeTileContext.tileOriginY : 0u,
        coordinateHeight,
        downsampleScale
      );
      const uint64_t reducedBytes =
        static_cast<uint64_t>(reducedWidth) * static_cast<uint64_t>(reducedHeight) * 4u * sizeof(float) * 2u;
      const std::string base = std::string(prefix) + "_diffusion_downsample";
      addComputePassSize(base, reducedWidth, reducedHeight, 1u, reducedBytes);
      if (groupCount <= 1u) {
        addComputePassSize(base + "_blur_x", reducedWidth, reducedHeight, 1u, reducedBytes);
        addComputePassSize(base + "_blur_y", reducedWidth, reducedHeight, 1u, reducedBytes);
        addComputePassSize(base + "_upsample_accumulate", localWidth, localHeight, 1u, fullBytes);
      } else {
        const uint64_t groupBytes = reducedBytes * static_cast<uint64_t>(groupCount);
        addComputePassSize(base + "_group_blur_x", reducedWidth, reducedHeight, groupCount, groupBytes);
        addComputePassSize(base + "_group_blur_y", reducedWidth, reducedHeight, groupCount, groupBytes);
        addComputePassSize(base + "_group_upsample_accumulate", localWidth, localHeight, groupCount, fullBytes);
      }
      component += groupCount;
    }
    addComputePass(std::string(prefix) + "_diffusion_resolve");
  };
  if (cameraDiffusionPath) {
    addDiffusionPasses("spektrafilm_camera", cameraDiffusionComponents);
    if (!halationPassEnabled) {
      addComputePass("spektrafilm_raw_to_log_raw");
    }
  }
  if (halationPassEnabled) {
    if (halationBoostEnabled) {
      if (halationBoostLocalReductionEnabled) {
        addComputePass("spektrafilm_halation_boost_max");
        addComputePass("spektrafilm_halation_boost_reduce_max");
      }
      addComputePass("spektrafilm_halation_boost_apply");
    }
    if (halationScatterEnabled) {
      addComputePass("spektrafilm_halation_scatter_core_blur_x");
      addComputePass("spektrafilm_halation_scatter_core_blur_y");
      addComputePass("spektrafilm_halation_scatter_tail_clear");
      for (uint32_t component = 0u; component < 3u; ++component) {
        addComputePass("spektrafilm_halation_scatter_tail_blur_x");
        addComputePass("spektrafilm_halation_scatter_tail_blur_y_accumulate");
      }
      addComputePass("spektrafilm_halation_scatter_resolve");
    }
    if (halationBounceEnabled) {
      addComputePass("spektrafilm_halation_bounce_clear");
      for (uint32_t bounce = 0u; bounce < 3u; ++bounce) {
        addComputePass("spektrafilm_halation_bounce_blur_x");
        addComputePass("spektrafilm_halation_bounce_blur_y_accumulate");
      }
      addComputePass("spektrafilm_halation_resolve_log_raw");
    } else {
      addComputePass("spektrafilm_raw_to_log_raw");
    }
  }

  RendererPassDiagnostics developPass{};
  developPass.name = "spektrafilm_curve_develop";
  developPass.width = static_cast<uint32_t>(width);
  developPass.height = static_cast<uint32_t>(height);
  developPass.threadgroupWidth = 32;
  developPass.threadgroupHeight = 8;
  developPass.estimatedBytes = static_cast<uint64_t>(pixelStorageByteCount) * 2u;
  diagnostics.passes.push_back(developPass);

  if (dirPath) {
    addComputePass("spektrafilm_dir_correction_from_density");
    if (dirBlurPath) {
      addComputePass("spektrafilm_dir_blur_x");
      addComputePass("spektrafilm_dir_blur_y");
    }
    if (dirTailPath) {
      addComputePass("spektrafilm_dir_tail_clear");
      for (uint32_t component = 0u; component < 3u; ++component) {
        addComputePass("spektrafilm_dir_tail_blur_x");
        addComputePass("spektrafilm_dir_tail_blur_y_accumulate");
      }
    }
    addComputePass("spektrafilm_dir_redevelop");
  }

  if (previewGrainPath) {
    addComputePass("spektrafilm_preview_grain_from_density");
  } else if (productionGrainPath) {
    addComputePassDepth("spektrafilm_production_grain_layers", 9u);
    addComputePassDepth("spektrafilm_grain_layer_blur_x", 9u);
    addComputePassDepth("spektrafilm_grain_layer_blur_y", 9u);
    addComputePass("spektrafilm_grain_microstructure_source");
    addComputePass("spektrafilm_grain_micro_blur_x");
    addComputePass("spektrafilm_grain_micro_blur_y");
    addComputePass("spektrafilm_grain_resolve_density");
    addComputePass("spektrafilm_grain_density_blur_x");
    addComputePass("spektrafilm_grain_density_blur_y");
    addComputePass("spektrafilm_grain_apply_controls");
  } else if (grainSynthesisPath) {
    addComputePassDepth("spektrafilm_grain_synthesis_layers_from_density", 9u);
    addComputePassDepth("spektrafilm_grain_layer_blur_x", 9u);
    addComputePassDepth("spektrafilm_grain_layer_blur_y", 9u);
    addComputePass("spektrafilm_grain_microstructure_source");
    addComputePass("spektrafilm_grain_micro_blur_x");
    addComputePass("spektrafilm_grain_micro_blur_y");
    addComputePass("spektrafilm_grain_synthesis_resolve_density");
    addComputePass("spektrafilm_grain_density_blur_x");
    addComputePass("spektrafilm_grain_density_blur_y");
    addComputePass("spektrafilm_grain_synthesis_copy_density");
  }

  if (printScanPassEnabled) {
    if (printDiffusionPath) {
      addComputePass(finalProcessNegative ? "spektrafilm_print_raw_from_negative_light" : "spektrafilm_print_raw_from_film_density");
      addDiffusionPasses("spektrafilm_print", printDiffusionComponents);
      addComputePass("spektrafilm_final_from_print_raw");
    } else {
      addComputePass("spektrafilm_print_scan");
    }
  }

  if (scannerPostPath) {
    if (printGlarePath) {
      addComputePass("spektrafilm_print_glare_generate");
      if (printGlareBlurPath) {
        addComputePass("spektrafilm_print_glare_blur_x");
        addComputePass("spektrafilm_print_glare_blur_y");
      }
      addComputePass("spektrafilm_print_glare_apply");
    }
    if (scannerBlurPath) {
      addComputePass("spektrafilm_scanner_blur_x");
      addComputePass("spektrafilm_scanner_blur_y");
    }
    if (scannerUnsharpPath) {
      addComputePass("spektrafilm_scanner_unsharp_blur_x");
      addComputePass("spektrafilm_scanner_unsharp_blur_y");
    }
    addComputePass("spektrafilm_scanner_finalize");
  }

  refreshTransientBudgetEntry();
  enforceTransientBudget();
  updateSharedDiagnostics();
  return true;
}

bool VulkanRenderer::Impl::computeHalationBoostMilestone(
  const ImageView &source,
  const RenderWindow &window,
  const RenderParams &params,
  uint32_t centerTileWidth,
  uint32_t centerTileHeight
) {
  const int32_t fullWidth = window.x2 - window.x1;
  const int32_t fullHeight = window.y2 - window.y1;
  if (fullWidth <= 0 || fullHeight <= 0) {
    return true;
  }
  if (!windowFitsView(source, window, fullWidth, fullHeight)) {
    lastError = "The requested halation boost milestone window does not fit inside the Vulkan source image view.";
    return false;
  }

  centerTileWidth = std::max(centerTileWidth, 1u);
  centerTileHeight = std::max(centerTileHeight, 1u);
  const uint32_t tileColumns =
    (static_cast<uint32_t>(fullWidth) + centerTileWidth - 1u) / centerTileWidth;
  const uint32_t tileRows =
    (static_cast<uint32_t>(fullHeight) + centerTileHeight - 1u) / centerTileHeight;

  if (!prepareStaticFilmResources(params, false, false) || !ensureCoreFrameResources()) {
    return false;
  }

  std::array<float, kCoreFrameFloatCount> frameFloats{};
  frameFloats[75] = params.halationBoostEv;
  frameFloats[76] = params.halationBoostRange;
  frameFloats[77] = params.halationProtectEv;
  if (!uploadScratchBuffer(
        coreFrame.frameFloats,
        frameFloats.data(),
        static_cast<VkDeviceSize>(frameFloats.size() * sizeof(float)),
        "core halation boost milestone frame params"
      )) {
    return false;
  }

  VkDescriptorBufferInfo logExposureBufferInfo{};
  logExposureBufferInfo.buffer = staticFilm.logExposure.buffer;
  logExposureBufferInfo.offset = 0;
  logExposureBufferInfo.range = staticFilm.logExposure.capacity;
  VkDescriptorBufferInfo densityCurvesBufferInfo{};
  densityCurvesBufferInfo.buffer = staticFilm.densityCurves.buffer;
  densityCurvesBufferInfo.offset = 0;
  densityCurvesBufferInfo.range = staticFilm.densityCurves.capacity;
  VkDescriptorBufferInfo inputToSrgbBufferInfo{};
  inputToSrgbBufferInfo.buffer = staticFilm.inputToSrgb.buffer;
  inputToSrgbBufferInfo.offset = 0;
  inputToSrgbBufferInfo.range = staticFilm.inputToSrgb.capacity;
  VkDescriptorBufferInfo colorDecodeLutsBufferInfo{};
  colorDecodeLutsBufferInfo.buffer = staticFilm.colorDecodeLuts.buffer;
  colorDecodeLutsBufferInfo.offset = 0;
  colorDecodeLutsBufferInfo.range = staticFilm.colorDecodeLuts.capacity;
  VkDescriptorBufferInfo colorTransferKindsBufferInfo{};
  colorTransferKindsBufferInfo.buffer = staticFilm.colorTransferKinds.buffer;
  colorTransferKindsBufferInfo.offset = 0;
  colorTransferKindsBufferInfo.range = staticFilm.colorTransferKinds.capacity;
  VkDescriptorBufferInfo mallettRawMatrixBufferInfo{};
  mallettRawMatrixBufferInfo.buffer = staticFilm.mallettRawMatrix.buffer;
  mallettRawMatrixBufferInfo.offset = 0;
  mallettRawMatrixBufferInfo.range = staticFilm.mallettRawMatrix.capacity;
  VkDescriptorBufferInfo inputToReferenceXyzBufferInfo{};
  inputToReferenceXyzBufferInfo.buffer = staticFilm.inputToReferenceXyz.buffer;
  inputToReferenceXyzBufferInfo.offset = 0;
  inputToReferenceXyzBufferInfo.range = staticFilm.inputToReferenceXyz.capacity;
  VkDescriptorBufferInfo hanatosRawResponseBufferInfo{};
  hanatosRawResponseBufferInfo.buffer = staticFilm.hanatosRawResponse.buffer;
  hanatosRawResponseBufferInfo.offset = 0;
  hanatosRawResponseBufferInfo.range = staticFilm.hanatosRawResponse.capacity;

  float globalMaxRaw = 0.0f;
  for (uint32_t tileY = 0u; tileY < tileRows; ++tileY) {
    for (uint32_t tileX = 0u; tileX < tileColumns; ++tileX) {
      const int32_t tileX0 = window.x1 + static_cast<int32_t>(tileX * centerTileWidth);
      const int32_t tileY0 = window.y1 + static_cast<int32_t>(tileY * centerTileHeight);
      const int32_t tileX1 = std::min(window.x2, tileX0 + static_cast<int32_t>(centerTileWidth));
      const int32_t tileY1 = std::min(window.y2, tileY0 + static_cast<int32_t>(centerTileHeight));
      const int32_t tileWidth = tileX1 - tileX0;
      const int32_t tileHeight = tileY1 - tileY0;
      if (tileWidth <= 0 || tileHeight <= 0) {
        continue;
      }

      const uint64_t pixelCount =
        static_cast<uint64_t>(tileWidth) * static_cast<uint64_t>(tileHeight);
      const uint64_t byteCount64 = pixelCount * 4u * sizeof(float);
      if (byteCount64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max()) ||
          byteCount64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        lastError = "The Vulkan halation boost milestone tile is too large.";
        return false;
      }
      const VkDeviceSize byteCount = static_cast<VkDeviceSize>(byteCount64);
      const uint32_t chunkCount = static_cast<uint32_t>(
        (pixelCount + kHalationBoostMaxChunkPixels - 1u) / kHalationBoostMaxChunkPixels
      );
      const VkDeviceSize chunkByteCount = static_cast<VkDeviceSize>(
        std::max<uint64_t>(chunkCount, 1u) * 4u * sizeof(float)
      );

      if (!ensureUploadScratchBuffer(
            coreFrame.sourceStaging,
            byteCount,
            "core halation boost milestone source staging",
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
          ) ||
          !ensurePrivateScratchBuffer(coreFrame.source, byteCount, "core halation boost milestone source") ||
          !ensurePrivateScratchBuffer(coreFrame.filmRaw, byteCount, "core halation boost milestone film raw") ||
          !ensurePrivateScratchBuffer(
            coreFrame.halationBoostChunks,
            chunkByteCount,
            "core halation boost milestone chunks"
          ) ||
          !ensurePrivateScratchBuffer(
            coreFrame.halationBoostInfo,
            4u * sizeof(float),
            "core halation boost milestone info"
          ) ||
          !ensureReadbackScratchBuffer(
            coreFrame.halationBoostInfoReadback,
            4u * sizeof(float),
            "core halation boost milestone info readback",
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
          )) {
        return false;
      }

      if (!coreFrame.sourceStaging.mapped) {
        lastError = "Vulkan halation boost milestone source staging buffer is not mapped.";
        return false;
      }
      const RenderWindow tileWindow{tileX0, tileY0, tileX1, tileY1};
      if (!copySourceToFloatStaging(
            source,
            tileWindow,
            tileWidth,
            tileHeight,
            static_cast<float *>(coreFrame.sourceStaging.mapped)
          )) {
        lastError = "The requested halation boost milestone tile does not fit inside the Vulkan source image view.";
        return false;
      }
      if (!flushMappedScratchBuffer(coreFrame.sourceStaging, byteCount, "core halation boost milestone source staging")) {
        return false;
      }

      VkDescriptorBufferInfo sourceBufferInfo{};
      sourceBufferInfo.buffer = coreFrame.source.buffer;
      sourceBufferInfo.offset = 0;
      sourceBufferInfo.range = byteCount;
      VkDescriptorBufferInfo filmRawBufferInfo{};
      filmRawBufferInfo.buffer = coreFrame.filmRaw.buffer;
      filmRawBufferInfo.offset = 0;
      filmRawBufferInfo.range = byteCount;
      VkDescriptorBufferInfo halationBoostChunksBufferInfo{};
      halationBoostChunksBufferInfo.buffer = coreFrame.halationBoostChunks.buffer;
      halationBoostChunksBufferInfo.offset = 0;
      halationBoostChunksBufferInfo.range = coreFrame.halationBoostChunks.capacity;
      VkDescriptorBufferInfo halationBoostInfoBufferInfo{};
      halationBoostInfoBufferInfo.buffer = coreFrame.halationBoostInfo.buffer;
      halationBoostInfoBufferInfo.offset = 0;
      halationBoostInfoBufferInfo.range = coreFrame.halationBoostInfo.capacity;
      VkDescriptorBufferInfo frameFloatsBufferInfo{};
      frameFloatsBufferInfo.buffer = coreFrame.frameFloats.buffer;
      frameFloatsBufferInfo.offset = 0;
      frameFloatsBufferInfo.range = coreFrame.frameFloats.capacity;

      std::array<VkWriteDescriptorSet, 20> writes{};
      uint32_t writeCount = 0;
      auto writeStorageBuffer = [&](VkDescriptorSet set, uint32_t binding, const VkDescriptorBufferInfo &bufferInfo) {
        VkWriteDescriptorSet &write = writes[writeCount++];
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        write.pBufferInfo = &bufferInfo;
      };
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 0, sourceBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 1, filmRawBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 2, logExposureBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 3, densityCurvesBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 4, inputToSrgbBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 5, colorDecodeLutsBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 6, colorTransferKindsBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 7, mallettRawMatrixBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 8, inputToReferenceXyzBufferInfo);
      writeStorageBuffer(coreFrame.exposureDescriptorSet, 9, hanatosRawResponseBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[10], 0, filmRawBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[10], 1, halationBoostChunksBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[10], 2, halationBoostInfoBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[10], 3, filmRawBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[10], 27, frameFloatsBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[11], 0, halationBoostChunksBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[11], 1, halationBoostInfoBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[11], 2, halationBoostInfoBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[11], 3, filmRawBufferInfo);
      writeStorageBuffer(coreFrame.halationDescriptorSets[11], 27, frameFloatsBufferInfo);
      vkUpdateDescriptorSets(device, writeCount, writes.data(), 0, nullptr);

      VkCommandBuffer commandBuffer = coreFrame.commandBuffer;
      VkResult result = vkResetCommandBuffer(commandBuffer, 0);
      if (result != VK_SUCCESS) {
        lastError = vkError("vkResetCommandBuffer(halation boost milestone)", result);
        return false;
      }

      VkCommandBufferBeginInfo beginInfo{};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
      if (result != VK_SUCCESS) {
        lastError = vkError("vkBeginCommandBuffer(halation boost milestone)", result);
        return false;
      }

      VkMemoryBarrier hostInputBarrier{};
      hostInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      hostInputBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      hostInputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &hostInputBarrier,
        0,
        nullptr,
        0,
        nullptr
      );

      VkBufferCopy sourceUploadRegion{};
      sourceUploadRegion.size = byteCount;
      vkCmdCopyBuffer(commandBuffer, coreFrame.sourceStaging.buffer, coreFrame.source.buffer, 1, &sourceUploadRegion);

      VkMemoryBarrier sourceUploadBarrier{};
      sourceUploadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      sourceUploadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      sourceUploadBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0,
        1,
        &sourceUploadBarrier,
        0,
        nullptr,
        0,
        nullptr
      );

      VulkanCorePushConstants pushConstants{};
      pushConstants.width = static_cast<uint32_t>(tileWidth);
      pushConstants.height = static_cast<uint32_t>(tileHeight);
      pushConstants.filmExposureEv = params.filmExposureEv;
      pushConstants.filmGamma = params.filmGamma *
        (params.filmPushPullMode == PushPullMode::Standard ? filmPushPullGamma(params.filmPushPullStops) : 1.0f);
      pushConstants.exposureCount = staticFilm.exposureCount;
      pushConstants.inputColorSpace = static_cast<int32_t>(params.inputColorSpace);
      pushConstants.rgbToRawMethod = static_cast<int32_t>(params.rgbToRawMethod);
      pushConstants.colorSpaceCount = kSpektraColorSpaceCount;
      pushConstants.transferLutSize = kSpektraColorTransferLutSize;
      pushConstants.colorDecodeMin = colorDecodeLutMin();
      pushConstants.colorDecodeMax = colorDecodeLutMax();
      pushConstants.hanatosWidth = staticFilm.hanatosWidth;
      pushConstants.hanatosHeight = staticFilm.hanatosHeight;
      pushConstants.filmPushPullMode = static_cast<int32_t>(params.filmPushPullMode);
      pushConstants.filmPushPullStops = params.filmPushPullStops;
      pushConstants.fullWidth = static_cast<uint32_t>(fullWidth);
      pushConstants.fullHeight = static_cast<uint32_t>(fullHeight);
      pushConstants.tileOriginX = static_cast<uint32_t>(tileX0 - window.x1);
      pushConstants.tileOriginY = static_cast<uint32_t>(tileY0 - window.y1);

      const uint32_t groupsX = (pushConstants.width + 31u) / 32u;
      const uint32_t groupsY = (pushConstants.height + 7u) / 8u;
      auto insertComputeBarrier = [&]() {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(
          commandBuffer,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          0,
          1,
          &barrier,
          0,
          nullptr,
          0,
          nullptr
        );
      };

      pushConstants._pad0 = 1u;
      pushConstants._pad1 = colorAdaptationFlags(params);
      pushConstants._pad2 = 0u;
      vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, filmExposurePipeline);
      vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &coreFrame.exposureDescriptorSet, 0, nullptr);
      vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
      vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);
      insertComputeBarrier();

      auto dispatchHalation1D = [&](VkDescriptorSet descriptorSet, uint32_t operation, uint32_t value, uint32_t itemCount) {
        pushConstants._pad0 = operation;
        pushConstants._pad1 = value;
        pushConstants._pad2 = 0u;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, halationPipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, corePipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer, corePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, (std::max(itemCount, 1u) + 31u) / 32u, 1u, 1u);
        insertComputeBarrier();
      };
      dispatchHalation1D(
        coreFrame.halationDescriptorSets[10],
        kHalationOpBoostMax,
        kHalationBoostMaxChunkPixels,
        chunkCount
      );
      dispatchHalation1D(
        coreFrame.halationDescriptorSets[11],
        kHalationOpBoostReduceMax,
        chunkCount,
        1u
      );

      VkMemoryBarrier readbackInputBarrier{};
      readbackInputBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      readbackInputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      readbackInputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        1,
        &readbackInputBarrier,
        0,
        nullptr,
        0,
        nullptr
      );

      VkBufferCopy boostInfoReadbackRegion{};
      boostInfoReadbackRegion.size = 4u * sizeof(float);
      vkCmdCopyBuffer(
        commandBuffer,
        coreFrame.halationBoostInfo.buffer,
        coreFrame.halationBoostInfoReadback.buffer,
        1,
        &boostInfoReadbackRegion
      );

      VkMemoryBarrier hostReadbackBarrier{};
      hostReadbackBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      hostReadbackBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      hostReadbackBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
      vkCmdPipelineBarrier(
        commandBuffer,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0,
        1,
        &hostReadbackBarrier,
        0,
        nullptr,
        0,
        nullptr
      );

      result = vkEndCommandBuffer(commandBuffer);
      if (result != VK_SUCCESS) {
        lastError = vkError("vkEndCommandBuffer(halation boost milestone)", result);
        return false;
      }

      VkFence fence = coreFrame.fence;
      result = vkResetFences(device, 1, &fence);
      if (result != VK_SUCCESS) {
        lastError = vkError("vkResetFences(halation boost milestone)", result);
        return false;
      }

      VkSubmitInfo submitInfo{};
      submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submitInfo.commandBufferCount = 1;
      submitInfo.pCommandBuffers = &commandBuffer;
      result = backend ? backend->submit(queueIndex, submitInfo, fence) : VK_ERROR_INITIALIZATION_FAILED;
      if (result != VK_SUCCESS) {
        lastError = vkError("vkQueueSubmit(halation boost milestone)", result);
        return false;
      }
      result = vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
      if (result != VK_SUCCESS) {
        lastError = vkError("vkWaitForFences(halation boost milestone)", result);
        return false;
      }
      if (!coreFrame.halationBoostInfoReadback.mapped) {
        lastError = "Vulkan halation boost milestone readback buffer is not mapped.";
        return false;
      }
      if (!invalidateMappedScratchBuffer(
            coreFrame.halationBoostInfoReadback,
            4u * sizeof(float),
            "core halation boost milestone info readback"
          )) {
        return false;
      }
      const auto *boostInfo = static_cast<const float *>(coreFrame.halationBoostInfoReadback.mapped);
      globalMaxRaw = std::max(globalMaxRaw, boostInfo[0]);
    }
  }

  const float rawX0 = std::clamp(
    0.184f * std::exp2(params.halationProtectEv),
    0.0f,
    globalMaxRaw
  );
  const float boostRange = std::clamp(params.halationBoostRange, 0.0f, 1.0f);
  const float a = std::pow(28.0f, 1.0f - boostRange);
  const float x0 = globalMaxRaw > 0.0f ? rawX0 / globalMaxRaw : 1.0f;
  const float denom = std::exp(a * (1.0f - x0)) - a * (1.0f - x0) - 1.0f;
  const float k = (globalMaxRaw > 0.0f && rawX0 < globalMaxRaw && denom > 1.0e-10f)
    ? (std::exp2(std::max(params.halationBoostEv, 0.0f)) - 1.0f) / denom
    : 0.0f;
  const std::array<float, 4> milestoneInfo = {globalMaxRaw, rawX0, a, k};
  return uploadScratchBuffer(
    coreFrame.tiledHalationBoostInfo,
    milestoneInfo.data(),
    static_cast<VkDeviceSize>(milestoneInfo.size() * sizeof(float)),
    "core tiled halation boost info"
  );
}

bool VulkanRenderer::Impl::renderTiledBootstrap(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  const RenderParams &params,
  double time
) {
  std::lock_guard<std::recursive_mutex> lock(renderMutex);
  diagnostics = {};
  diagnostics.renderSerialized = true;
  diagnostics.tiledRendering = true;
  lastError.clear();
  updateSharedDiagnostics();

  const int32_t fullWidth = window.x2 - window.x1;
  const int32_t fullHeight = window.y2 - window.y1;
  if (fullWidth <= 0 || fullHeight <= 0) {
    return true;
  }
  if (!isSupportedRgba(source, destination)) {
    lastError = "Only RGBA 16-bit half and 32-bit float images are supported by the Windows Vulkan path.";
    return false;
  }
  if (!windowFitsView(source, window, fullWidth, fullHeight) ||
      !windowFitsView(destination, window, fullWidth, fullHeight)) {
    lastError = "The requested tiled render window does not fit inside the Vulkan source or destination image view.";
    return false;
  }
  if (envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_GRAIN_PASS", true) &&
      params.grainEnabled &&
      params.grainModel == GrainModel::GrainSynthesis) {
    lastError =
      "Vulkan tiled rendering requires a full-frame pre-grain density milestone for Grain Synthesis parity; "
      "use GPU Render Tiling: Full-frame for this setting until that milestone path is implemented.";
    return false;
  }
  RenderParams tileParams = params;
  std::vector<float> fullSourcePixels;
  if (params.autoExposure) {
    const uint64_t fullPixelCount =
      static_cast<uint64_t>(fullWidth) * static_cast<uint64_t>(fullHeight);
    if (fullPixelCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 4u / sizeof(float))) {
      lastError = "The Vulkan tiled auto exposure window is too large.";
      return false;
    }
    fullSourcePixels.resize(static_cast<size_t>(fullPixelCount) * 4u);
    if (!copySourceToFloatStaging(source, window, fullWidth, fullHeight, fullSourcePixels.data())) {
      lastError = "The requested tiled auto exposure window does not fit inside the Vulkan source image view.";
      return false;
    }
    tileParams.filmExposureEv += measureAutoExposureEv(fullSourcePixels.data(), fullWidth, fullHeight, params);
    tileParams.autoExposure = false;
  }

  uint32_t centerTileWidth = std::min<uint32_t>(
    static_cast<uint32_t>(fullWidth),
    envTileDimension("SPEKTRAFILM_VULKAN_TILE_WIDTH", kDefaultVulkanTileWidth)
  );
  uint32_t centerTileHeight = std::min<uint32_t>(
    static_cast<uint32_t>(fullHeight),
    envTileDimension("SPEKTRAFILM_VULKAN_TILE_HEIGHT", kDefaultVulkanTileHeight)
  );
  const uint32_t overlap = estimateVulkanTileOverlap(params);
  auto growTileForOverlap = [&](uint32_t current, uint32_t fullDimension) {
    if (overlap <= current) {
      return current;
    }
    const uint32_t overlapSizedTile = overlap > std::numeric_limits<uint32_t>::max() / 2u
      ? std::numeric_limits<uint32_t>::max()
      : overlap * 2u;
    return std::min<uint32_t>(fullDimension, std::max(current, overlapSizedTile));
  };
  centerTileWidth = growTileForOverlap(centerTileWidth, static_cast<uint32_t>(fullWidth));
  centerTileHeight = growTileForOverlap(centerTileHeight, static_cast<uint32_t>(fullHeight));
  const uint32_t tileColumns =
    (static_cast<uint32_t>(fullWidth) + centerTileWidth - 1u) / centerTileWidth;
  const uint32_t tileRows =
    (static_cast<uint32_t>(fullHeight) + centerTileHeight - 1u) / centerTileHeight;
  const uint32_t plannedTileCount = tileColumns * tileRows;
  const bool halationBoostMilestoneEnabled =
    envFlagEnabledOrDefault("SPEKTRAFILM_VULKAN_HALATION_PASS", true) &&
    tileParams.halationEnabled &&
    tileParams.halationBoostEv > 0.0f;
  if (halationBoostMilestoneEnabled &&
      !computeHalationBoostMilestone(source, window, tileParams, centerTileWidth, centerTileHeight)) {
    const std::string milestoneError = lastError.empty() ? "unknown Vulkan halation boost milestone failure" : lastError;
    lastError = "Vulkan tiled render failed while computing the halation boost milestone: " + milestoneError;
    updateSharedDiagnostics();
    return false;
  }

  const uint64_t fullSourcePixelCount =
    static_cast<uint64_t>(fullWidth) * static_cast<uint64_t>(fullHeight);
  const uint64_t fullSourceByteCount64 = fullSourcePixelCount * 4u * sizeof(float);
  if (fullSourcePixelCount == 0u ||
      fullSourceByteCount64 > static_cast<uint64_t>(std::numeric_limits<VkDeviceSize>::max()) ||
      fullSourceByteCount64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
    lastError = "The Vulkan tiled full-frame source upload is too large.";
    updateSharedDiagnostics();
    return false;
  }
  const VkDeviceSize fullSourceByteCount = static_cast<VkDeviceSize>(fullSourceByteCount64);
  if (!ensureCoreFrameResources() ||
      !ensureUploadScratchBuffer(
        coreFrame.sourceStaging,
        fullSourceByteCount,
        "core tiled full-frame source staging",
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
      ) ||
      !ensurePrivateScratchBuffer(coreFrame.source, fullSourceByteCount, "core tiled full-frame source")) {
    updateSharedDiagnostics();
    return false;
  }
  if (!coreFrame.sourceStaging.mapped) {
    lastError = "Vulkan tiled full-frame source staging buffer is not mapped.";
    updateSharedDiagnostics();
    return false;
  }
  if (!fullSourcePixels.empty()) {
    std::memcpy(coreFrame.sourceStaging.mapped, fullSourcePixels.data(), static_cast<size_t>(fullSourceByteCount));
  } else if (!copySourceToFloatStaging(
               source,
               window,
               fullWidth,
               fullHeight,
               static_cast<float *>(coreFrame.sourceStaging.mapped)
             )) {
    lastError = "The requested tiled full-frame source upload does not fit inside the Vulkan source image view.";
    updateSharedDiagnostics();
    return false;
  }
  if (!flushMappedScratchBuffer(coreFrame.sourceStaging, fullSourceByteCount, "core tiled full-frame source staging") ||
      !copyBufferImmediate(
        coreFrame.sourceStaging.buffer,
        coreFrame.source.buffer,
        fullSourceByteCount,
        "vkQueueSubmit(tiled full-frame source upload)"
      )) {
    updateSharedDiagnostics();
    return false;
  }

  RendererDiagnostics aggregate{};
  aggregate.renderSerialized = true;
  aggregate.tiledRendering = true;
  aggregate.tileCount = plannedTileCount;
  aggregate.tileWidth = centerTileWidth;
  aggregate.tileHeight = centerTileHeight;
  aggregate.tileOverlap = overlap;
  aggregate.uploadBytes = static_cast<uint64_t>(fullSourceByteCount);

  const int32_t pixelBytes = destination.components * destination.bytesPerComponent;
  bool capturedPassList = false;
  uint32_t tileIndex = 0u;
  for (uint32_t tileY = 0u; tileY < tileRows; ++tileY) {
    for (uint32_t tileX = 0u; tileX < tileColumns; ++tileX, ++tileIndex) {
      const int32_t centerX0 = window.x1 + static_cast<int32_t>(tileX * centerTileWidth);
      const int32_t centerY0 = window.y1 + static_cast<int32_t>(tileY * centerTileHeight);
      const int32_t centerX1 = std::min(window.x2, centerX0 + static_cast<int32_t>(centerTileWidth));
      const int32_t centerY1 = std::min(window.y2, centerY0 + static_cast<int32_t>(centerTileHeight));
      const int32_t centerWidth = centerX1 - centerX0;
      const int32_t centerHeight = centerY1 - centerY0;
      if (centerWidth <= 0 || centerHeight <= 0) {
        continue;
      }

      const int32_t overlapPx = static_cast<int32_t>(std::min<uint32_t>(
        overlap,
        static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
      ));
      const int32_t workingX0 = std::max(window.x1, centerX0 - overlapPx);
      const int32_t workingY0 = std::max(window.y1, centerY0 - overlapPx);
      const int32_t workingX1 = std::min(window.x2, centerX1 + overlapPx);
      const int32_t workingY1 = std::min(window.y2, centerY1 + overlapPx);
      const int32_t workingWidth = workingX1 - workingX0;
      const int32_t workingHeight = workingY1 - workingY0;
      if (workingWidth <= 0 || workingHeight <= 0) {
        lastError = "The Vulkan tiled planner produced an empty working tile.";
        diagnostics = aggregate;
        updateSharedDiagnostics();
        return false;
      }

      const uint64_t tileByteCount =
        static_cast<uint64_t>(workingWidth) *
        static_cast<uint64_t>(workingHeight) *
        static_cast<uint64_t>(pixelBytes);
      if (tileByteCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        lastError = "The Vulkan tiled planner produced a tile larger than host address space.";
        diagnostics = aggregate;
        updateSharedDiagnostics();
        return false;
      }

      std::vector<uint8_t> tileOutput(static_cast<size_t>(tileByteCount));
      MutableImageView tileDestination{};
      tileDestination.data = tileOutput.data();
      tileDestination.x1 = workingX0;
      tileDestination.y1 = workingY0;
      tileDestination.width = workingWidth;
      tileDestination.height = workingHeight;
      tileDestination.rowBytes = workingWidth * pixelBytes;
      tileDestination.components = destination.components;
      tileDestination.bytesPerComponent = destination.bytesPerComponent;

      activeTileContext.enabled = true;
      activeTileContext.fullWidth = static_cast<uint32_t>(fullWidth);
      activeTileContext.fullHeight = static_cast<uint32_t>(fullHeight);
      activeTileContext.tileOriginX = static_cast<uint32_t>(workingX0 - window.x1);
      activeTileContext.tileOriginY = static_cast<uint32_t>(workingY0 - window.y1);
      activeTileContext.centerOriginX = static_cast<uint32_t>(centerX0 - workingX0);
      activeTileContext.centerOriginY = static_cast<uint32_t>(centerY0 - workingY0);
      activeTileContext.centerWidth = static_cast<uint32_t>(centerWidth);
      activeTileContext.centerHeight = static_cast<uint32_t>(centerHeight);
      activeTileContext.halationBoostMilestoneEnabled = halationBoostMilestoneEnabled;
      activeTileContext.fullFrameSource = true;
      const RenderWindow workingWindow{workingX0, workingY0, workingX1, workingY1};
      const bool tileOk = renderCoreBootstrap(source, tileDestination, workingWindow, tileParams, time);
      activeTileContext = {};

      const RendererDiagnostics tileDiagnostics = diagnostics;
      aggregate.cpuSetupMs += tileDiagnostics.cpuSetupMs;
      aggregate.sourceCopyMs += tileDiagnostics.sourceCopyMs;
      aggregate.commandBufferMs += tileDiagnostics.commandBufferMs;
      aggregate.outputCopyMs += tileDiagnostics.outputCopyMs;
      aggregate.staticAllocationBytes = std::max(aggregate.staticAllocationBytes, tileDiagnostics.staticAllocationBytes);
      aggregate.staticAllocationCount = std::max(aggregate.staticAllocationCount, tileDiagnostics.staticAllocationCount);
      aggregate.scratchAllocationBytes = std::max(aggregate.scratchAllocationBytes, tileDiagnostics.scratchAllocationBytes);
      aggregate.scratchAllocationCount = std::max(aggregate.scratchAllocationCount, tileDiagnostics.scratchAllocationCount);
      aggregate.sharedScratchAllocationBytes =
        std::max(aggregate.sharedScratchAllocationBytes, tileDiagnostics.sharedScratchAllocationBytes);
      aggregate.sharedScratchAllocationCount =
        std::max(aggregate.sharedScratchAllocationCount, tileDiagnostics.sharedScratchAllocationCount);
      aggregate.privateScratchAllocationBytes =
        std::max(aggregate.privateScratchAllocationBytes, tileDiagnostics.privateScratchAllocationBytes);
      aggregate.privateScratchAllocationCount =
        std::max(aggregate.privateScratchAllocationCount, tileDiagnostics.privateScratchAllocationCount);
      aggregate.uploadBytes = std::max(aggregate.uploadBytes, tileDiagnostics.uploadBytes);
      aggregate.sharedBackend = tileDiagnostics.sharedBackend;
      aggregate.sharedBackendGeneration = tileDiagnostics.sharedBackendGeneration;
      aggregate.sharedQueueCount = tileDiagnostics.sharedQueueCount;
      aggregate.transientCachedBytes = std::max(aggregate.transientCachedBytes, tileDiagnostics.transientCachedBytes);
      aggregate.transientBudgetBytes = tileDiagnostics.transientBudgetBytes;
      aggregate.passCount += tileDiagnostics.passCount;
      aggregate.sourceNoCopy = aggregate.sourceNoCopy && tileDiagnostics.sourceNoCopy;
      aggregate.destinationNoCopy = aggregate.destinationNoCopy && tileDiagnostics.destinationNoCopy;
      aggregate.passGpuTimingEnabled = aggregate.passGpuTimingEnabled || tileDiagnostics.passGpuTimingEnabled;
      aggregate.passGpuTimingAvailable = aggregate.passGpuTimingAvailable || tileDiagnostics.passGpuTimingAvailable;
      aggregate.privateScratchEnabled = aggregate.privateScratchEnabled || tileDiagnostics.privateScratchEnabled;
      aggregate.halationPath = aggregate.halationPath || tileDiagnostics.halationPath;
      aggregate.cameraDiffusionPath = aggregate.cameraDiffusionPath || tileDiagnostics.cameraDiffusionPath;
      aggregate.printDiffusionPath = aggregate.printDiffusionPath || tileDiagnostics.printDiffusionPath;
      aggregate.dirPath = aggregate.dirPath || tileDiagnostics.dirPath;
      aggregate.productionGrainPath = aggregate.productionGrainPath || tileDiagnostics.productionGrainPath;
      aggregate.grainSynthesisPath = aggregate.grainSynthesisPath || tileDiagnostics.grainSynthesisPath;
      aggregate.finalPostProcessPath = aggregate.finalPostProcessPath || tileDiagnostics.finalPostProcessPath;
      aggregate.scannerTextureIntermediates =
        aggregate.scannerTextureIntermediates || tileDiagnostics.scannerTextureIntermediates;
      aggregate.halationGroupedTail = aggregate.halationGroupedTail || tileDiagnostics.halationGroupedTail;
      aggregate.scannerMps = aggregate.scannerMps || tileDiagnostics.scannerMps;
      aggregate.grainBlurRecurrence = tileDiagnostics.grainBlurRecurrence;
      aggregate.diffusionGroupSize = tileDiagnostics.diffusionGroupSize;
      aggregate.threadgroupMode = tileDiagnostics.threadgroupMode;
      aggregate.passTimingMode = tileDiagnostics.passTimingMode;
      aggregate.blurBackend = tileDiagnostics.blurBackend;
      aggregate.blurDownsample = tileDiagnostics.blurDownsample;
      aggregate.intermediatePrecision = tileDiagnostics.intermediatePrecision;
      aggregate.diffusionClusterSigma = tileDiagnostics.diffusionClusterSigma;
      aggregate.dirTailBackend = tileDiagnostics.dirTailBackend;
      aggregate.densityCurveLookup = tileDiagnostics.densityCurveLookup;
      aggregate.spectralTransmittance = tileDiagnostics.spectralTransmittance;
      if (!capturedPassList) {
        aggregate.passes = tileDiagnostics.passes;
        capturedPassList = true;
      }

      RendererTileDiagnostics tileRecord{};
      tileRecord.index = tileIndex;
      tileRecord.outputX = centerX0;
      tileRecord.outputY = centerY0;
      tileRecord.outputWidth = centerWidth;
      tileRecord.outputHeight = centerHeight;
      tileRecord.workingX = workingX0;
      tileRecord.workingY = workingY0;
      tileRecord.workingWidth = workingWidth;
      tileRecord.workingHeight = workingHeight;
      tileRecord.overlap = overlap;
      tileRecord.allocatedBytes = transientAllocationBytes();
      tileRecord.submitMs = tileDiagnostics.commandBufferMs;
      tileRecord.passCount = tileDiagnostics.passCount;
      aggregate.tiles.push_back(tileRecord);

      if (!tileOk) {
        const std::string tileError = lastError.empty() ? "unknown Vulkan tile failure" : lastError;
        lastError = "Vulkan tiled render failed on tile " +
          std::to_string(tileIndex + 1u) + "/" + std::to_string(plannedTileCount) +
          " center=[" + std::to_string(centerX0) + "," + std::to_string(centerY0) +
          "-" + std::to_string(centerX1) + "," + std::to_string(centerY1) +
          "] working=[" + std::to_string(workingX0) + "," + std::to_string(workingY0) +
          "-" + std::to_string(workingX1) + "," + std::to_string(workingY1) +
          "]: " + tileError;
        diagnostics = aggregate;
        updateSharedDiagnostics();
        return false;
      }

      const RenderWindow centerWindow{centerX0, centerY0, centerX1, centerY1};
      if (!copyMappedBytesRegionToWindow(
            tileOutput.data(),
            tileOutput.size(),
            workingWidth,
            workingHeight,
            centerX0 - workingX0,
            centerY0 - workingY0,
            destination,
            centerWindow,
            centerWidth,
            centerHeight
          )) {
        lastError = "Vulkan tiled render failed to copy a tile center into the destination image view.";
        diagnostics = aggregate;
        updateSharedDiagnostics();
        return false;
      }
    }
  }

  diagnostics = aggregate;
  diagnostics.tiledRendering = true;
  diagnostics.tileCount = plannedTileCount;
  diagnostics.tileWidth = centerTileWidth;
  diagnostics.tileHeight = centerTileHeight;
  diagnostics.tileOverlap = overlap;
  refreshTransientBudgetEntry();
  enforceTransientBudget();
  updateSharedDiagnostics();
  return true;
}

VulkanRenderer::VulkanRenderer() : impl_(std::make_unique<Impl>()) {}

VulkanRenderer::~VulkanRenderer() = default;

bool VulkanRenderer::isAvailable() const {
  return impl_ && impl_->available;
}

const VulkanRenderDiagnostics &VulkanRenderer::lastDiagnostics() const {
  if (!lastRenderError_.empty()) {
    return lastRenderDiagnostics_;
  }
  return impl_ ? impl_->diagnostics : lastRenderDiagnostics_;
}

const std::string &VulkanRenderer::lastError() const {
  static const std::string empty;
  if (!lastRenderError_.empty()) {
    return lastRenderError_;
  }
  return impl_ ? impl_->lastError : empty;
}

void VulkanRenderer::releaseTransientResources() {
  if (impl_) {
    impl_->releaseTransientResources();
  }
}

bool VulkanRenderer::render(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  const RenderParams &params,
  double time
) {
  if (!impl_ || !impl_->available) {
    if (impl_) {
      std::lock_guard<std::recursive_mutex> lock(impl_->renderMutex);
      if (impl_->lastError.empty()) {
        impl_->lastError = "Vulkan is not available on this system.";
      }
      lastRenderDiagnostics_ = impl_->diagnostics;
      lastRenderError_ = impl_->lastError;
    }
    return false;
  }

  bool ok = false;
  if (envFlagEnabled("SPEKTRAFILM_VULKAN_COPY_PASS")) {
    ok = impl_->renderCopyValidation(source, destination, window, time);
  } else {
    const GpuRenderTilingMode tileMode = resolveVulkanTileMode(params);
    if (tileMode == GpuRenderTilingMode::Tiled) {
      ok = impl_->renderTiledBootstrap(source, destination, window, params, time);
    } else {
      ok = impl_->renderCoreBootstrap(source, destination, window, params, time);
    }
  }

  if (ok) {
    if (impl_) {
      lastRenderDiagnostics_ = impl_->diagnostics;
    }
    lastRenderError_.clear();
    return true;
  }

  if (impl_) {
    lastRenderDiagnostics_ = impl_->diagnostics;
  }
  lastRenderError_ = impl_ ? impl_->lastError : std::string();
  if (isDeviceLostError(lastRenderError_)) {
    if (impl_) {
      impl_->markBackendLost();
    }
    lastRenderError_ += " Recreated the shared Vulkan backend for the next render.";
    impl_.reset();
    impl_ = std::make_unique<Impl>();
    if (!impl_ || !impl_->available) {
      lastRenderError_ += " Vulkan device recreation failed.";
      if (impl_ && !impl_->lastError.empty()) {
        lastRenderError_ += " ";
        lastRenderError_ += impl_->lastError;
      }
    }
  }
  return false;
}

std::unique_ptr<Renderer> createVulkanRenderer() {
  return std::make_unique<VulkanRenderer>();
}

#if !defined(_WIN32)
std::unique_ptr<Renderer> createNativeRenderer() {
  return createVulkanRenderer();
}
#endif

} // namespace spektrafilm
