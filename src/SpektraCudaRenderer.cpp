#include "SpektraCudaRenderer.h"

#include "SpektraCudaKernels.cuh"
#include "SpektraRenderCore.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

namespace spektrafilm {
namespace {

// small runtime switches, mostly useful while checking perf/parity
std::string cudaZeroCopyModeFromEnv() {
  const char *value = std::getenv("SPEKTRAFILM_CUDA_ZERO_COPY");
  return value && *value ? std::string(value) : std::string();
}

std::string cudaTransferPolicyFromEnv() {
  const char *value = std::getenv("SPEKTRAFILM_CUDA_TRANSFER_POLICY");
  std::string text = value && *value ? std::string(value) : std::string("auto");
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return text;
}

bool envFlag(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value) {
    return false;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return text == "1" || text == "on" || text == "true" || text == "yes";
}

bool defaultHalationStrengthSelected(const RenderParams &params) {
  return std::abs(params.halationStrengthR - 0.05f) <= 1.0e-6f &&
    std::abs(params.halationStrengthG - 0.015f) <= 1.0e-6f &&
    std::abs(params.halationStrengthB) <= 1.0e-6f;
}

void applyProfileHalationDefaults(
  KernelParams &kernelParams,
  const RenderParams &params,
  const ProfileCurveSet &filmCurves
) {
  // Generated stock data defines the default halation shape; explicit user strengths still win.
  if (filmCurves.halationFirstSigmaUm) {
    kernelParams.halationFirstSigmaUmR = filmCurves.halationFirstSigmaUm[0];
    kernelParams.halationFirstSigmaUmG = filmCurves.halationFirstSigmaUm[1];
    kernelParams.halationFirstSigmaUmB = filmCurves.halationFirstSigmaUm[2];
  }
  if (filmCurves.halationStrength && defaultHalationStrengthSelected(params)) {
    kernelParams.halationStrengthR = filmCurves.halationStrength[0];
    kernelParams.halationStrengthG = filmCurves.halationStrength[1];
    kernelParams.halationStrengthB = filmCurves.halationStrength[2];
  }
}

uint32_t diffusionGroupSizeFromEnv() {
  const char *value = std::getenv("SPEKTRAFILM_DIFFUSION_GROUP_SIZE");
  if (!value || !*value) {
    return 2u;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (text == "1") {
    return 1u;
  }
  if (text == "4") {
    return 4u;
  }
  return 2u;
}

int32_t cudaGrainSynthesisSampleCap(int32_t width, int32_t height) {
  // Grain synthesis is a per-pixel Monte Carlo pass; cap samples by frame size so 4K/8K stays interactive.
  const char *value = std::getenv("SPEKTRAFILM_CUDA_GRAIN_SYNTHESIS_SAMPLE_CAP");
  if (value && *value) {
    std::string text(value);
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
      return static_cast<char>(std::tolower(ch));
    });
    if (text == "0" || text == "off" || text == "none" || text == "unlimited") {
      return 0;
    }
    char *end = nullptr;
    const long parsed = std::strtol(value, &end, 10);
    if (end != value && parsed > 0) {
      return std::clamp(static_cast<int32_t>(parsed), 1, 1024);
    }
  }

  const double pixels = static_cast<double>(std::max(width, 1)) * static_cast<double>(std::max(height, 1));
  const double referencePixels = 1920.0 * 1080.0;
  const int32_t cap = static_cast<int32_t>(std::lround(96.0 * referencePixels / pixels));
  return std::clamp(cap, 8, 128);
}

std::string blurDownsampleModeFromEnv() {
  const char *value = std::getenv("SPEKTRAFILM_BLUR_DOWNSAMPLE");
  std::string text = value && *value ? std::string(value) : std::string("auto");
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (text == "off" || text == "2" || text == "4" || text == "8" || text == "auto") {
    return text;
  }
  return "off";
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

bool anyDiffusionComponentDownsamples(const std::vector<KernelDiffusionComponent> &components, const std::string &mode) {
  for (const KernelDiffusionComponent &component : components) {
    if (diffusionDownsampleScaleForSigma(mode, component.sigmaPx) > 1u) {
      return true;
    }
  }
  return false;
}

double elapsedMs(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - start).count();
}

float sampleHostTransferLut(float value, int32_t colorSpace, const float *luts) {
  if (!luts || kSpektraColorTransferLutSize <= 1u) {
    return value;
  }
  const float decodeMin = colorDecodeLutMin();
  const float decodeMax = colorDecodeLutMax();
  const float range = std::max(decodeMax - decodeMin, 1.0e-6f);
  const float step = range / static_cast<float>(kSpektraColorTransferLutSize - 1u);
  const uint32_t offset = static_cast<uint32_t>(std::clamp<int32_t>(
                            colorSpace, 0, static_cast<int32_t>(kSpektraColorSpaceCount - 1u))) *
                          kSpektraColorTransferLutSize;
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
    static_cast<int32_t>(kSpektraColorSpaceCount - 1u));
  const uint32_t *transferKinds = colorTransferKinds();
  if (transferKinds && transferKinds[colorSpace] != 0u) {
    const float *decodeLuts = colorDecodeLuts();
    r = sampleHostTransferLut(r, colorSpace, decodeLuts);
    g = sampleHostTransferLut(g, colorSpace, decodeLuts);
    b = sampleHostTransferLut(b, colorSpace, decodeLuts);
  }
  const float *meterMatrices = inputMeterXyzMatrices();
  if (!meterMatrices) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
  }
  const float *matrix = meterMatrices + static_cast<size_t>(colorSpace) * 9u;
  return matrix[3] * r + matrix[4] * g + matrix[5] * b;
}

std::pair<int32_t, int32_t> autoExposurePreviewDimensions(int32_t width, int32_t height) {
  constexpr int32_t kPreviewMaxSize = 256;
  int32_t previewWidth = width;
  int32_t previewHeight = height;
  const int32_t longEdge = std::max(width, height);
  if (longEdge > kPreviewMaxSize) {
    const double scale = static_cast<double>(kPreviewMaxSize) / static_cast<double>(longEdge);
    previewWidth = std::max(1, static_cast<int32_t>(std::lround(static_cast<double>(width) * scale)));
    previewHeight = std::max(1, static_cast<int32_t>(std::lround(static_cast<double>(height) * scale)));
  }
  return {previewWidth, previewHeight};
}

float measureAutoExposureEvFromLuminance(
  std::vector<float> luminance,
  int32_t previewWidth,
  int32_t previewHeight,
  const RenderParams &params
) {
  if (luminance.empty()) {
    return 0.0f;
  }
  double meteredY = 0.0;
  if (params.autoExposureMethod == AutoExposureMethod::Median) {
    const size_t mid = luminance.size() / 2u;
    std::nth_element(luminance.begin(), luminance.begin() + static_cast<std::ptrdiff_t>(mid), luminance.end());
    meteredY = luminance[mid];
    if ((luminance.size() & 1u) == 0u && mid > 0u) {
      const float upper = luminance[mid];
      std::nth_element(
        luminance.begin(),
        luminance.begin() + static_cast<std::ptrdiff_t>(mid - 1u),
        luminance.begin() + static_cast<std::ptrdiff_t>(mid));
      meteredY = 0.5 * (static_cast<double>(luminance[mid - 1u]) + static_cast<double>(upper));
    }
  } else {
    const double normX = static_cast<double>(previewWidth) / static_cast<double>(std::max(previewWidth, previewHeight));
    const double normY = static_cast<double>(previewHeight) / static_cast<double>(std::max(previewWidth, previewHeight));
    constexpr double kSigma = 0.2;
    double weightedSum = 0.0;
    double weightSum = 0.0;
    size_t index = 0u;
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

float measureAutoExposureEv(const float *source, int32_t width, int32_t height, const RenderParams &params) {
  if (!source || width <= 0 || height <= 0) {
    return 0.0f;
  }
  const auto [previewWidth, previewHeight] = autoExposurePreviewDimensions(width, height);
  std::vector<float> luminance;
  luminance.reserve(static_cast<size_t>(previewWidth) * static_cast<size_t>(previewHeight));
  for (int32_t y = 0; y < previewHeight; ++y) {
    const int32_t sourceY = std::min(
      height - 1,
      static_cast<int32_t>((static_cast<int64_t>(y) * height) / previewHeight));
    for (int32_t x = 0; x < previewWidth; ++x) {
      const int32_t sourceX = std::min(
        width - 1,
        static_cast<int32_t>((static_cast<int64_t>(x) * width) / previewWidth));
      const float *pixel = source + (static_cast<size_t>(sourceY) * width + sourceX) * 4u;
      luminance.push_back(autoExposureMeterY(pixel[0], pixel[1], pixel[2], params));
    }
  }
  return measureAutoExposureEvFromLuminance(std::move(luminance), previewWidth, previewHeight, params);
}

std::string parentPath(const std::string &path) {
  const size_t slash = path.find_last_of("\\/");
  if (slash == std::string::npos) {
    return {};
  }
  return path.substr(0, slash);
}

std::string joinPath(const std::string &a, const std::string &b) {
  if (a.empty()) {
    return b;
  }
  const char last = a.back();
  if (last == '\\' || last == '/') {
    return a + b;
  }
  return a + "\\" + b;
}

bool fileExists(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  return file.good();
}

bool canUseContiguousFloatSourceWindow(
  const ImageView &source,
  const EffectiveRenderWindow &window,
  int32_t width
) {
  return source.bytesPerComponent == 4 &&
         source.components == 4 &&
         window.x1 == source.x1 &&
         width == source.width &&
         source.rowBytes == width * static_cast<int32_t>(4u * sizeof(float));
}

bool canUseContiguousFloatDestinationWindow(
  const MutableImageView &destination,
  const EffectiveRenderWindow &window,
  int32_t width
) {
  return destination.bytesPerComponent == 4 &&
         destination.components == 4 &&
         window.x1 == destination.x1 &&
         width == destination.width &&
         destination.rowBytes == width * static_cast<int32_t>(4u * sizeof(float));
}

const float *sourceWindowFloatPointer(const ImageView &source, const EffectiveRenderWindow &window) {
  const auto *base = static_cast<const unsigned char *>(source.data);
  const auto *row = base +
    static_cast<ptrdiff_t>(window.y1 - source.y1) * source.rowBytes;
  return reinterpret_cast<const float *>(row);
}

float *destinationWindowFloatPointer(const MutableImageView &destination, const EffectiveRenderWindow &window) {
  auto *base = static_cast<unsigned char *>(destination.data);
  auto *row = base +
    static_cast<ptrdiff_t>(window.y1 - destination.y1) * destination.rowBytes;
  return reinterpret_cast<float *>(row);
}

#if defined(_WIN32)
std::string moduleDirectory() {
  HMODULE module = nullptr;
  const auto address = reinterpret_cast<LPCSTR>(&moduleDirectory);
  if (!GetModuleHandleExA(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        address,
        &module)) {
    return {};
  }
  std::array<char, MAX_PATH> path{};
  const DWORD length = GetModuleFileNameA(module, path.data(), static_cast<DWORD>(path.size()));
  if (length == 0 || length >= path.size()) {
    return {};
  }
  return parentPath(std::string(path.data(), length));
}
#else
std::string moduleDirectory() {
  return {};
}
#endif

std::string resourcePath(const std::string &resourceFileName) {
  const std::string moduleDir = moduleDirectory();
  const std::array<std::string, 6> candidates = {
    joinPath(moduleDir, resourceFileName),
    joinPath(joinPath(joinPath(moduleDir, "Contents"), "Resources"), resourceFileName),
    joinPath(joinPath(parentPath(moduleDir), "Resources"), resourceFileName),
    resourceFileName,
    joinPath("Ninja", resourceFileName),
    joinPath(joinPath("Ninja", "generated"), resourceFileName),
  };
  for (const std::string &candidate : candidates) {
    if (fileExists(candidate)) {
      return candidate;
    }
  }
  return candidates.front();
}

std::vector<float> readFloatFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Unable to open resource: " + path);
  }
  const std::streamsize size = file.tellg();
  if (size <= 0 || (size % static_cast<std::streamsize>(sizeof(float))) != 0) {
    throw std::runtime_error("Resource has an invalid float payload size: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> values(static_cast<size_t>(size) / sizeof(float));
  if (!file.read(reinterpret_cast<char *>(values.data()), size)) {
    throw std::runtime_error("Unable to read resource: " + path);
  }
  return values;
}

} // namespace

CudaRenderer::CudaRenderer() {
  initialize();
}

CudaRenderer::~CudaRenderer() {
  releaseDeviceBuffer(sourceDevice_);
  releaseDeviceBuffer(destinationDevice_);
  releaseDeviceBuffer(autoExposurePreviewDevice_);
  releaseDeviceBuffer(scratchDeviceA_);
  releaseDeviceBuffer(scratchDeviceB_);
  releaseDeviceBuffer(diffusionGroupTempDevice_);
  releaseDeviceBuffer(diffusionReducedSourceDevice_);
  releaseDeviceBuffer(diffusionReducedTempDevice_);
  releaseDeviceBuffer(diffusionReducedBlurDevice_);
  releaseDeviceBuffer(dirTailScratchDevice_);
  releaseDeviceBuffer(paramsDevice_);
  releaseDeviceBuffer(frameConstantsDevice_);
  releaseDeviceBuffer(dirInfoDevice_);
  releaseDeviceBuffer(dirCoreBlurInfoDevice_);
  releaseDeviceBuffer(dirTailBlurInfosDevice_);
  releaseDeviceBuffer(dirCorrectedDensityCurvesDevice_);
  releaseDeviceBuffer(halationBoostInfoDevice_);
  releaseDeviceBuffer(cameraDiffusionInfoDevice_);
  releaseDeviceBuffer(cameraDiffusionComponentsDevice_);
  releaseDeviceBuffer(printDiffusionInfoDevice_);
  releaseDeviceBuffer(printDiffusionComponentsDevice_);
  releaseDeviceBuffer(grainLayerDeviceA_);
  releaseDeviceBuffer(grainLayerDeviceB_);
  releaseDeviceBuffer(grainMicroDeviceA_);
  releaseDeviceBuffer(grainMicroDeviceB_);
  releaseDeviceBuffer(spectralInfoDevice_);
  releaseDeviceBuffer(colorInfoDevice_);
  releaseDeviceBuffer(curveInfoDevice_);
  releaseDeviceBuffer(hanatosRawResponseDevice_);
  releaseDeviceBuffer(paperHanatosResponseDevice_);
  releaseDeviceBuffer(preflashPaperHanatosResponseDevice_);
  releaseDeviceBuffer(mallettBasisIlluminantDevice_);
  releaseDeviceBuffer(inputToReferenceXyzDevice_);
  releaseDeviceBuffer(inputToSrgbDevice_);
  releaseDeviceBuffer(colorDecodeLutDevice_);
  releaseDeviceBuffer(colorTransferKindDevice_);
  releaseDeviceBuffer(logExposureDevice_);
  releaseDeviceBuffer(densityCurvesDevice_);
  releaseDeviceBuffer(paperCurveInfoDevice_);
  releaseDeviceBuffer(paperLogExposureDevice_);
  releaseDeviceBuffer(paperDensityCurvesDevice_);
  releaseDeviceBuffer(filmChannelDensityDevice_);
  releaseDeviceBuffer(filmBaseDensityDevice_);
  releaseDeviceBuffer(paperLogSensitivityDevice_);
  releaseDeviceBuffer(thKg3IlluminantDevice_);
  releaseDeviceBuffer(customEnlargerFiltersDevice_);
  releaseDeviceBuffer(neutralPrintFiltersDevice_);
  releaseDeviceBuffer(academyPrinterDensityDataDevice_);
  releaseDeviceBuffer(paperScanDensityDataDevice_);
  releaseDeviceBuffer(scanIlluminantsAndCmfsDevice_);
  releaseDeviceBuffer(scanToOutputRgbDataDevice_);
  releaseDeviceBuffer(colorEncodeLutDevice_);
  releasePinnedHostBuffer(pinnedSourceStaging_);
  releasePinnedHostBuffer(pinnedDestinationStaging_);
  releasePinnedHostBuffer(pinnedAutoExposurePreview_);
}

bool CudaRenderer::initialize() {
  if (initialized_) {
    return available_;
  }
  initialized_ = true;

  CudaDeviceInfo info{};
  std::array<char, 512> error{};
  if (!spektraCudaInitialize(&deviceIndex_, &info, error.data(), error.size())) {
    lastError_ = error.data();
    available_ = false;
    return false;
  }

  diagnostics_ = RendererDiagnostics{};
  diagnostics_.backendName = "cuda";
  diagnostics_.deviceName = info.name;
  deviceName_ = info.name;
  computeCapabilityMajor_ = info.major;
  computeCapabilityMinor_ = info.minor;
  diagnostics_.cudaComputeCapabilityMajor = info.major;
  diagnostics_.cudaComputeCapabilityMinor = info.minor;
  diagnostics_.cudaTransferMode = cudaZeroCopyModeFromEnv() == "mapped" ? "mapped-requested" : "device-staging";
  available_ = true;
  return true;
}

bool CudaRenderer::isAvailable() const {
  return available_;
}

const std::string &CudaRenderer::lastError() const {
  return lastError_;
}

const RendererDiagnostics &CudaRenderer::lastDiagnostics() const {
  return diagnostics_;
}

bool CudaRenderer::ensureDeviceBuffer(DeviceBuffer &buffer, size_t bytes) {
  if (buffer.bytes >= bytes && buffer.pointer) {
    return true;
  }
  releaseDeviceBuffer(buffer);
  cudaError_t status = cudaMalloc(&buffer.pointer, bytes);
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << "cudaMalloc failed for " << bytes << " bytes: " << cudaGetErrorString(status);
    lastError_ = out.str();
    return false;
  }
  buffer.bytes = bytes;
  return true;
}

void CudaRenderer::releaseDeviceBuffer(DeviceBuffer &buffer) {
  if (buffer.pointer) {
    cudaFree(buffer.pointer);
  }
  buffer.pointer = nullptr;
  buffer.bytes = 0u;
}

bool CudaRenderer::ensurePinnedHostBuffer(PinnedHostBuffer &buffer, size_t bytes) {
  if (bytes == 0u) {
    return true;
  }
  if (buffer.bytes >= bytes && buffer.pointer) {
    return true;
  }
  releasePinnedHostBuffer(buffer);
  cudaError_t status = cudaHostAlloc(&buffer.pointer, bytes, cudaHostAllocDefault);
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << "cudaHostAlloc failed for " << bytes << " bytes: " << cudaGetErrorString(status);
    lastError_ = out.str();
    return false;
  }
  buffer.bytes = bytes;
  return true;
}

void CudaRenderer::releasePinnedHostBuffer(PinnedHostBuffer &buffer) {
  if (buffer.pointer) {
    cudaFreeHost(buffer.pointer);
  }
  buffer.pointer = nullptr;
  buffer.bytes = 0u;
}

bool CudaRenderer::uploadDeviceBytes(DeviceBuffer &buffer, const void *data, size_t bytes) {
  if (!data || bytes == 0u) {
    return true;
  }
  if (!ensureDeviceBuffer(buffer, bytes)) {
    return false;
  }
  cudaError_t status = cudaMemcpy(buffer.pointer, data, bytes, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    std::ostringstream out;
    out << "cudaMemcpy static upload failed for " << bytes << " bytes: " << cudaGetErrorString(status);
    lastError_ = out.str();
    return false;
  }
  return true;
}

bool CudaRenderer::uploadDeviceFloats(DeviceBuffer &buffer, const std::vector<float> &values) {
  return uploadDeviceBytes(buffer, values.data(), values.size() * sizeof(float));
}

bool CudaRenderer::uploadDeviceUInts(DeviceBuffer &buffer, const std::vector<uint32_t> &values) {
  return uploadDeviceBytes(buffer, values.data(), values.size() * sizeof(uint32_t));
}

bool CudaRenderer::ensureStaticResources(const RenderParams &params) {
  if (staticBuffersUploaded_ && staticResources_.validFor(params)) {
    return true;
  }

  try {
    if (hanatosSpectraData_.empty()) {
      hanatosSpectraData_ = readFloatFile(resourcePath("SpektraHanatos2025Spectra.f32"));
    }
    if (outputGamutCompressionData_.empty()) {
      outputGamutCompressionData_ = readFloatFile(resourcePath("SpektraOutputGamutCompression.f32"));
    }
  } catch (const std::exception &ex) {
    lastError_ = ex.what();
    staticBuffersUploaded_ = false;
    return false;
  }

  std::string error;
  if (!buildStaticProfileResourceData(params, hanatosSpectraData_, staticResources_, error)) {
    lastError_ = error;
    staticBuffersUploaded_ = false;
    return false;
  }
  if (outputGamutCompressionData_.size() != kSpektraOutputGamutCompressionElementCount) {
    lastError_ = "Unable to load CUDA output gamut compression data.";
    staticBuffersUploaded_ = false;
    return false;
  }
  staticResources_.colorEncodeLut.insert(
    staticResources_.colorEncodeLut.end(),
    outputGamutCompressionData_.begin(),
    outputGamutCompressionData_.end());
  staticResources_.colorEncodeLut.insert(
    staticResources_.colorEncodeLut.end(),
    colorTransferParams(),
    colorTransferParams() + static_cast<size_t>(kSpektraColorSpaceCount));

  if (!uploadDeviceStruct(spectralInfoDevice_, staticResources_.spectralInfo) ||
      !uploadDeviceStruct(colorInfoDevice_, staticResources_.colorInfo) ||
      !uploadDeviceStruct(curveInfoDevice_, staticResources_.curveInfo) ||
      !uploadDeviceFloats(hanatosRawResponseDevice_, staticResources_.hanatosRawResponse) ||
      !uploadDeviceFloats(paperHanatosResponseDevice_, staticResources_.paperHanatosResponse) ||
      !uploadDeviceFloats(preflashPaperHanatosResponseDevice_, staticResources_.preflashPaperHanatosResponse) ||
      !uploadDeviceBytes(
        mallettBasisIlluminantDevice_,
        staticResources_.mallettBasisIlluminant.data(),
        staticResources_.mallettBasisIlluminant.size() * sizeof(float)) ||
      !uploadDeviceFloats(inputToReferenceXyzDevice_, staticResources_.inputToReferenceXyz) ||
      !uploadDeviceFloats(inputToSrgbDevice_, staticResources_.inputToSrgb) ||
      !uploadDeviceFloats(colorDecodeLutDevice_, staticResources_.colorDecodeLut) ||
      !uploadDeviceUInts(colorTransferKindDevice_, staticResources_.colorTransferKind) ||
      !uploadDeviceFloats(logExposureDevice_, staticResources_.logExposure) ||
      !uploadDeviceFloats(densityCurvesDevice_, staticResources_.densityCurves) ||
      !uploadDeviceStruct(paperCurveInfoDevice_, staticResources_.paperCurveInfo) ||
      !uploadDeviceFloats(paperLogExposureDevice_, staticResources_.paperLogExposure) ||
      !uploadDeviceFloats(paperDensityCurvesDevice_, staticResources_.paperDensityCurves) ||
      !uploadDeviceFloats(filmChannelDensityDevice_, staticResources_.filmChannelDensity) ||
      !uploadDeviceFloats(filmBaseDensityDevice_, staticResources_.filmBaseDensity) ||
      !uploadDeviceFloats(paperLogSensitivityDevice_, staticResources_.paperLogSensitivity) ||
      !uploadDeviceFloats(thKg3IlluminantDevice_, staticResources_.thKg3Illuminant) ||
      !uploadDeviceFloats(customEnlargerFiltersDevice_, staticResources_.customEnlargerFilters) ||
      !uploadDeviceFloats(neutralPrintFiltersDevice_, staticResources_.neutralPrintFilters) ||
      !uploadDeviceFloats(academyPrinterDensityDataDevice_, staticResources_.academyPrinterDensityData) ||
      !uploadDeviceFloats(paperScanDensityDataDevice_, staticResources_.paperScanDensityData) ||
      !uploadDeviceFloats(scanIlluminantsAndCmfsDevice_, staticResources_.scanIlluminantsAndCmfs) ||
      !uploadDeviceFloats(scanToOutputRgbDataDevice_, staticResources_.scanToOutputRgbData) ||
      !uploadDeviceFloats(colorEncodeLutDevice_, staticResources_.colorEncodeLut)) {
    staticBuffersUploaded_ = false;
    return false;
  }
  staticBuffersUploaded_ = true;
  return true;
}

bool CudaRenderer::cudaFilmPipelineEligible(const RenderParams &params, bool &densityOutput, std::string &reason) const {
  densityOutput = params.renderOutput == RenderOutputMode::FilmDensityCmy;
  const bool logRawOutput = params.renderOutput == RenderOutputMode::FilmLogRaw;
  const bool densityWithGrainOutput = params.renderOutput == RenderOutputMode::FilmDensityCmyWithGrain;
  const bool printLogRawOutput = params.renderOutput == RenderOutputMode::PrintLogRaw;
  const bool printDensityOutput = params.renderOutput == RenderOutputMode::PrintDensityCmy;
  const bool finalOutput = params.renderOutput == RenderOutputMode::FinalPreview;
  if (!logRawOutput && !densityOutput && !densityWithGrainOutput && !printLogRawOutput && !printDensityOutput && !finalOutput) {
    reason = "The requested output mode is not implemented by the Windows CUDA backend.";
    return false;
  }
  (void)finalOutput;
  return true;
}

bool CudaRenderer::renderCudaOwned(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  const RenderParams &params,
  double time
) {
  // this is the complete Windows frame graph; keep host transfers at its edges
  const bool copyDiagnostic = envFlag("SPEKTRAFILM_CUDA_COPY_DIAGNOSTIC");
  bool densityOutput = false;
  std::string ineligibleReason;
  if (!copyDiagnostic && !cudaFilmPipelineEligible(params, densityOutput, ineligibleReason)) {
    lastError_ = ineligibleReason;
    return false;
  }

  diagnostics_ = RendererDiagnostics{};
  diagnostics_.backendName = "cuda";
  diagnostics_.deviceName = deviceName_;
  diagnostics_.cudaComputeCapabilityMajor = computeCapabilityMajor_;
  diagnostics_.cudaComputeCapabilityMinor = computeCapabilityMinor_;
  diagnostics_.cudaTransferMode = cudaZeroCopyModeFromEnv() == "mapped" ? "mapped-requested" : "device-staging";
  diagnostics_.threadgroupMode = "256";
  diagnostics_.intermediatePrecision = "float";
  diagnostics_.renderSerialized = true;
  const bool cudaPassTiming = envFlag("SPEKTRAFILM_CUDA_PASS_TIMING");
  diagnostics_.passGpuTimingEnabled = cudaPassTiming;
  diagnostics_.passGpuTimingAvailable = cudaPassTiming;
  diagnostics_.passTimingMode = cudaPassTiming ? "cuda-event-per-pass" : "cuda-single-frame-sync";

  if (!validateRgbaFloatOrHalfImages(source, destination, lastError_)) {
    return false;
  }
  const bool sourceIsCudaDevice = source.memoryDomain == ImageMemoryDomain::CudaDevice;
  const bool destinationIsCudaDevice = destination.memoryDomain == ImageMemoryDomain::CudaDevice;
  if (sourceIsCudaDevice != destinationIsCudaDevice) {
    lastError_ = "CUDA render requires source and destination images to use the same memory domain.";
    return false;
  }
  const bool hostCudaRender = sourceIsCudaDevice && destinationIsCudaDevice;
  if (hostCudaRender) {
    // Resolve gave us device pointers. validate the device before reusing persistent allocations.
    cudaPointerAttributes sourceAttributes{};
    cudaPointerAttributes destinationAttributes{};
    cudaError_t status = cudaPointerGetAttributes(&sourceAttributes, source.data);
    if (status == cudaSuccess) {
      status = cudaPointerGetAttributes(&destinationAttributes, destination.data);
    }
    if (status != cudaSuccess ||
        sourceAttributes.type != cudaMemoryTypeDevice ||
        destinationAttributes.type != cudaMemoryTypeDevice ||
        sourceAttributes.device != destinationAttributes.device) {
      lastError_ = "The host CUDA render action supplied invalid or cross-device image pointers.";
      return false;
    }
    const int hostDevice = sourceAttributes.device;
    if (hostDevice != deviceIndex_) {
      if (sourceDevice_.pointer || destinationDevice_.pointer || staticBuffersUploaded_) {
        lastError_ = "The host changed CUDA devices after persistent renderer buffers were allocated.";
        return false;
      }
      status = cudaSetDevice(hostDevice);
      if (status != cudaSuccess) {
        lastError_ = std::string("Unable to select the host CUDA image device: ") + cudaGetErrorString(status);
        return false;
      }
      deviceIndex_ = hostDevice;
      cudaDeviceProp properties{};
      if (cudaGetDeviceProperties(&properties, hostDevice) == cudaSuccess) {
        deviceName_ = properties.name;
        computeCapabilityMajor_ = properties.major;
        computeCapabilityMinor_ = properties.minor;
        diagnostics_.deviceName = deviceName_;
      }
    } else {
      status = cudaSetDevice(hostDevice);
      if (status != cudaSuccess) {
        lastError_ = std::string("Unable to activate the host CUDA image device: ") + cudaGetErrorString(status);
        return false;
      }
    }
  }
  const EffectiveRenderWindow effectiveWindow = intersectRenderWindow(source, destination, window);
  if (effectiveWindow.empty()) {
    return true;
  }

  const int32_t width = effectiveWindow.width();
  const int32_t height = effectiveWindow.height();
  const size_t floatCount = static_cast<size_t>(width) * static_cast<size_t>(height) * 4u;
  const size_t bytes = floatCount * sizeof(float);
  const size_t grainLayerBytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 9u * sizeof(float);
  const bool filmLogRawOutput = params.renderOutput == RenderOutputMode::FilmLogRaw;
  const bool filmDensityOutput = params.renderOutput == RenderOutputMode::FilmDensityCmy;
  const bool filmDensityWithGrainOutput = params.renderOutput == RenderOutputMode::FilmDensityCmyWithGrain;
  const bool printLogRawOutput = params.renderOutput == RenderOutputMode::PrintLogRaw;
  const bool printDensityOutput = params.renderOutput == RenderOutputMode::PrintDensityCmy;
  const bool finalOutput = params.renderOutput == RenderOutputMode::FinalPreview;
  const bool finalProcessNegative = finalOutput && params.process == ProcessMode::ProcessNegative;
  const bool rcmOutput = finalOutput && params.outputRole == OutputRole::Rcm;
  const bool printPipelineOutput = printLogRawOutput || printDensityOutput;
  const bool previewGrainPath = params.grainEnabled &&
    !finalProcessNegative &&
    params.grainModel == GrainModel::Preview &&
    (filmDensityWithGrainOutput || printPipelineOutput || finalOutput);
  const bool productionGrainPath = params.grainEnabled &&
    !finalProcessNegative &&
    params.grainModel == GrainModel::Production &&
    (filmDensityWithGrainOutput || printPipelineOutput || finalOutput);
  const bool grainSynthesisPath = params.grainEnabled &&
    !finalProcessNegative &&
    params.grainModel == GrainModel::GrainSynthesis &&
    (filmDensityWithGrainOutput || printPipelineOutput || finalOutput);
  int32_t grainSynthesisRequestedSamples = 0;
  int32_t grainSynthesisEffectiveSamples = 0;
  bool grainSynthesisSamplesCapped = false;

  // decide optional passes once here, dispatch code below just follows these flags
  KernelParams kernelParams{};
  bool dirPath = !copyDiagnostic &&
    !finalProcessNegative &&
    params.dirCouplersAmount > 0.0f &&
    params.renderOutput != RenderOutputMode::FilmLogRaw;
  bool dirBlurPath = dirPath && params.dirCouplersDiffusionUm > 0.0f;
  bool dirTailPath = dirBlurPath &&
    params.dirCouplersDiffusionTailUm > 0.0f &&
    params.dirCouplersDiffusionTailWeight > 0.0f;
  KernelDirInfo dirInfo{};
  KernelGaussianBlurInfo dirCoreBlurInfo{};
  std::array<KernelGaussianBlurInfo, 3> dirTailBlurInfos{};
  std::vector<float> dirCorrectedDensityCurves;
  KernelDiffusionInfo cameraDiffusionInfo{};
  KernelDiffusionInfo printDiffusionInfo{};
  std::vector<KernelDiffusionComponent> cameraDiffusionComponents;
  std::vector<KernelDiffusionComponent> printDiffusionComponents;
  bool cameraDiffusionPath = false;
  bool printDiffusionPath = false;
  bool halationBoostPath = false;
  bool halationScatterPath = false;
  bool halationBouncePath = false;
  const std::string transferPolicy = cudaTransferPolicyFromEnv();
  const bool forcePinnedTransfer = transferPolicy == "pinned";
  const bool sourceTightFloat =
    canUseContiguousFloatSourceWindow(source, effectiveWindow, width);
  const bool destinationTightFloat =
    canUseContiguousFloatDestinationWindow(destination, effectiveWindow, width);
  const bool autoPinnedTransfer = transferPolicy == "auto" && sourceTightFloat && destinationTightFloat;
  const bool usePinnedTransfer = forcePinnedTransfer || autoPinnedTransfer;
  const bool directSourceHostCopy = sourceTightFloat && !usePinnedTransfer;
  const bool directDestinationHostCopy = destinationTightFloat && !usePinnedTransfer;
  const uint32_t diffusionGroupSize = diffusionGroupSizeFromEnv();
  const std::string blurDownsampleMode = blurDownsampleModeFromEnv();
  const float *sourceUploadHost = nullptr;
  float *destinationDownloadHost = nullptr;
  diagnostics_.sourceNoCopy = hostCudaRender && sourceTightFloat;
  diagnostics_.destinationNoCopy = false;
  diagnostics_.cudaTransferMode = hostCudaRender
    ? (sourceTightFloat && destinationTightFloat
        ? "host-cuda-device-direct"
        : "host-cuda-device-gpu-pack")
    : (directSourceHostCopy && directDestinationHostCopy
        ? "direct-host-copy"
        : (directSourceHostCopy || directDestinationHostCopy ? "mixed-direct-pinned-staging" : "pinned-staging"));
  diagnostics_.diffusionGroupSize = diffusionGroupSize;
  diagnostics_.blurDownsample = blurDownsampleMode;
  diagnostics_.intermediatePrecision = "float";

  // CPU images use pinned staging. host CUDA images stay on-device and may only need a layout pack.
  {
    const auto start = std::chrono::steady_clock::now();
    if (hostCudaRender) {
      sourceUploadHost = nullptr;
    } else if (directSourceHostCopy) {
      sourceUploadHost = sourceWindowFloatPointer(source, effectiveWindow);
    } else {
      if (!ensurePinnedHostBuffer(pinnedSourceStaging_, bytes)) {
        return false;
      }
      auto *staging = static_cast<float *>(pinnedSourceStaging_.pointer);
      if (sourceTightFloat) {
        std::memcpy(staging, sourceWindowFloatPointer(source, effectiveWindow), bytes);
      } else {
        copySourceToFloatStaging(source, effectiveWindow, width, height, staging);
      }
      sourceUploadHost = staging;
      diagnostics_.cudaPinnedStagingBytes += static_cast<uint64_t>(pinnedSourceStaging_.bytes);
    }
    diagnostics_.sourceCopyMs = elapsedMs(start, std::chrono::steady_clock::now());
  }

  if (hostCudaRender) {
    destinationDownloadHost = nullptr;
  } else if (directDestinationHostCopy) {
    destinationDownloadHost = destinationWindowFloatPointer(destination, effectiveWindow);
  } else {
    if (!ensurePinnedHostBuffer(pinnedDestinationStaging_, bytes)) {
      return false;
    }
    destinationDownloadHost = static_cast<float *>(pinnedDestinationStaging_.pointer);
    diagnostics_.cudaPinnedStagingBytes += static_cast<uint64_t>(pinnedDestinationStaging_.bytes);
  }

  if (!ensureDeviceBuffer(sourceDevice_, bytes) ||
      !ensureDeviceBuffer(destinationDevice_, bytes) ||
      !ensureDeviceBuffer(scratchDeviceA_, bytes) ||
      !ensureDeviceBuffer(scratchDeviceB_, bytes)) {
    return false;
  }
  if (dirTailPath && !ensureDeviceBuffer(dirTailScratchDevice_, bytes * 3u)) {
    return false;
  }
  if ((productionGrainPath || grainSynthesisPath) &&
      (!ensureDeviceBuffer(grainLayerDeviceA_, grainLayerBytes) ||
       !ensureDeviceBuffer(grainLayerDeviceB_, grainLayerBytes) ||
       !ensureDeviceBuffer(grainMicroDeviceA_, bytes) ||
       !ensureDeviceBuffer(grainMicroDeviceB_, bytes))) {
    return false;
  }
  diagnostics_.cudaDeviceScratchBytes =
    static_cast<uint64_t>(
      sourceDevice_.bytes +
      destinationDevice_.bytes +
      scratchDeviceA_.bytes +
      scratchDeviceB_.bytes +
      (dirTailPath ? dirTailScratchDevice_.bytes : 0u) +
      (halationBoostPath ? halationBoostInfoDevice_.bytes : 0u) +
      ((productionGrainPath || grainSynthesisPath)
        ? grainLayerDeviceA_.bytes + grainLayerDeviceB_.bytes + grainMicroDeviceA_.bytes + grainMicroDeviceB_.bytes
        : 0u));
  diagnostics_.scratchAllocationBytes = diagnostics_.cudaDeviceScratchBytes;
  diagnostics_.scratchAllocationCount =
    4u + (dirTailPath ? 1u : 0u) + (halationBoostPath ? 1u : 0u) + ((productionGrainPath || grainSynthesisPath) ? 4u : 0u);

  if (!copyDiagnostic) {
    if (!ensureStaticResources(params)) {
      return false;
    }
    kernelParams = toKernelParams(params, time, width, height);
    if (staticResources_.filmCurves) {
      applyProfileHalationDefaults(kernelParams, params, *staticResources_.filmCurves);
    }
    if (params.autoExposure) {
      kernelParams.autoExposureEv = measureAutoExposureEv(sourceUploadHost, width, height, params);
    }
    if (grainSynthesisPath) {
      grainSynthesisRequestedSamples = kernelParams.grainSynthesisSamples;
      const int32_t synthesisSampleCap = cudaGrainSynthesisSampleCap(width, height);
      if (synthesisSampleCap > 0 && kernelParams.grainSynthesisSamples > synthesisSampleCap) {
        kernelParams.grainSynthesisSamples = synthesisSampleCap;
        grainSynthesisSamplesCapped = true;
      }
      grainSynthesisEffectiveSamples = kernelParams.grainSynthesisSamples;
    }
    if (!uploadDeviceStruct(paramsDevice_, kernelParams)) {
      return false;
    }
    // DIR starts after development; keep FilmLogRaw on CUDA and simply bypass the density-only stage.
    dirPath = params.dirCouplersAmount > 0.0f && params.renderOutput != RenderOutputMode::FilmLogRaw;
    dirBlurPath = dirPath && kernelParams.dirCouplersDiffusionUm > 0.0f;
    dirTailPath = dirBlurPath &&
      kernelParams.dirCouplersDiffusionTailUm > 0.0f &&
      kernelParams.dirCouplersDiffusionTailWeight > 0.0f;
    if (dirPath) {
      if (!staticResources_.filmCurves) {
        lastError_ = "CUDA DIR path requires film profile curves.";
        return false;
      }
      dirInfo = makeDirInfo(*staticResources_.filmCurves, params);
      dirCoreBlurInfo = makeGaussianBlurInfo(
        std::max(kernelParams.dirCouplersDiffusionUm, 0.0f) /
          std::max(kernelParams.filmPixelSizeUm, 1.0e-6f),
        256u);
      dirTailBlurInfos = makeDirTailBlurInfos(kernelParams);
      dirCorrectedDensityCurves = makeDirCorrectedDensityCurves(*staticResources_.filmCurves, dirInfo);
      if (!uploadDeviceStruct(dirInfoDevice_, dirInfo) ||
          !uploadDeviceStruct(dirCoreBlurInfoDevice_, dirCoreBlurInfo) ||
          !uploadDeviceBytes(
            dirTailBlurInfosDevice_,
            dirTailBlurInfos.data(),
            dirTailBlurInfos.size() * sizeof(KernelGaussianBlurInfo)) ||
          !uploadDeviceFloats(dirCorrectedDensityCurvesDevice_, dirCorrectedDensityCurves)) {
        return false;
      }
      diagnostics_.dirPath = true;
      diagnostics_.dirTailBackend = dirTailPath ? "custom" : "none";
      diagnostics_.uploadBytes +=
        sizeof(KernelDirInfo) +
        sizeof(KernelGaussianBlurInfo) +
        dirTailBlurInfos.size() * sizeof(KernelGaussianBlurInfo) +
        dirCorrectedDensityCurves.size() * sizeof(float);
    }
    cameraDiffusionComponents =
      makeCameraDiffusionComponents(params, kernelParams.filmPixelSizeUm, cameraDiffusionInfo);
    printDiffusionComponents =
      makePrintDiffusionComponents(params, kernelParams.filmPixelSizeUm, printDiffusionInfo);
    // Camera diffusion works on raw exposure, so FilmLogRaw must include it before raw-to-log conversion.
    cameraDiffusionPath =
      !finalProcessNegative &&
      params.cameraDiffusionEnabled &&
      params.cameraDiffusionStrength > 0.0f &&
      params.cameraDiffusionSpatialScale > 0.0f &&
      cameraDiffusionInfo.componentCount > 0u &&
      !cameraDiffusionComponents.empty();
    printDiffusionPath =
      params.printDiffusionEnabled &&
      params.printDiffusionStrength > 0.0f &&
      params.printDiffusionSpatialScale > 0.0f &&
      printDiffusionInfo.componentCount > 0u &&
      !printDiffusionComponents.empty() &&
      (params.renderOutput == RenderOutputMode::PrintLogRaw ||
       params.renderOutput == RenderOutputMode::PrintDensityCmy ||
       (finalOutput && (params.process == ProcessMode::PrintSimulation ||
                        params.process == ProcessMode::ProcessNegative)));
    halationBoostPath =
      !finalProcessNegative &&
      params.halationEnabled &&
      kernelParams.halationBoostEv > 0.0f &&
      !filmLogRawOutput &&
      (filmDensityOutput || printPipelineOutput || finalOutput);
    halationScatterPath =
      !finalProcessNegative &&
      params.halationEnabled &&
      kernelParams.scatterAmount > 0.0f &&
      kernelParams.scatterScale > 0.0f;
    halationBouncePath =
      !finalProcessNegative &&
      params.halationEnabled &&
      kernelParams.halationAmount > 0.0f &&
      kernelParams.halationScale > 0.0f &&
      (kernelParams.halationStrengthR > 0.0f ||
       kernelParams.halationStrengthG > 0.0f ||
       kernelParams.halationStrengthB > 0.0f);
    if (halationBoostPath && !ensureDeviceBuffer(halationBoostInfoDevice_, 4u * sizeof(float))) {
      return false;
    }
    if (halationBoostPath) {
      diagnostics_.cudaDeviceScratchBytes += static_cast<uint64_t>(halationBoostInfoDevice_.bytes);
      diagnostics_.scratchAllocationBytes = diagnostics_.cudaDeviceScratchBytes;
      diagnostics_.scratchAllocationCount += 1u;
    }
    diagnostics_.halationPath = halationBoostPath || halationScatterPath || halationBouncePath;
    if (cameraDiffusionPath) {
      if (!uploadDeviceStruct(cameraDiffusionInfoDevice_, cameraDiffusionInfo) ||
          !uploadDeviceBytes(
            cameraDiffusionComponentsDevice_,
            cameraDiffusionComponents.data(),
            cameraDiffusionComponents.size() * sizeof(KernelDiffusionComponent))) {
        return false;
      }
      diagnostics_.cameraDiffusionPath = true;
      diagnostics_.uploadBytes +=
        sizeof(KernelDiffusionInfo) + cameraDiffusionComponents.size() * sizeof(KernelDiffusionComponent);
    }
    if (printDiffusionPath) {
      if (!uploadDeviceStruct(printDiffusionInfoDevice_, printDiffusionInfo) ||
          !uploadDeviceBytes(
            printDiffusionComponentsDevice_,
            printDiffusionComponents.data(),
            printDiffusionComponents.size() * sizeof(KernelDiffusionComponent))) {
        return false;
      }
      diagnostics_.printDiffusionPath = true;
      diagnostics_.uploadBytes +=
        sizeof(KernelDiffusionInfo) + printDiffusionComponents.size() * sizeof(KernelDiffusionComponent);
    }
    if (cameraDiffusionPath || printDiffusionPath) {
      const uint32_t maxGroupSize = std::min<uint32_t>(std::max<uint32_t>(diffusionGroupSize, 1u), 4u);
      const bool diffusionDownsamplePath =
        anyDiffusionComponentDownsamples(cameraDiffusionComponents, blurDownsampleMode) ||
        anyDiffusionComponentDownsamples(printDiffusionComponents, blurDownsampleMode);
      if (maxGroupSize > 1u && !ensureDeviceBuffer(diffusionGroupTempDevice_, bytes * maxGroupSize)) {
        return false;
      }
      if (diffusionDownsamplePath) {
        const int reducedWidth = (width + 1) / 2;
        const int reducedHeight = (height + 1) / 2;
        const size_t reducedBytes =
          static_cast<size_t>(reducedWidth) * static_cast<size_t>(reducedHeight) * 4u * sizeof(float);
        if (!ensureDeviceBuffer(diffusionReducedSourceDevice_, reducedBytes) ||
            !ensureDeviceBuffer(diffusionReducedTempDevice_, reducedBytes * maxGroupSize) ||
            !ensureDeviceBuffer(diffusionReducedBlurDevice_, reducedBytes * maxGroupSize)) {
          return false;
        }
      }
      diagnostics_.cudaDeviceScratchBytes += maxGroupSize > 1u
        ? static_cast<uint64_t>(diffusionGroupTempDevice_.bytes)
        : 0u;
      diagnostics_.cudaDeviceScratchBytes += diffusionDownsamplePath
        ? static_cast<uint64_t>(
            diffusionReducedSourceDevice_.bytes +
            diffusionReducedTempDevice_.bytes +
            diffusionReducedBlurDevice_.bytes)
        : 0u;
      diagnostics_.scratchAllocationBytes = diagnostics_.cudaDeviceScratchBytes;
      diagnostics_.scratchAllocationCount += maxGroupSize > 1u ? 1u : 0u;
      diagnostics_.scratchAllocationCount += diffusionDownsamplePath ? 3u : 0u;
    }
  }

  // Device frame begins here; host transfers stay outside the processing graph.
  const auto gpuWorkStart = std::chrono::steady_clock::now();
  float kernelMs = 0.0f;
  std::array<char, 512> error{};
  const float *sourceFrameDevice = nullptr;
  {
    const auto start = std::chrono::steady_clock::now();
    if (hostCudaRender) {
      diagnostics_.cudaHostToDeviceMs = 0.0;
      diagnostics_.uploadBytes = 0u;
      if (sourceTightFloat) {
        sourceFrameDevice = sourceWindowFloatPointer(source, effectiveWindow);
      } else {
        float passMs = 0.0f;
        if (!spektraCudaPackDeviceImageToFloat(
              source.data,
              source.x1,
              source.y1,
              source.rowBytes,
              source.bytesPerComponent,
              effectiveWindow.x1,
              effectiveWindow.y1,
              width,
              height,
              static_cast<float *>(sourceDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        sourceFrameDevice = static_cast<const float *>(sourceDevice_.pointer);
        kernelMs += passMs;
        diagnostics_.passes.push_back({
          "cuda_host_image_pack",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
      }
    } else {
      cudaError_t status = cudaMemcpyAsync(sourceDevice_.pointer, sourceUploadHost, bytes, cudaMemcpyHostToDevice, 0);
      if (status == cudaSuccess) {
        status = cudaStreamSynchronize(0);
      }
      diagnostics_.cudaHostToDeviceMs = elapsedMs(start, std::chrono::steady_clock::now());
      diagnostics_.uploadBytes = static_cast<uint64_t>(bytes);
      if (status != cudaSuccess) {
        std::ostringstream out;
        out << "cudaMemcpy H2D failed: " << cudaGetErrorString(status);
        lastError_ = out.str();
        return false;
      }
      sourceFrameDevice = static_cast<const float *>(sourceDevice_.pointer);
    }
  }

  if (hostCudaRender && params.autoExposure) {
    const auto [previewWidth, previewHeight] = autoExposurePreviewDimensions(width, height);
    const size_t previewCount = static_cast<size_t>(previewWidth) * static_cast<size_t>(previewHeight);
    const size_t previewBytes = previewCount * sizeof(float);
    if (!ensureDeviceBuffer(autoExposurePreviewDevice_, previewBytes) ||
        !ensurePinnedHostBuffer(pinnedAutoExposurePreview_, previewBytes)) {
      return false;
    }
    const int32_t colorSpace = std::clamp(
      static_cast<int32_t>(params.inputColorSpace),
      0,
      static_cast<int32_t>(kSpektraColorSpaceCount - 1u));
    const float *meterMatrices = inputMeterXyzMatrices();
    const float *meter = meterMatrices ? meterMatrices + static_cast<size_t>(colorSpace) * 9u : nullptr;
    const float meterR = meter ? meter[3] : 0.2126f;
    const float meterG = meter ? meter[4] : 0.7152f;
    const float meterB = meter ? meter[5] : 0.0722f;
    float passMs = 0.0f;
    if (!spektraCudaAutoExposurePreview(
          sourceFrameDevice,
          static_cast<float *>(autoExposurePreviewDevice_.pointer),
          width,
          height,
          previewWidth,
          previewHeight,
          static_cast<const KernelParams *>(paramsDevice_.pointer),
          static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
          static_cast<const float *>(colorDecodeLutDevice_.pointer),
          static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
          meterR,
          meterG,
          meterB,
          &passMs,
          error.data(),
          error.size())) {
      lastError_ = error.data();
      return false;
    }
    cudaError_t status = cudaMemcpy(
      pinnedAutoExposurePreview_.pointer,
      autoExposurePreviewDevice_.pointer,
      previewBytes,
      cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      lastError_ = std::string("CUDA auto-exposure preview readback failed: ") + cudaGetErrorString(status);
      return false;
    }
    const auto *preview = static_cast<const float *>(pinnedAutoExposurePreview_.pointer);
    kernelParams.autoExposureEv = measureAutoExposureEvFromLuminance(
      std::vector<float>(preview, preview + previewCount),
      previewWidth,
      previewHeight,
      params);
    if (!uploadDeviceStruct(paramsDevice_, kernelParams)) {
      return false;
    }
    kernelMs += passMs;
    diagnostics_.passes.push_back({
      "cuda_auto_exposure_preview",
      static_cast<double>(passMs),
      static_cast<uint32_t>(previewWidth),
      static_cast<uint32_t>(previewHeight),
      1u,
      256u,
      1u,
      static_cast<uint64_t>(previewBytes) * 2u,
      true
    });
    diagnostics_.cudaPinnedStagingBytes += static_cast<uint64_t>(pinnedAutoExposurePreview_.bytes);
    diagnostics_.cudaTransferMode += "+auto-exposure-preview-readback";
  }

  if (copyDiagnostic) {
    if (!spektraCudaCopyFrame(
          sourceFrameDevice,
          static_cast<float *>(destinationDevice_.pointer),
          width,
          height,
          &kernelMs,
          error.data(),
          error.size())) {
      lastError_ = error.data();
      return false;
    }
    diagnostics_.passes.push_back({
      "cuda_float_copy",
      static_cast<double>(kernelMs),
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height),
      1u,
      256u,
      1u,
      static_cast<uint64_t>(bytes) * 2u,
      true
    });
  } else {
    float passMs = 0.0f;
    // Shared dispatch helpers keep spatial pass ordering in one place.
    auto recordCudaPass = [&](const char *name, float milliseconds, uint32_t depth, uint64_t trafficBytes) {
      diagnostics_.passes.push_back({
        name,
        static_cast<double>(milliseconds),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        depth,
        256u,
        1u,
        trafficBytes,
        true
      });
    };
    // camera and print diffusion share this runner; wide lobes may use the reduced-res path
    auto dispatchDiffusion = [&](
      const char *labelPrefix,
      const float *sourceFrame,
      float *tempFrame,
      float *accumFrame,
      float *resolvedFrame,
      const KernelDiffusionInfo *infoDevice,
      const KernelDiffusionComponent *componentsDevice,
      const std::vector<KernelDiffusionComponent> &components
    ) -> bool {
      passMs = 0.0f;
      if (!spektraCudaClearFrame(accumFrame, width, height, &passMs, error.data(), error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass((std::string(labelPrefix) + "_diffusion_clear").c_str(), passMs, 1u, static_cast<uint64_t>(bytes));
      kernelMs += passMs;
      const uint32_t maxGroupSize = std::min<uint32_t>(std::max<uint32_t>(diffusionGroupSize, 1u), 4u);
      const uint32_t componentCount = static_cast<uint32_t>(components.size());
      for (uint32_t component = 0u; component < componentCount;) {
        const uint32_t downsampleScale =
          diffusionDownsampleScaleForSigma(blurDownsampleMode, components[component].sigmaPx);
        uint32_t groupCount = 1u;
        while (component + groupCount < componentCount &&
               groupCount < maxGroupSize &&
               diffusionDownsampleScaleForSigma(blurDownsampleMode, components[component + groupCount].sigmaPx) == downsampleScale) {
          ++groupCount;
        }
        if (downsampleScale <= 1u && groupCount <= 1u) {
          passMs = 0.0f;
          if (!spektraCudaDiffusionBlurX(
                sourceFrame,
                tempFrame,
                width,
                height,
                componentsDevice,
                component,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_blur_x").c_str(), passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaDiffusionBlurYAccumulate(
                tempFrame,
                accumFrame,
                width,
                height,
                componentsDevice,
                component,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_blur_y_accumulate").c_str(), passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;
        } else if (downsampleScale <= 1u) {
          auto *groupTemp = static_cast<float *>(diffusionGroupTempDevice_.pointer);
          passMs = 0.0f;
          if (!spektraCudaDiffusionGroupBlurX(
                sourceFrame,
                groupTemp,
                width,
                height,
                componentsDevice,
                component,
                groupCount,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_group_blur_x").c_str(), passMs, groupCount, static_cast<uint64_t>(bytes) * (groupCount + 1u));
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaDiffusionGroupBlurYAccumulate(
                groupTemp,
                accumFrame,
                width,
                height,
                componentsDevice,
                component,
                groupCount,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_group_blur_y_accumulate").c_str(), passMs, groupCount, static_cast<uint64_t>(bytes) * (groupCount + 1u));
          kernelMs += passMs;
        } else {
          const int reducedWidth = (width + static_cast<int>(downsampleScale) - 1) / static_cast<int>(downsampleScale);
          const int reducedHeight = (height + static_cast<int>(downsampleScale) - 1) / static_cast<int>(downsampleScale);
          const size_t reducedBytes =
            static_cast<size_t>(reducedWidth) * static_cast<size_t>(reducedHeight) * 4u * sizeof(float);
          auto *reducedSource = static_cast<float *>(diffusionReducedSourceDevice_.pointer);
          auto *reducedTemp = static_cast<float *>(diffusionReducedTempDevice_.pointer);
          auto *reducedBlur = static_cast<float *>(diffusionReducedBlurDevice_.pointer);

          passMs = 0.0f;
          if (!spektraCudaDiffusionDownsample(
                sourceFrame,
                reducedSource,
                width,
                height,
                downsampleScale,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_downsample").c_str(), passMs, 1u, static_cast<uint64_t>(bytes + reducedBytes));
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaDiffusionReducedGroupBlurX(
                reducedSource,
                reducedTemp,
                reducedWidth,
                reducedHeight,
                downsampleScale,
                componentsDevice,
                component,
                groupCount,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_downsample_group_blur_x").c_str(), passMs, groupCount, static_cast<uint64_t>(reducedBytes) * (groupCount + 1u));
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaDiffusionReducedGroupBlurY(
                reducedTemp,
                reducedBlur,
                reducedWidth,
                reducedHeight,
                downsampleScale,
                componentsDevice,
                component,
                groupCount,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_downsample_group_blur_y").c_str(), passMs, groupCount, static_cast<uint64_t>(reducedBytes) * groupCount * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaDiffusionReducedGroupUpsampleAccumulate(
                reducedBlur,
                accumFrame,
                width,
                height,
                downsampleScale,
                componentsDevice,
                component,
                groupCount,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass((std::string(labelPrefix) + "_diffusion_downsample_group_upsample_accumulate").c_str(), passMs, groupCount, static_cast<uint64_t>(bytes) + static_cast<uint64_t>(reducedBytes) * groupCount);
          kernelMs += passMs;
        }
        component += groupCount;
      }
      passMs = 0.0f;
      if (!spektraCudaDiffusionResolve(
            sourceFrame,
            accumFrame,
            resolvedFrame,
            width,
            height,
            infoDevice,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass((std::string(labelPrefix) + "_diffusion_resolve").c_str(), passMs, 1u, static_cast<uint64_t>(bytes) * 3u);
      kernelMs += passMs;
      return true;
    };
    auto dispatchScannerPost = [&](bool printGlarePath) -> bool {
      const KernelParams &kp = kernelParams;
      if (printGlarePath) {
        passMs = 0.0f;
        if (!spektraCudaPrintGlareGenerate(
              static_cast<float *>(scratchDeviceA_.pointer),
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_print_glare_generate", passMs, 1u, static_cast<uint64_t>(bytes));
        kernelMs += passMs;

        if (kp.glareBlur > 1.0e-4f) {
          passMs = 0.0f;
          if (!spektraCudaGaussianBlurX(
                static_cast<const float *>(scratchDeviceA_.pointer),
                static_cast<float *>(scratchDeviceB_.pointer),
                width,
                height,
                kp.glareBlur,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_print_glare_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaGaussianBlurY(
                static_cast<const float *>(scratchDeviceB_.pointer),
                static_cast<float *>(scratchDeviceA_.pointer),
                width,
                height,
                kp.glareBlur,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_print_glare_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;
        }

        passMs = 0.0f;
        if (!spektraCudaPrintGlareApply(
              static_cast<const float *>(destinationDevice_.pointer),
              static_cast<const float *>(scratchDeviceA_.pointer),
              static_cast<float *>(destinationDevice_.pointer),
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
              static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
              static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
              static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_print_glare_apply", passMs, 1u, static_cast<uint64_t>(bytes) * 3u);
        kernelMs += passMs;
      }

      if (kp.scannerBlurSigmaPx > 1.0e-4f) {
        passMs = 0.0f;
        if (!spektraCudaGaussianBlurX(
              static_cast<const float *>(destinationDevice_.pointer),
              static_cast<float *>(scratchDeviceA_.pointer),
              width,
              height,
              kp.scannerBlurSigmaPx,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_scanner_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;

        passMs = 0.0f;
        if (!spektraCudaGaussianBlurY(
              static_cast<const float *>(scratchDeviceA_.pointer),
              static_cast<float *>(destinationDevice_.pointer),
              width,
              height,
              kp.scannerBlurSigmaPx,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_scanner_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;
      }

      const bool needsUnsharp = kp.scannerUnsharpSigmaPx > 1.0e-4f && kp.scannerUnsharpAmount > 0.0f;
      const float *unsharpDevice = nullptr;
      if (needsUnsharp) {
        passMs = 0.0f;
        if (!spektraCudaGaussianBlurX(
              static_cast<const float *>(destinationDevice_.pointer),
              static_cast<float *>(scratchDeviceA_.pointer),
              width,
              height,
              kp.scannerUnsharpSigmaPx,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_unsharp_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;

        passMs = 0.0f;
        if (!spektraCudaGaussianBlurY(
              static_cast<const float *>(scratchDeviceA_.pointer),
              static_cast<float *>(scratchDeviceB_.pointer),
              width,
              height,
              kp.scannerUnsharpSigmaPx,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_unsharp_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;
        unsharpDevice = static_cast<const float *>(scratchDeviceB_.pointer);
      }

      passMs = 0.0f;
      if (!spektraCudaScannerFinalize(
            static_cast<const float *>(destinationDevice_.pointer),
            unsharpDevice,
            static_cast<float *>(destinationDevice_.pointer),
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
            static_cast<const float *>(colorEncodeLutDevice_.pointer),
            static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_scanner_finalize", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;
      diagnostics_.finalPostProcessPath = true;
      return true;
    };
    // Process-negative skips film development and exposes the print stock directly.
    if (finalProcessNegative) {
      float *printRawDevice = static_cast<float *>(scratchDeviceA_.pointer);
      passMs = 0.0f;
      if (!spektraCudaPrintRawFromNegativeLight(
            sourceFrameDevice,
            printRawDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
            static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
            static_cast<const float *>(colorDecodeLutDevice_.pointer),
            static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
            static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
            static_cast<const float *>(paperHanatosResponseDevice_.pointer),
            static_cast<const float *>(preflashPaperHanatosResponseDevice_.pointer),
            static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_print_raw_from_negative_light", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;

      if (printDiffusionPath) {
        if (!dispatchDiffusion(
              "cuda_print",
              printRawDevice,
              static_cast<float *>(destinationDevice_.pointer),
              static_cast<float *>(sourceDevice_.pointer),
              printRawDevice,
              static_cast<const KernelDiffusionInfo *>(printDiffusionInfoDevice_.pointer),
              static_cast<const KernelDiffusionComponent *>(printDiffusionComponentsDevice_.pointer),
              printDiffusionComponents)) {
          return false;
        }
      }

      const bool scannerPostPath = params.scannerEnabled && !rcmOutput;
      passMs = 0.0f;
      if (!spektraCudaFinalFromPrintRaw(
            printRawDevice,
            static_cast<float *>(destinationDevice_.pointer),
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
            static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
            static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
            static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
            static_cast<const float *>(logExposureDevice_.pointer),
            static_cast<const float *>(densityCurvesDevice_.pointer),
            static_cast<const float *>(paperLogExposureDevice_.pointer),
            static_cast<const float *>(paperDensityCurvesDevice_.pointer),
            static_cast<const float *>(filmChannelDensityDevice_.pointer),
            static_cast<const float *>(filmBaseDensityDevice_.pointer),
            static_cast<const float *>(paperLogSensitivityDevice_.pointer),
            static_cast<const float *>(thKg3IlluminantDevice_.pointer),
            static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
            static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
            static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
            static_cast<const float *>(paperScanDensityDataDevice_.pointer),
            static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
            static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
            static_cast<const float *>(colorEncodeLutDevice_.pointer),
            static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
            static_cast<const float *>(hanatosRawResponseDevice_.pointer),
            static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
            static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
            !scannerPostPath,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_final_from_process_negative", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;
      if (scannerPostPath && !dispatchScannerPost(kernelParams.glarePercent > 0.0f)) {
        return false;
      }
    } else {
    // Main film path: exposure and spatial effects before development.
    const bool useEnlarger = !finalProcessNegative && enlargerTransformActive(params);
    const float *rawSourceDevice = sourceFrameDevice;
    float *rawDevice = static_cast<float *>(scratchDeviceA_.pointer);
    if (useEnlarger) {
      if (!spektraCudaEnlargerResample(
            sourceFrameDevice,
            static_cast<float *>(scratchDeviceA_.pointer),
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      kernelMs += passMs;
      diagnostics_.passes.push_back({
        "cuda_enlarger_resample",
        static_cast<double>(passMs),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1u,
        256u,
        1u,
        static_cast<uint64_t>(bytes) * 2u,
        true
      });
      rawSourceDevice = static_cast<const float *>(scratchDeviceA_.pointer);
      rawDevice = static_cast<float *>(scratchDeviceB_.pointer);
      passMs = 0.0f;
    }
    if (!spektraCudaRawExposure(
          rawSourceDevice,
          rawDevice,
          width,
          height,
          static_cast<const KernelParams *>(paramsDevice_.pointer),
          static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
          static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
          static_cast<const float *>(hanatosRawResponseDevice_.pointer),
          static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
          static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
          static_cast<const float *>(inputToSrgbDevice_.pointer),
          static_cast<const float *>(colorDecodeLutDevice_.pointer),
          static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
          &passMs,
          error.data(),
          error.size())) {
      lastError_ = error.data();
      return false;
    }
    kernelMs += passMs;
    diagnostics_.passes.push_back({
      "cuda_halation_raw_exposure",
      static_cast<double>(passMs),
      static_cast<uint32_t>(width),
      static_cast<uint32_t>(height),
      1u,
      256u,
      1u,
      static_cast<uint64_t>(bytes) * 2u,
      true
    });

    passMs = 0.0f;
    if (halationBoostPath) {
      if (!spektraCudaHalationBoostInfo(
            rawDevice,
            static_cast<float *>(halationBoostInfoDevice_.pointer),
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_boost_max", passMs, 1u, static_cast<uint64_t>(bytes));
      kernelMs += passMs;

      passMs = 0.0f;
      float *boostedRawDevice = rawDevice == static_cast<float *>(scratchDeviceA_.pointer)
        ? static_cast<float *>(scratchDeviceB_.pointer)
        : static_cast<float *>(scratchDeviceA_.pointer);
      if (!spektraCudaHalationBoostApply(
            rawDevice,
            static_cast<const float *>(halationBoostInfoDevice_.pointer),
            boostedRawDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_boost_apply", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;
      rawDevice = boostedRawDevice;
      passMs = 0.0f;
    }

    if (cameraDiffusionPath) {
      float *cameraDiffusedRawDevice = rawDevice == static_cast<float *>(scratchDeviceA_.pointer)
        ? static_cast<float *>(scratchDeviceB_.pointer)
        : static_cast<float *>(scratchDeviceA_.pointer);
      if (!dispatchDiffusion(
            "cuda_camera",
            rawDevice,
            static_cast<float *>(destinationDevice_.pointer),
            static_cast<float *>(sourceDevice_.pointer),
            cameraDiffusedRawDevice,
            static_cast<const KernelDiffusionInfo *>(cameraDiffusionInfoDevice_.pointer),
            static_cast<const KernelDiffusionComponent *>(cameraDiffusionComponentsDevice_.pointer),
            cameraDiffusionComponents)) {
        return false;
      }
      rawDevice = cameraDiffusedRawDevice;
      passMs = 0.0f;
    }

    if (halationScatterPath) {
      float *coreDevice = rawDevice == static_cast<float *>(scratchDeviceA_.pointer)
        ? static_cast<float *>(scratchDeviceB_.pointer)
        : static_cast<float *>(scratchDeviceA_.pointer);
      float *tempDevice = static_cast<float *>(destinationDevice_.pointer);
      float *tailDevice = static_cast<float *>(sourceDevice_.pointer);

      if (!spektraCudaHalationChannelBlurX(
            rawDevice,
            tempDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            0u,
            0u,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_scatter_core_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;

      passMs = 0.0f;
      if (!spektraCudaHalationChannelBlurY(
            tempDevice,
            coreDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            0u,
            0u,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_scatter_core_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
      kernelMs += passMs;

      passMs = 0.0f;
      if (!spektraCudaClearFrame(tailDevice, width, height, &passMs, error.data(), error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_clear", passMs, 1u, static_cast<uint64_t>(bytes));
      kernelMs += passMs;

      for (uint32_t component = 0u; component < 3u; ++component) {
        passMs = 0.0f;
        if (!spektraCudaHalationChannelBlurX(
              rawDevice,
              tempDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              1u,
              component,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_halation_scatter_tail_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;

        passMs = 0.0f;
        if (!spektraCudaHalationScatterTailBlurYAccumulate(
              tempDevice,
              tailDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              component,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_halation_scatter_tail_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;
      }

      passMs = 0.0f;
      if (!spektraCudaHalationScatterResolve(
            rawDevice,
            coreDevice,
            tailDevice,
            coreDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_scatter_resolve", passMs, 1u, static_cast<uint64_t>(bytes) * 4u);
      kernelMs += passMs;
      rawDevice = coreDevice;
      passMs = 0.0f;
    }

    if (halationBouncePath) {
      float *accumDevice = static_cast<float *>(sourceDevice_.pointer);
      float *tempDevice = static_cast<float *>(destinationDevice_.pointer);
      float *resolvedDevice = rawDevice == static_cast<float *>(scratchDeviceA_.pointer)
        ? static_cast<float *>(scratchDeviceB_.pointer)
        : static_cast<float *>(scratchDeviceA_.pointer);

      if (!spektraCudaClearFrame(accumDevice, width, height, &passMs, error.data(), error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_clear", passMs, 1u, static_cast<uint64_t>(bytes));
      kernelMs += passMs;

      for (uint32_t bounce = 0u; bounce < 3u; ++bounce) {
        passMs = 0.0f;
        if (!spektraCudaHalationChannelBlurX(
              rawDevice,
              tempDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              2u,
              bounce,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_halation_bounce_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;

        passMs = 0.0f;
        if (!spektraCudaHalationBounceBlurYAccumulate(
              tempDevice,
              accumDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              bounce,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_halation_bounce_blur_y_accumulate", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
        kernelMs += passMs;
      }

      passMs = 0.0f;
      if (!spektraCudaHalationResolveRaw(
            rawDevice,
            accumDevice,
            resolvedDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      recordCudaPass("cuda_halation_resolve_raw", passMs, 1u, static_cast<uint64_t>(bytes) * 3u);
      kernelMs += passMs;
      rawDevice = resolvedDevice;
      passMs = 0.0f;
    }

    if (filmLogRawOutput) {
      if (!spektraCudaRawToLogRaw(
            rawDevice,
            static_cast<float *>(destinationDevice_.pointer),
            width,
            height,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      diagnostics_.passes.push_back({
        "cuda_raw_to_log_raw",
        static_cast<double>(passMs),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1u,
        256u,
        1u,
        static_cast<uint64_t>(bytes) * 2u,
        true
      });
      kernelMs += passMs;
    } else {
      float *dirLogRawDevice = nullptr;
      if (dirPath) {
        // Preserve the raw baseline before development; spatial paths may reuse rawDevice for density.
        dirLogRawDevice = static_cast<float *>(sourceDevice_.pointer);
        if (!spektraCudaRawToLogRaw(
              rawDevice,
              dirLogRawDevice,
              width,
              height,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          "cuda_raw_to_log_raw",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
        kernelMs += passMs;
        passMs = 0.0f;
      }

      float *filmDensityDevice = (filmDensityOutput || (filmDensityWithGrainOutput && !previewGrainPath))
        ? static_cast<float *>(destinationDevice_.pointer)
        : (useEnlarger ? static_cast<float *>(scratchDeviceA_.pointer) : static_cast<float *>(scratchDeviceB_.pointer));
      if (!spektraCudaDevelopFromRaw(
            rawDevice,
            filmDensityDevice,
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
            static_cast<const float *>(logExposureDevice_.pointer),
            static_cast<const float *>(densityCurvesDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      diagnostics_.passes.push_back({
        "cuda_develop_from_raw",
        static_cast<double>(passMs),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1u,
        256u,
        1u,
        static_cast<uint64_t>(bytes) * 2u,
        true
      });
      kernelMs += passMs;

      // DIR corrects developed density, then rejoins the normal grain/print pipeline
      if (dirPath) {
        passMs = 0.0f;
        float *dirCorrectionOriginalDevice = rawDevice;
        if (!spektraCudaDirCorrectionFromDensity(
              filmDensityDevice,
              dirCorrectionOriginalDevice,
              width,
              height,
              static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
              static_cast<const KernelDirInfo *>(dirInfoDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          "cuda_dir_correction_from_density",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
        kernelMs += passMs;

        float *dirFinalCorrectionDevice = dirCorrectionOriginalDevice;
        if (dirBlurPath) {
          passMs = 0.0f;
          if (!spektraCudaDirBlurX(
                dirCorrectionOriginalDevice,
                static_cast<float *>(destinationDevice_.pointer),
                width,
                height,
                static_cast<const KernelGaussianBlurInfo *>(dirCoreBlurInfoDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_dir_blur_x",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;

          passMs = 0.0f;
          dirFinalCorrectionDevice = filmDensityDevice != static_cast<float *>(destinationDevice_.pointer)
            ? filmDensityDevice
            : (rawDevice == static_cast<float *>(scratchDeviceA_.pointer)
                ? static_cast<float *>(scratchDeviceB_.pointer)
                : static_cast<float *>(scratchDeviceA_.pointer));
          if (!spektraCudaDirBlurY(
                static_cast<const float *>(destinationDevice_.pointer),
                dirFinalCorrectionDevice,
                width,
                height,
                static_cast<const KernelGaussianBlurInfo *>(dirCoreBlurInfoDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_dir_blur_y",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;

          if (dirTailPath) {
            passMs = 0.0f;
            if (!spektraCudaDirTailBlurX(
                  dirCorrectionOriginalDevice,
                  static_cast<float *>(dirTailScratchDevice_.pointer),
                  width,
                  height,
                  static_cast<const KernelGaussianBlurInfo *>(dirTailBlurInfosDevice_.pointer),
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_dir_tail_blur_x",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              3u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 4u,
              true
            });
            kernelMs += passMs;

            passMs = 0.0f;
            if (!spektraCudaDirTailBlurYAccumulate(
                  static_cast<const float *>(dirTailScratchDevice_.pointer),
                  dirFinalCorrectionDevice,
                  width,
                  height,
                  static_cast<const KernelParams *>(paramsDevice_.pointer),
                  static_cast<const KernelGaussianBlurInfo *>(dirTailBlurInfosDevice_.pointer),
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_dir_tail_blur_y_accumulate",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              3u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 4u,
              true
            });
            kernelMs += passMs;
          }
        }

        passMs = 0.0f;
        float *redevelopedDensityDevice = static_cast<float *>(destinationDevice_.pointer);
        if (!spektraCudaDirRedevelop(
              dirLogRawDevice,
              dirFinalCorrectionDevice,
              redevelopedDensityDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
              static_cast<const float *>(logExposureDevice_.pointer),
              static_cast<const float *>(dirCorrectedDensityCurvesDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          "cuda_dir_redevelop",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 3u,
          true
        });
        kernelMs += passMs;
        filmDensityDevice = redevelopedDensityDevice;
      }

      // Grain runs in density space before print/final conversion.
      if (productionGrainPath || grainSynthesisPath) {
        diagnostics_.productionGrainPath = productionGrainPath;
        diagnostics_.grainSynthesisPath = grainSynthesisPath;
        const bool grainControlsPath =
          std::abs(kernelParams.grainAmount - 1.0f) > 1.0e-6f ||
          std::abs(kernelParams.grainSaturation - 1.0f) > 1.0e-6f;
        const bool grainLayerBlurPath =
          kernelParams.grainSublayersEnabled != 0u &&
          kernelParams.grainBlurDyeCloudsUm > 0.0f;
        const float grainMicroBlurSigmaPx =
          std::max(kernelParams.grainMicroStructureScale, 0.0f) /
          std::max(kernelParams.filmPixelSizeUm, 1.0e-6f);
        const bool grainMicroBlurPath =
          kernelParams.grainSublayersEnabled != 0u &&
          grainMicroBlurSigmaPx > 0.4f;
        const float grainFinalBlurUm =
          std::max(params.grainFinalBlurUm, 0.0f) *
          std::pow(std::max(filmFormatMm(params.filmFormat) / 35.0f, 1.0e-6f), 0.62f);
        const float grainFinalBlurSigmaPx =
          grainFinalBlurUm / std::max(kernelParams.filmPixelSizeUm, 1.0e-6f);
        const bool grainFinalBlurPath =
          grainFinalBlurSigmaPx > 0.0f &&
          (kernelParams.grainSublayersEnabled != 0u || grainFinalBlurSigmaPx > 0.4f);

        float *layerDevice = static_cast<float *>(grainLayerDeviceA_.pointer);
        float *layerTempDevice = static_cast<float *>(grainLayerDeviceB_.pointer);
        passMs = 0.0f;
        if (productionGrainPath) {
          if (!spektraCudaProductionGrainLayersFromDensity(
                filmDensityDevice,
                layerDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_production_grain_layers_from_density", passMs, 9u, static_cast<uint64_t>(bytes) + static_cast<uint64_t>(grainLayerBytes));
        } else {
          if (!spektraCudaGrainSynthesisLayersFromDensity(
                filmDensityDevice,
                layerDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_synthesis_layers_from_density", passMs, 9u, static_cast<uint64_t>(bytes) + static_cast<uint64_t>(grainLayerBytes));
        }
        kernelMs += passMs;

        if (grainLayerBlurPath) {
          passMs = 0.0f;
          if (!spektraCudaGrainLayerBlurX(
                layerDevice,
                layerTempDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_layer_blur_x", passMs, 9u, static_cast<uint64_t>(grainLayerBytes) * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaGrainLayerBlurY(
                layerTempDevice,
                layerDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_layer_blur_y", passMs, 9u, static_cast<uint64_t>(grainLayerBytes) * 2u);
          kernelMs += passMs;
        }

        float *microDevice = static_cast<float *>(grainMicroDeviceA_.pointer);
        float *microTempDevice = static_cast<float *>(grainMicroDeviceB_.pointer);
        passMs = 0.0f;
        if (!spektraCudaGrainMicrostructureSource(
              microDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        recordCudaPass("cuda_grain_microstructure_source", passMs, 1u, static_cast<uint64_t>(bytes));
        kernelMs += passMs;

        if (grainMicroBlurPath) {
          passMs = 0.0f;
          if (!spektraCudaGrainMicroBlurX(
                microDevice,
                microTempDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_micro_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaGrainMicroBlurY(
                microTempDevice,
                microDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_micro_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;
        }

        float *grainedDensityDevice = (grainControlsPath || grainFinalBlurPath)
          ? static_cast<float *>(sourceDevice_.pointer)
          : (filmDensityWithGrainOutput
              ? static_cast<float *>(destinationDevice_.pointer)
              : static_cast<float *>(sourceDevice_.pointer));
        passMs = 0.0f;
        if (productionGrainPath) {
          if (!spektraCudaGrainResolveDensity(
                layerDevice,
                microDevice,
                filmDensityDevice,
                grainedDensityDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_resolve_density", passMs, 1u, static_cast<uint64_t>(grainLayerBytes) + static_cast<uint64_t>(bytes) * 3u);
        } else {
          if (!spektraCudaGrainSynthesisResolveDensity(
                layerDevice,
                microDevice,
                filmDensityDevice,
                grainedDensityDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_synthesis_resolve_density", passMs, 1u, static_cast<uint64_t>(grainLayerBytes) + static_cast<uint64_t>(bytes) * 3u);
        }
        kernelMs += passMs;

        float *grainPipelineDensityDevice = grainedDensityDevice;
        if (grainControlsPath) {
          float *controlledDensityDevice = grainFinalBlurPath
            ? static_cast<float *>(grainMicroDeviceA_.pointer)
            : (filmDensityWithGrainOutput
                ? static_cast<float *>(destinationDevice_.pointer)
                : (filmDensityDevice == static_cast<float *>(scratchDeviceA_.pointer)
                    ? static_cast<float *>(scratchDeviceB_.pointer)
                    : static_cast<float *>(scratchDeviceA_.pointer)));
          passMs = 0.0f;
          if (!spektraCudaGrainApplyControls(
                filmDensityDevice,
                grainPipelineDensityDevice,
                controlledDensityDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_apply_controls", passMs, 1u, static_cast<uint64_t>(bytes) * 3u);
          kernelMs += passMs;
          grainPipelineDensityDevice = controlledDensityDevice;
        }

        if (grainFinalBlurPath) {
          float *blurTempDevice = static_cast<float *>(grainMicroDeviceB_.pointer);
          float *blurredDensityDevice = filmDensityWithGrainOutput
            ? static_cast<float *>(destinationDevice_.pointer)
            : (grainPipelineDensityDevice == static_cast<float *>(scratchDeviceA_.pointer)
                ? static_cast<float *>(scratchDeviceB_.pointer)
                : static_cast<float *>(scratchDeviceA_.pointer));
          passMs = 0.0f;
          if (!spektraCudaGrainDensityBlurX(
                grainPipelineDensityDevice,
                blurTempDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_density_blur_x", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;

          passMs = 0.0f;
          if (!spektraCudaGrainDensityBlurY(
                blurTempDevice,
                blurredDensityDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          recordCudaPass("cuda_grain_density_blur_y", passMs, 1u, static_cast<uint64_t>(bytes) * 2u);
          kernelMs += passMs;
          grainPipelineDensityDevice = blurredDensityDevice;
        }
        filmDensityDevice = grainPipelineDensityDevice;
        passMs = 0.0f;
      }

      if (previewGrainPath) {
        passMs = 0.0f;
        float *grainedDensityDevice = filmDensityWithGrainOutput
          ? static_cast<float *>(destinationDevice_.pointer)
          : (filmDensityDevice == static_cast<float *>(scratchDeviceA_.pointer)
              ? static_cast<float *>(scratchDeviceB_.pointer)
              : static_cast<float *>(scratchDeviceA_.pointer));
        if (!spektraCudaPreviewGrainFromDensity(
              filmDensityDevice,
              grainedDensityDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
              static_cast<const float *>(densityCurvesDevice_.pointer),
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          "cuda_preview_grain_from_density",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
        kernelMs += passMs;
        filmDensityDevice = grainedDensityDevice;
      }

      if (printLogRawOutput || printDensityOutput) {
        passMs = 0.0f;
        float *printRawDevice = (printLogRawOutput && !printDiffusionPath)
          ? static_cast<float *>(destinationDevice_.pointer)
          : static_cast<float *>(scratchDeviceA_.pointer);
        if (!spektraCudaPrintRawFromFilmDensity(
              filmDensityDevice,
              printRawDevice,
              width,
              height,
              static_cast<const KernelParams *>(paramsDevice_.pointer),
              static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
              static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
              static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
              static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
              static_cast<const float *>(logExposureDevice_.pointer),
              static_cast<const float *>(densityCurvesDevice_.pointer),
              static_cast<const float *>(paperLogExposureDevice_.pointer),
              static_cast<const float *>(paperDensityCurvesDevice_.pointer),
              static_cast<const float *>(filmChannelDensityDevice_.pointer),
              static_cast<const float *>(filmBaseDensityDevice_.pointer),
              static_cast<const float *>(paperLogSensitivityDevice_.pointer),
              static_cast<const float *>(thKg3IlluminantDevice_.pointer),
              static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
              static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
              static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
              static_cast<const float *>(hanatosRawResponseDevice_.pointer),
              static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
              static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
              printLogRawOutput && !printDiffusionPath,
              &passMs,
              error.data(),
              error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          printLogRawOutput ? "cuda_print_log_raw_from_film_density" : "cuda_print_raw_from_film_density",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
        kernelMs += passMs;

        if (printDiffusionPath) {
          if (!dispatchDiffusion(
                "cuda_print",
                printRawDevice,
                static_cast<float *>(destinationDevice_.pointer),
                static_cast<float *>(sourceDevice_.pointer),
                printRawDevice,
                static_cast<const KernelDiffusionInfo *>(printDiffusionInfoDevice_.pointer),
                static_cast<const KernelDiffusionComponent *>(printDiffusionComponentsDevice_.pointer),
                printDiffusionComponents)) {
            return false;
          }
          if (printLogRawOutput) {
            passMs = 0.0f;
            if (!spektraCudaRawToLogRaw(
                  printRawDevice,
                  static_cast<float *>(destinationDevice_.pointer),
                  width,
                  height,
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_print_raw_to_log_raw",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 2u,
              true
            });
            kernelMs += passMs;
          }
        }
      }

      if (printDensityOutput) {
        passMs = 0.0f;
        if (!spektraCudaPrintDensityFromPrintRaw(
            static_cast<const float *>(scratchDeviceA_.pointer),
            static_cast<float *>(destinationDevice_.pointer),
            width,
            height,
            static_cast<const KernelParams *>(paramsDevice_.pointer),
            static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
            static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
            static_cast<const float *>(paperLogExposureDevice_.pointer),
            static_cast<const float *>(paperDensityCurvesDevice_.pointer),
            &passMs,
            error.data(),
            error.size())) {
          lastError_ = error.data();
          return false;
        }
        diagnostics_.passes.push_back({
          "cuda_print_density_from_print_raw",
          static_cast<double>(passMs),
          static_cast<uint32_t>(width),
          static_cast<uint32_t>(height),
          1u,
          256u,
          1u,
          static_cast<uint64_t>(bytes) * 2u,
          true
        });
        kernelMs += passMs;
      }
      // Final output chooses scan-negative or print simulation, then scanner post.
      if (finalOutput) {
        passMs = 0.0f;
        const bool scannerPostPath = params.scannerEnabled && !rcmOutput;
        if (printDiffusionPath) {
          float *printRawDevice = static_cast<float *>(scratchDeviceA_.pointer);
          if (!spektraCudaPrintRawFromFilmDensity(
                filmDensityDevice,
                printRawDevice,
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
                static_cast<const float *>(logExposureDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperLogExposureDevice_.pointer),
                static_cast<const float *>(paperDensityCurvesDevice_.pointer),
                static_cast<const float *>(filmChannelDensityDevice_.pointer),
                static_cast<const float *>(filmBaseDensityDevice_.pointer),
                static_cast<const float *>(paperLogSensitivityDevice_.pointer),
                static_cast<const float *>(thKg3IlluminantDevice_.pointer),
                static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
                static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
                static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
                static_cast<const float *>(hanatosRawResponseDevice_.pointer),
                static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
                static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
                false,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_print_raw_from_film_density",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;

          if (!dispatchDiffusion(
                "cuda_print",
                printRawDevice,
                static_cast<float *>(destinationDevice_.pointer),
                static_cast<float *>(sourceDevice_.pointer),
                printRawDevice,
                static_cast<const KernelDiffusionInfo *>(printDiffusionInfoDevice_.pointer),
                static_cast<const KernelDiffusionComponent *>(printDiffusionComponentsDevice_.pointer),
                printDiffusionComponents)) {
            return false;
          }

          passMs = 0.0f;
          if (!spektraCudaFinalFromPrintRaw(
                printRawDevice,
                static_cast<float *>(destinationDevice_.pointer),
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
                static_cast<const float *>(logExposureDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperLogExposureDevice_.pointer),
                static_cast<const float *>(paperDensityCurvesDevice_.pointer),
                static_cast<const float *>(filmChannelDensityDevice_.pointer),
                static_cast<const float *>(filmBaseDensityDevice_.pointer),
                static_cast<const float *>(paperLogSensitivityDevice_.pointer),
                static_cast<const float *>(thKg3IlluminantDevice_.pointer),
                static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
                static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
                static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
                static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
                static_cast<const float *>(colorEncodeLutDevice_.pointer),
                static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
                static_cast<const float *>(hanatosRawResponseDevice_.pointer),
                static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
                static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
                !scannerPostPath,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_final_from_print_raw",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;
        } else {
          if (!ensureDeviceBuffer(frameConstantsDevice_, sizeof(KernelFrameConstants))) {
            return false;
          }
          passMs = 0.0f;
          if (!spektraCudaMakeFrameConstants(
                static_cast<KernelFrameConstants *>(frameConstantsDevice_.pointer),
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
                static_cast<const float *>(logExposureDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperLogExposureDevice_.pointer),
                static_cast<const float *>(paperDensityCurvesDevice_.pointer),
                static_cast<const float *>(filmChannelDensityDevice_.pointer),
                static_cast<const float *>(filmBaseDensityDevice_.pointer),
                static_cast<const float *>(paperLogSensitivityDevice_.pointer),
                static_cast<const float *>(thKg3IlluminantDevice_.pointer),
                static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
                static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
                static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
                static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
                static_cast<const float *>(hanatosRawResponseDevice_.pointer),
                static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
                static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          kernelMs += passMs;
          passMs = 0.0f;
          if (!spektraCudaFinalFromFilmDensity(
                filmDensityDevice,
                static_cast<float *>(destinationDevice_.pointer),
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(curveInfoDevice_.pointer),
                static_cast<const KernelCurveInfo *>(paperCurveInfoDevice_.pointer),
                static_cast<const float *>(logExposureDevice_.pointer),
                static_cast<const float *>(densityCurvesDevice_.pointer),
                static_cast<const float *>(paperLogExposureDevice_.pointer),
                static_cast<const float *>(paperDensityCurvesDevice_.pointer),
                static_cast<const float *>(filmChannelDensityDevice_.pointer),
                static_cast<const float *>(filmBaseDensityDevice_.pointer),
                static_cast<const float *>(paperLogSensitivityDevice_.pointer),
                static_cast<const float *>(thKg3IlluminantDevice_.pointer),
                static_cast<const float *>(customEnlargerFiltersDevice_.pointer),
                static_cast<const float *>(neutralPrintFiltersDevice_.pointer),
                static_cast<const float *>(academyPrinterDensityDataDevice_.pointer),
                static_cast<const float *>(paperScanDensityDataDevice_.pointer),
                static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
                static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
                static_cast<const float *>(colorEncodeLutDevice_.pointer),
                static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
                static_cast<const KernelFrameConstants *>(frameConstantsDevice_.pointer),
                static_cast<const float *>(hanatosRawResponseDevice_.pointer),
                static_cast<const float *>(mallettBasisIlluminantDevice_.pointer),
                static_cast<const float *>(inputToReferenceXyzDevice_.pointer),
                !scannerPostPath,
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_final_from_film_density",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;
        }
        if (scannerPostPath) {
          const KernelParams &kp = kernelParams;
          const bool printGlarePath =
            params.process == ProcessMode::PrintSimulation && kp.glarePercent > 0.0f;
          if (printGlarePath) {
            passMs = 0.0f;
            if (!spektraCudaPrintGlareGenerate(
                  static_cast<float *>(scratchDeviceA_.pointer),
                  width,
                  height,
                  static_cast<const KernelParams *>(paramsDevice_.pointer),
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_print_glare_generate",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes),
              true
            });
            kernelMs += passMs;

            if (kp.glareBlur > 1.0e-4f) {
              passMs = 0.0f;
              if (!spektraCudaGaussianBlurX(
                    static_cast<const float *>(scratchDeviceA_.pointer),
                    static_cast<float *>(scratchDeviceB_.pointer),
                    width,
                    height,
                    kp.glareBlur,
                    &passMs,
                    error.data(),
                    error.size())) {
                lastError_ = error.data();
                return false;
              }
              diagnostics_.passes.push_back({
                "cuda_print_glare_blur_x",
                static_cast<double>(passMs),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
                1u,
                256u,
                1u,
                static_cast<uint64_t>(bytes) * 2u,
                true
              });
              kernelMs += passMs;

              passMs = 0.0f;
              if (!spektraCudaGaussianBlurY(
                    static_cast<const float *>(scratchDeviceB_.pointer),
                    static_cast<float *>(scratchDeviceA_.pointer),
                    width,
                    height,
                    kp.glareBlur,
                    &passMs,
                    error.data(),
                    error.size())) {
                lastError_ = error.data();
                return false;
              }
              diagnostics_.passes.push_back({
                "cuda_print_glare_blur_y",
                static_cast<double>(passMs),
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height),
                1u,
                256u,
                1u,
                static_cast<uint64_t>(bytes) * 2u,
                true
              });
              kernelMs += passMs;
            }

            passMs = 0.0f;
            if (!spektraCudaPrintGlareApply(
                  static_cast<const float *>(destinationDevice_.pointer),
                  static_cast<const float *>(scratchDeviceA_.pointer),
                  static_cast<float *>(destinationDevice_.pointer),
                  width,
                  height,
                  static_cast<const KernelParams *>(paramsDevice_.pointer),
                  static_cast<const KernelSpectralInfo *>(spectralInfoDevice_.pointer),
                  static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                  static_cast<const float *>(scanIlluminantsAndCmfsDevice_.pointer),
                  static_cast<const float *>(scanToOutputRgbDataDevice_.pointer),
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_print_glare_apply",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 3u,
              true
            });
            kernelMs += passMs;
          }

          if (kp.scannerBlurSigmaPx > 1.0e-4f) {
            passMs = 0.0f;
            if (!spektraCudaGaussianBlurX(
                  static_cast<const float *>(destinationDevice_.pointer),
                  static_cast<float *>(scratchDeviceA_.pointer),
                  width,
                  height,
                  kp.scannerBlurSigmaPx,
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_scanner_blur_x",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 2u,
              true
            });
            kernelMs += passMs;

            passMs = 0.0f;
            if (!spektraCudaGaussianBlurY(
                  static_cast<const float *>(scratchDeviceA_.pointer),
                  static_cast<float *>(destinationDevice_.pointer),
                  width,
                  height,
                  kp.scannerBlurSigmaPx,
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_scanner_blur_y",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 2u,
              true
            });
            kernelMs += passMs;
          }

          const bool needsUnsharp = kp.scannerUnsharpSigmaPx > 1.0e-4f && kp.scannerUnsharpAmount > 0.0f;
          const float *unsharpDevice = nullptr;
          if (needsUnsharp) {
            passMs = 0.0f;
            if (!spektraCudaGaussianBlurX(
                  static_cast<const float *>(destinationDevice_.pointer),
                  static_cast<float *>(scratchDeviceA_.pointer),
                  width,
                  height,
                  kp.scannerUnsharpSigmaPx,
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_unsharp_blur_x",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 2u,
              true
            });
            kernelMs += passMs;

            passMs = 0.0f;
            if (!spektraCudaGaussianBlurY(
                  static_cast<const float *>(scratchDeviceA_.pointer),
                  static_cast<float *>(scratchDeviceB_.pointer),
                  width,
                  height,
                  kp.scannerUnsharpSigmaPx,
                  &passMs,
                  error.data(),
                  error.size())) {
              lastError_ = error.data();
              return false;
            }
            diagnostics_.passes.push_back({
              "cuda_unsharp_blur_y",
              static_cast<double>(passMs),
              static_cast<uint32_t>(width),
              static_cast<uint32_t>(height),
              1u,
              256u,
              1u,
              static_cast<uint64_t>(bytes) * 2u,
              true
            });
            kernelMs += passMs;
            unsharpDevice = static_cast<const float *>(scratchDeviceB_.pointer);
          }

          passMs = 0.0f;
          if (!spektraCudaScannerFinalize(
                static_cast<const float *>(destinationDevice_.pointer),
                unsharpDevice,
                static_cast<float *>(destinationDevice_.pointer),
                width,
                height,
                static_cast<const KernelParams *>(paramsDevice_.pointer),
                static_cast<const KernelColorInfo *>(colorInfoDevice_.pointer),
                static_cast<const float *>(colorEncodeLutDevice_.pointer),
                static_cast<const uint32_t *>(colorTransferKindDevice_.pointer),
                &passMs,
                error.data(),
                error.size())) {
            lastError_ = error.data();
            return false;
          }
          diagnostics_.passes.push_back({
            "cuda_scanner_finalize",
            static_cast<double>(passMs),
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height),
            1u,
            256u,
            1u,
            static_cast<uint64_t>(bytes) * 2u,
            true
          });
          kernelMs += passMs;
          diagnostics_.finalPostProcessPath = true;
        }
      }
    }
    }
  }
  {
    const auto start = std::chrono::steady_clock::now();
    if (hostCudaRender) {
      // layout conversion back into Resolve's CUDA image, still GPU-to-GPU
      float passMs = 0.0f;
      if (!spektraCudaUnpackFloatToDeviceImage(
            static_cast<const float *>(destinationDevice_.pointer),
            destination.data,
            destination.x1,
            destination.y1,
            destination.rowBytes,
            destination.bytesPerComponent,
            effectiveWindow.x1,
            effectiveWindow.y1,
            width,
            height,
            &passMs,
            error.data(),
            error.size())) {
        lastError_ = error.data();
        return false;
      }
      const cudaError_t syncStatus = cudaStreamSynchronize(0);
      if (syncStatus != cudaSuccess) {
        lastError_ = std::string("CUDA host-device render completion failed: ") + cudaGetErrorString(syncStatus);
        return false;
      }
      kernelMs += passMs;
      diagnostics_.passes.push_back({
        "cuda_host_image_unpack",
        static_cast<double>(passMs),
        static_cast<uint32_t>(width),
        static_cast<uint32_t>(height),
        1u,
        256u,
        1u,
        static_cast<uint64_t>(bytes) * 2u,
        true
      });
      diagnostics_.cudaDeviceToHostMs = 0.0;
    } else {
      // Host-image fallback only; Resolve CUDA images use the device unpack path above.
      cudaError_t status = cudaMemcpyAsync(destinationDownloadHost, destinationDevice_.pointer, bytes, cudaMemcpyDeviceToHost, 0);
      if (status == cudaSuccess) {
        status = cudaStreamSynchronize(0);
      }
      diagnostics_.cudaDeviceToHostMs = elapsedMs(start, std::chrono::steady_clock::now());
      if (status != cudaSuccess) {
        std::ostringstream out;
        out << "cudaMemcpy D2H failed: " << cudaGetErrorString(status);
        lastError_ = out.str();
        return false;
      }
    }
  }

  {
    const auto start = std::chrono::steady_clock::now();
    if (!hostCudaRender && !directDestinationHostCopy) {
      if (destinationTightFloat) {
        std::memcpy(destinationWindowFloatPointer(destination, effectiveWindow), destinationDownloadHost, bytes);
      } else {
        copyFloatStagingToDestination(destinationDownloadHost, destination, effectiveWindow, width, height);
      }
    }
    diagnostics_.outputCopyMs = elapsedMs(start, std::chrono::steady_clock::now());
  }

  diagnostics_.cudaKernelMs = cudaPassTiming ? kernelMs : 0.0;
  diagnostics_.commandBufferMs = elapsedMs(gpuWorkStart, std::chrono::steady_clock::now());
  diagnostics_.passCount = static_cast<uint32_t>(diagnostics_.passes.size());
  for (RendererPassDiagnostics &pass : diagnostics_.passes) {
    pass.gpuTimeAvailable = cudaPassTiming;
  }
  if (copyDiagnostic) {
    diagnostics_.backendFallbackReason =
      "CUDA copy diagnostic is active; film processing passes were intentionally skipped.";
  } else if (hostCudaRender) {
    diagnostics_.backendFallbackReason =
      "Host-native CUDA device images are active; full-frame CPU staging and H2D/D2H transfers are disabled.";
  } else {
    diagnostics_.backendFallbackReason =
      "CUDA film/print/final slice is active for FilmLogRaw, FilmDensityCmy, FilmDensityCmyWithGrain preview/production/synthesis grain, PrintLogRaw, PrintDensityCmy, FinalPreview, diffusion, auto exposure, enlarger resample, print glare, and scanner blur/unsharp.";
  }
  if (grainSynthesisSamplesCapped) {
    std::ostringstream note;
    note << diagnostics_.backendFallbackReason
         << " Grain synthesis samples capped for CUDA interactive render: "
         << grainSynthesisRequestedSamples << " -> " << grainSynthesisEffectiveSamples
         << " (SPEKTRAFILM_CUDA_GRAIN_SYNTHESIS_SAMPLE_CAP overrides; 0 disables).";
    diagnostics_.backendFallbackReason = note.str();
  }
  return true;
}

bool CudaRenderer::render(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window,
  const RenderParams &params,
  double time
) {
  // CUDA gate: the native selector can fall back before render, but CUDA still reports why it refused work
  lastError_.clear();
  if (!initialize()) {
    return false;
  }

  bool cudaDensityOutput = false;
  std::string cudaIneligibleReason;
  const bool cudaCopyDiagnostic = envFlag("SPEKTRAFILM_CUDA_COPY_DIAGNOSTIC");
  const bool cudaFilmEligible = cudaFilmPipelineEligible(params, cudaDensityOutput, cudaIneligibleReason);
  if (!cudaCopyDiagnostic && !cudaFilmEligible) {
    lastError_ = cudaIneligibleReason.empty()
      ? "The requested render path is not implemented by the CUDA backend."
      : cudaIneligibleReason;
    return false;
  }

  return renderCudaOwned(source, destination, window, params, time);
}

std::unique_ptr<Renderer> createCudaRenderer() {
  return std::make_unique<CudaRenderer>();
}

} // namespace spektrafilm
