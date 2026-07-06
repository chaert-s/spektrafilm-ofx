#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "SpektraParameters.h"

namespace spektrafilm {

struct RendererPassDiagnostics {
  RendererPassDiagnostics() = default;

  RendererPassDiagnostics(
    std::string passName,
    double passGpuMs,
    uint32_t passWidth,
    uint32_t passHeight,
    uint32_t passDepth,
    uint32_t passThreadgroupWidth,
    uint32_t passThreadgroupHeight,
    uint64_t passEstimatedBytes,
    bool passGpuTimeAvailable
  )
    : name(std::move(passName))
    , gpuMs(passGpuMs)
    , width(passWidth)
    , height(passHeight)
    , depth(passDepth)
    , threadgroupWidth(passThreadgroupWidth)
    , threadgroupHeight(passThreadgroupHeight)
    , estimatedBytes(passEstimatedBytes)
    , gpuTimeAvailable(passGpuTimeAvailable) {}

  std::string name;
  double gpuMs = 0.0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 1;
  uint32_t threadgroupWidth = 0;
  uint32_t threadgroupHeight = 0;
  uint32_t threadExecutionWidth = 0;
  uint32_t maxTotalThreadsPerThreadgroup = 0;
  uint64_t estimatedBytes = 0;
  bool gpuTimeAvailable = false;
};

struct RendererTileDiagnostics {
  uint32_t index = 0;
  int32_t outputX = 0;
  int32_t outputY = 0;
  int32_t outputWidth = 0;
  int32_t outputHeight = 0;
  int32_t workingX = 0;
  int32_t workingY = 0;
  int32_t workingWidth = 0;
  int32_t workingHeight = 0;
  uint32_t overlap = 0;
  uint64_t allocatedBytes = 0;
  double submitMs = 0.0;
  uint32_t passCount = 0;
};

struct RendererDiagnostics {
  std::string backendName;
  std::string backendFallbackReason;
  double cpuSetupMs = 0.0;
  double sourceCopyMs = 0.0;
  double commandEncodingMs = 0.0;
  double commandBufferMs = 0.0;
  double gpuCommandBufferMs = 0.0;
  double outputCopyMs = 0.0;
  double cudaHostToDeviceMs = 0.0;
  double cudaDeviceToHostMs = 0.0;
  double cudaKernelMs = 0.0;
  uint64_t staticAllocationBytes = 0;
  uint64_t staticAllocationCount = 0;
  uint64_t scratchAllocationBytes = 0;
  uint64_t scratchAllocationCount = 0;
  uint64_t cudaPinnedStagingBytes = 0;
  uint64_t cudaDeviceScratchBytes = 0;
  uint64_t sharedScratchAllocationBytes = 0;
  uint64_t sharedScratchAllocationCount = 0;
  uint64_t privateScratchAllocationBytes = 0;
  uint64_t privateScratchAllocationCount = 0;
  uint64_t uploadBytes = 0;
  bool sharedBackend = false;
  uint32_t sharedBackendGeneration = 0;
  uint32_t sharedQueueCount = 0;
  uint64_t transientCachedBytes = 0;
  uint64_t transientBudgetBytes = 0;
  uint32_t passCount = 0;
  std::string metalDeviceName;
  uint64_t metalRecommendedMaxWorkingSetSize = 0;
  uint64_t metalCurrentAllocatedSize = 0;
  uint64_t metalMaxBufferLength = 0;
  bool passDispatchCounterSamplingSupported = false;
  bool passStageCounterSamplingSupported = false;
  bool sourceNoCopy = false;
  bool destinationNoCopy = false;
  bool cudaMappedHostMemory = false;
  bool cudaGraphEnabled = false;
  bool passGpuTimingEnabled = false;
  bool passGpuTimingAvailable = false;
  bool privateScratchEnabled = false;
  bool renderSerialized = false;
  bool halationPath = false;
  bool cameraDiffusionPath = false;
  bool printDiffusionPath = false;
  bool dirPath = false;
  bool productionGrainPath = false;
  bool grainSynthesisPath = false;
  bool finalPostProcessPath = false;
  bool scannerTextureIntermediates = false;
  bool halationGroupedTail = false;
  bool scannerMps = false;
  bool grainBlurRecurrence = true;
  bool tiledRendering = false;
  uint32_t tileCount = 0;
  uint32_t tileWidth = 0;
  uint32_t tileHeight = 0;
  uint32_t tileOverlap = 0;
  uint32_t diffusionGroupSize = 2;
  std::string deviceName;
  std::string cudaTransferMode;
  int cudaComputeCapabilityMajor = 0;
  int cudaComputeCapabilityMinor = 0;
  std::string threadgroupMode = "auto";
  std::string passTimingMode;
  std::string blurBackend = "custom";
  std::string blurDownsample = "auto";
  std::string intermediatePrecision = "float";
  std::string diffusionClusterSigma = "0.10";
  std::string dirTailBackend = "mps";
  std::string densityCurveLookup = "binary";
  std::string spectralTransmittance = "pow";
  std::string finalCoreMode = "fused";
  std::vector<RendererPassDiagnostics> passes;
  std::vector<RendererTileDiagnostics> tiles;
};

class Renderer {
public:
  virtual ~Renderer() = default;

  virtual bool isAvailable() const = 0;
  virtual const std::string &lastError() const = 0;
  virtual const RendererDiagnostics &lastDiagnostics() const = 0;
  virtual void releaseTransientResources() {}

  virtual bool render(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  ) = 0;
};

std::unique_ptr<Renderer> createNativeRenderer();

} // namespace spektrafilm
