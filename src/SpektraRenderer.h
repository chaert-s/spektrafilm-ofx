#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "SpektraParameters.h"

namespace spektrafilm {

struct RendererPassDiagnostics {
  std::string name;
  double gpuMs = 0.0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t depth = 1;
  uint32_t threadgroupWidth = 0;
  uint32_t threadgroupHeight = 0;
  uint64_t estimatedBytes = 0;
  bool gpuTimeAvailable = false;
};

struct RendererDiagnostics {
  std::string backendName;
  std::string backendFallbackReason;
  double cpuSetupMs = 0.0;
  double sourceCopyMs = 0.0;
  double commandBufferMs = 0.0;
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
  uint32_t passCount = 0;
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
  uint32_t diffusionGroupSize = 2;
  std::string threadgroupMode = "auto";
  std::string passTimingMode;
  std::string blurBackend = "custom";
  std::string blurDownsample = "auto";
  std::string intermediatePrecision = "float";
  std::string diffusionClusterSigma = "0.10";
  std::string dirTailBackend = "mps";
  std::string densityCurveLookup = "binary";
  std::string spectralTransmittance = "pow";
  std::string deviceName;
  std::string cudaTransferMode;
  int cudaComputeCapabilityMajor = 0;
  int cudaComputeCapabilityMinor = 0;
  std::vector<RendererPassDiagnostics> passes;
};

class Renderer {
public:
  virtual ~Renderer() = default;

  virtual bool isAvailable() const = 0;
  virtual const std::string &lastError() const = 0;
  virtual const RendererDiagnostics &lastDiagnostics() const = 0;

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
