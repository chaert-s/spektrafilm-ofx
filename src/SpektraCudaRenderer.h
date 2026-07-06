#pragma once

#include "SpektraRenderCore.h"
#include "SpektraRenderer.h"

#include <memory>
#include <cstddef>
#include <vector>

namespace spektrafilm {

class CudaRenderer final : public Renderer {
public:
  CudaRenderer();
  ~CudaRenderer() override;

  CudaRenderer(const CudaRenderer &) = delete;
  CudaRenderer &operator=(const CudaRenderer &) = delete;

  bool isAvailable() const override;
  const std::string &lastError() const override;
  const RendererDiagnostics &lastDiagnostics() const override;

  bool render(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  ) override;

private:
  struct DeviceBuffer {
    void *pointer = nullptr;
    size_t bytes = 0;
  };

  struct PinnedHostBuffer {
    void *pointer = nullptr;
    size_t bytes = 0;
  };

  bool initialize();
  bool ensureDeviceBuffer(DeviceBuffer &buffer, size_t bytes);
  void releaseDeviceBuffer(DeviceBuffer &buffer);
  bool ensurePinnedHostBuffer(PinnedHostBuffer &buffer, size_t bytes);
  void releasePinnedHostBuffer(PinnedHostBuffer &buffer);
  bool uploadDeviceBytes(DeviceBuffer &buffer, const void *data, size_t bytes);
  template <typename T>
  bool uploadDeviceStruct(DeviceBuffer &buffer, const T &value) {
    return uploadDeviceBytes(buffer, &value, sizeof(T));
  }
  bool uploadDeviceFloats(DeviceBuffer &buffer, const std::vector<float> &values);
  bool uploadDeviceUInts(DeviceBuffer &buffer, const std::vector<uint32_t> &values);
  bool ensureStaticResources(const RenderParams &params);
  bool cudaFilmPipelineEligible(const RenderParams &params, bool &densityOutput, std::string &reason) const;
  bool renderCudaOwned(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  );

  std::string lastError_;
  RendererDiagnostics diagnostics_;
  std::string deviceName_;
  int computeCapabilityMajor_ = 0;
  int computeCapabilityMinor_ = 0;
  bool initialized_ = false;
  bool available_ = false;
  int deviceIndex_ = 0;
  DeviceBuffer sourceDevice_;
  DeviceBuffer destinationDevice_;
  DeviceBuffer autoExposurePreviewDevice_;
  DeviceBuffer scratchDeviceA_;
  DeviceBuffer scratchDeviceB_;
  DeviceBuffer diffusionGroupTempDevice_;
  DeviceBuffer diffusionReducedSourceDevice_;
  DeviceBuffer diffusionReducedTempDevice_;
  DeviceBuffer diffusionReducedBlurDevice_;
  DeviceBuffer dirTailScratchDevice_;
  DeviceBuffer paramsDevice_;
  DeviceBuffer frameConstantsDevice_;
  DeviceBuffer dirInfoDevice_;
  DeviceBuffer dirCoreBlurInfoDevice_;
  DeviceBuffer dirTailBlurInfosDevice_;
  DeviceBuffer dirCorrectedDensityCurvesDevice_;
  DeviceBuffer halationBoostInfoDevice_;
  DeviceBuffer cameraDiffusionInfoDevice_;
  DeviceBuffer cameraDiffusionComponentsDevice_;
  DeviceBuffer printDiffusionInfoDevice_;
  DeviceBuffer printDiffusionComponentsDevice_;
  DeviceBuffer grainLayerDeviceA_;
  DeviceBuffer grainLayerDeviceB_;
  DeviceBuffer grainMicroDeviceA_;
  DeviceBuffer grainMicroDeviceB_;
  DeviceBuffer spectralInfoDevice_;
  DeviceBuffer colorInfoDevice_;
  DeviceBuffer curveInfoDevice_;
  DeviceBuffer hanatosRawResponseDevice_;
  DeviceBuffer paperHanatosResponseDevice_;
  DeviceBuffer preflashPaperHanatosResponseDevice_;
  DeviceBuffer mallettBasisIlluminantDevice_;
  DeviceBuffer inputToReferenceXyzDevice_;
  DeviceBuffer inputToSrgbDevice_;
  DeviceBuffer colorDecodeLutDevice_;
  DeviceBuffer colorTransferKindDevice_;
  DeviceBuffer logExposureDevice_;
  DeviceBuffer densityCurvesDevice_;
  DeviceBuffer paperCurveInfoDevice_;
  DeviceBuffer paperLogExposureDevice_;
  DeviceBuffer paperDensityCurvesDevice_;
  DeviceBuffer filmChannelDensityDevice_;
  DeviceBuffer filmBaseDensityDevice_;
  DeviceBuffer paperLogSensitivityDevice_;
  DeviceBuffer thKg3IlluminantDevice_;
  DeviceBuffer customEnlargerFiltersDevice_;
  DeviceBuffer neutralPrintFiltersDevice_;
  DeviceBuffer academyPrinterDensityDataDevice_;
  DeviceBuffer paperScanDensityDataDevice_;
  DeviceBuffer scanIlluminantsAndCmfsDevice_;
  DeviceBuffer scanToOutputRgbDataDevice_;
  DeviceBuffer colorEncodeLutDevice_;
  StaticProfileResourceData staticResources_;
  std::vector<float> hanatosSpectraData_;
  std::vector<float> outputGamutCompressionData_;
  bool staticBuffersUploaded_ = false;
  PinnedHostBuffer pinnedSourceStaging_;
  PinnedHostBuffer pinnedDestinationStaging_;
  PinnedHostBuffer pinnedAutoExposurePreview_;
};

std::unique_ptr<Renderer> createCudaRenderer();

} // namespace spektrafilm
