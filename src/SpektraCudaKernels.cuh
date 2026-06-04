#pragma once

#include "SpektraRenderCore.h"

#include <cstddef>
#include <cstdint>

namespace spektrafilm {

struct CudaDeviceInfo {
  char name[256];
  int major;
  int minor;
};

bool spektraCudaInitialize(int *deviceIndex, CudaDeviceInfo *deviceInfo, char *error, size_t errorSize);
bool spektraCudaSmokeCopy(const float *source, float *destination, size_t floatCount, float *kernelMs, char *error, size_t errorSize);
bool spektraCudaCopyFrame(
  const float *source,
  float *destination,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

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
);

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
);

bool spektraCudaEnlargerResample(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaRawToLogRaw(
  const float *raw,
  float *logRaw,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

bool spektraCudaDirBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaDirBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelGaussianBlurInfo *blurInfo,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaDirTailBlurX(
  const float *source,
  float *tailPlanes,
  int width,
  int height,
  const KernelGaussianBlurInfo *tailBlurInfos,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

bool spektraCudaClearFrame(
  float *destination,
  int width,
  int height,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

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
);

bool spektraCudaDiffusionDownsample(
  const float *source,
  float *destination,
  int width,
  int height,
  uint32_t scale,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

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
);

bool spektraCudaHalationBoostInfo(
  const float *raw,
  float *boostInfo,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

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
);

bool spektraCudaGrainMicrostructureSource(
  float *micro,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaGrainMicroBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaGrainMicroBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

bool spektraCudaGrainDensityBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaGrainDensityBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

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
);

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
);

bool spektraCudaGaussianBlurX(
  const float *source,
  float *destination,
  int width,
  int height,
  float sigma,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaGaussianBlurY(
  const float *source,
  float *destination,
  int width,
  int height,
  float sigma,
  float *kernelMs,
  char *error,
  size_t errorSize
);

bool spektraCudaPrintGlareGenerate(
  float *glareAmount,
  int width,
  int height,
  const KernelParams *params,
  float *kernelMs,
  char *error,
  size_t errorSize
);

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
);

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
);

} // namespace spektrafilm
