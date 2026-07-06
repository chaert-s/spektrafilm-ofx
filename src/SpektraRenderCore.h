#pragma once

#include "SpektraParameters.h"
#include "SpektraProfileCurves.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace spektrafilm {

struct KernelParams {
  int32_t process;
  int32_t rgbToRawMethod;
  int32_t inputColorSpace;
  int32_t outputColorSpace;
  int32_t outputRole;
  int32_t hdrPreset;
  int32_t hdrTransfer;
  float hdrReferenceWhiteNits;
  float hdrPeakNits;
  float hdrExposureEv;
  int32_t hdrToneMapping;
  uint32_t colorAdaptationFlags;
  int32_t film;
  int32_t paper;
  int32_t printTiming;
  float filmExposureEv;
  uint32_t autoExposureEnabled;
  int32_t autoExposureMethod;
  float autoExposureEv;
  float _padAutoExposure0;
  float printExposureEv;
  float filmGamma;
  float printGamma;
  float printShadowShape;
  float printHighlightShape;
  int32_t filmPushPullMode;
  float filmPushPullStops;
  int32_t printPushPullMode;
  float printPushPullStops;
  float negativeBleachBypassAmount;
  float negativeLeucoCyanCoupling;
  float printBleachBypassAmount;
  uint32_t scanNegativeInvert;
  float filterC;
  float filterMShift;
  float filterYShift;
  float enlargerScale;
  float enlargerOffsetXPercent;
  float enlargerOffsetYPercent;
  float _padEnlarger0;
  float preflashExposure;
  float preflashMFilterShift;
  float preflashYFilterShift;
  float printerLightsR;
  float printerLightsG;
  float printerLightsB;
  uint32_t printerLightsGang;
  uint32_t printerLightCalibration;
  float dirCouplersAmount;
  float dirCouplersDiffusionUm;
  float dirCouplersDiffusionTailUm;
  float dirCouplersDiffusionTailWeight;
  uint32_t grainEnabled;
  int32_t grainModel;
  int32_t filmFormat;
  float grainAmount;
  float grainSaturation;
  uint32_t grainSublayersEnabled;
  int32_t grainSubLayerCount;
  float grainParticleAreaUm2;
  float grainParticleScaleR;
  float grainParticleScaleG;
  float grainParticleScaleB;
  float grainParticleScaleLayer0;
  float grainParticleScaleLayer1;
  float grainParticleScaleLayer2;
  float grainDensityMinR;
  float grainDensityMinG;
  float grainDensityMinB;
  float grainUniformityR;
  float grainUniformityG;
  float grainUniformityB;
  float grainFinalBlurUm;
  float grainBlurDyeCloudsUm;
  float grainMicroStructureScale;
  float grainMicroStructureSigmaNm;
  uint32_t grainSeed;
  uint32_t grainAnimate;
  float filmPixelSizeUm;
  float _padGrain0;
  int32_t grainSynthesisSamples;
  float grainSynthesisAmount;
  float grainSynthesisMeanRadiusUm;
  float grainSynthesisRadiusStdDevRatio;
  float grainSynthesisObservationSigmaUm;
  float grainSynthesisCellSizeRatio;
  float grainSynthesisMaxRadiusQuantile;
  float grainSynthesisCoverageEpsilon;
  int32_t grainSynthesisMaxGrainsPerCell;
  float grainSynthesisRadiusScaleR;
  float grainSynthesisRadiusScaleG;
  float grainSynthesisRadiusScaleB;
  float grainSynthesisLayerScale0;
  float grainSynthesisLayerScale1;
  float grainSynthesisLayerScale2;
  uint32_t grainSynthesisLayered;
  uint32_t _padGrainSynthesis0;
  uint32_t halationEnabled;
  float scatterAmount;
  float scatterScale;
  float halationAmount;
  float halationScale;
  float halationStrengthR;
  float halationStrengthG;
  float halationStrengthB;
  float halationFirstSigmaUmR;
  float halationFirstSigmaUmG;
  float halationFirstSigmaUmB;
  float halationBoostEv;
  float halationBoostRange;
  float halationProtectEv;
  float _padHalation0;
  uint32_t cameraDiffusionEnabled;
  int32_t cameraDiffusionFamily;
  float cameraDiffusionStrength;
  float cameraDiffusionSpatialScale;
  float cameraDiffusionHaloWarmth;
  float cameraDiffusionCoreIntensity;
  float cameraDiffusionCoreSize;
  float cameraDiffusionHaloIntensity;
  float cameraDiffusionHaloSize;
  float cameraDiffusionBloomIntensity;
  float cameraDiffusionBloomSize;
  uint32_t printDiffusionEnabled;
  int32_t printDiffusionFamily;
  float printDiffusionStrength;
  float printDiffusionSpatialScale;
  float printDiffusionHaloWarmth;
  float printDiffusionCoreIntensity;
  float printDiffusionCoreSize;
  float printDiffusionHaloIntensity;
  float printDiffusionHaloSize;
  float printDiffusionBloomIntensity;
  float printDiffusionBloomSize;
  uint32_t scannerEnabled;
  uint32_t scannerWhiteCorrection;
  uint32_t scannerBlackCorrection;
  float scannerWhiteLevel;
  float scannerBlackLevel;
  float glarePercent;
  float glareRoughness;
  float glareBlur;
  float scannerBlurSigmaPx;
  float scannerUnsharpSigmaPx;
  float scannerUnsharpAmount;
  uint32_t densityCurveLookupMode;
  uint32_t spectralTransmittanceMode;
  uint32_t _padPerf0;
  float time;
};

static_assert(sizeof(KernelParams) == 596u, "KernelParams must match Metal and CUDA shader layout.");

struct KernelDiffusionInfo {
  uint32_t componentCount = 0;
  float scatterFraction = 0.0f;
  uint32_t _pad0 = 0;
  uint32_t _pad1 = 0;
};

struct KernelDiffusionComponent {
  float sigmaPx = 0.0f;
  float weightR = 0.0f;
  float weightG = 0.0f;
  float weightB = 0.0f;
};

struct KernelDirInfo {
  float matrix00 = 0.0f;
  float matrix01 = 0.0f;
  float matrix02 = 0.0f;
  float matrix10 = 0.0f;
  float matrix11 = 0.0f;
  float matrix12 = 0.0f;
  float matrix20 = 0.0f;
  float matrix21 = 0.0f;
  float matrix22 = 0.0f;
  float densityMax0 = 0.0f;
  float densityMax1 = 0.0f;
  float densityMax2 = 0.0f;
};

struct KernelGaussianBlurInfo {
  float firstWeight = 0.0f;
  float firstRatio = 0.0f;
  float ratioStep = 0.0f;
  float invWeightSum = 1.0f;
  uint32_t radius = 0u;
  uint32_t active = 0u;
  uint32_t _pad0 = 0u;
  uint32_t _pad1 = 0u;
};

struct KernelCurveInfo {
  uint32_t exposureCount = 0;
  uint32_t _pad0 = 0;
  uint32_t _pad1 = 0;
  uint32_t _pad2 = 0;
};

struct KernelSpectralInfo {
  uint32_t filmWavelengthCount = 0;
  uint32_t hanatosWidth = 0;
  uint32_t hanatosHeight = 0;
  uint32_t hanatosWavelengthCount = 0;
  uint32_t filmCount = 0;
  uint32_t paperCount = 0;
  uint32_t filmPositive = 0;
  uint32_t _padCount1 = 0;
  float mallettRawMidgrayGreen = 1.0f;
  float filmDensityCurveMinimum0 = 0.0f;
  float filmDensityCurveMinimum1 = 0.0f;
  float filmDensityCurveMinimum2 = 0.0f;
  float filmDensityCurveMaximum0 = 0.0f;
  float filmDensityCurveMaximum1 = 0.0f;
  float filmDensityCurveMaximum2 = 0.0f;
  float _padDensityCurveMaximum0 = 0.0f;
  float paperDensityCurveMaximum0 = 0.0f;
  float paperDensityCurveMaximum1 = 0.0f;
  float paperDensityCurveMaximum2 = 0.0f;
  float _padPaperDensityCurveMaximum0 = 0.0f;
};

struct KernelColorInfo {
  uint32_t colorSpaceCount = 0;
  uint32_t transferLutSize = 0;
  float decodeMin = 0.0f;
  float decodeMax = 0.0f;
  float encodeMin = 0.0f;
  float encodeMax = 0.0f;
  float _pad0 = 0.0f;
  float _pad1 = 0.0f;
};

struct KernelFrameConstants {
  float print[4] = {};
  float film[4] = {};
  float glare[4] = {};
  float preflash[4] = {};
  float filmDmaxScan[4] = {};
  float filmDminScan[4] = {};
};

static_assert(sizeof(KernelCurveInfo) == 16u);
static_assert(sizeof(KernelSpectralInfo) == 80u);
static_assert(sizeof(KernelColorInfo) == 32u);
static_assert(sizeof(KernelDiffusionInfo) == 16u);
static_assert(sizeof(KernelDiffusionComponent) == 16u);
static_assert(sizeof(KernelDirInfo) == 48u);
static_assert(sizeof(KernelGaussianBlurInfo) == 32u);
static_assert(sizeof(KernelFrameConstants) == 96u);

struct StaticProfileResourceData {
  int32_t film = -1;
  int32_t paper = -1;
  RgbToRawMethod rgbToRawMethod = RgbToRawMethod::Hanatos2026;
  bool cameraUvFilterEnabled = false;
  float cameraUvCutNm = 410.0f;
  bool cameraIrFilterEnabled = false;
  float cameraIrCutNm = 675.0f;
  PrintTimingMode processNegativePrintTiming = PrintTimingMode::FilteredEnlarger;
  float processNegativeFilterC = 0.0f;
  float processNegativeFilterMShift = 0.0f;
  float processNegativeFilterYShift = 0.0f;
  float processNegativePreflashMFilterShift = 0.0f;
  float processNegativePreflashYFilterShift = 0.0f;
  const ProfileCurveSet *filmCurves = nullptr;
  const ProfileCurveSet *paperCurves = nullptr;
  KernelCurveInfo curveInfo{};
  KernelCurveInfo paperCurveInfo{};
  KernelColorInfo colorInfo{};
  KernelSpectralInfo spectralInfo{};
  std::vector<float> logExposure;
  std::vector<float> densityCurves;
  std::vector<float> wavelengths;
  std::vector<float> logSensitivity;
  std::vector<float> bandpassHanatos;
  std::vector<float> hanatosRawResponse;
  std::vector<float> paperHanatosResponse;
  std::vector<float> preflashPaperHanatosResponse;
  std::array<float, 9> mallettBasisIlluminant{};
  std::vector<float> inputToReferenceXyz;
  std::vector<float> inputToSrgb;
  std::vector<float> colorDecodeLut;
  std::vector<uint32_t> colorTransferKind;
  std::vector<float> paperLogExposure;
  std::vector<float> paperDensityCurves;
  std::vector<float> filmChannelDensity;
  std::vector<float> filmBaseDensity;
  std::vector<float> paperLogSensitivity;
  std::vector<float> thKg3Illuminant;
  std::vector<float> customEnlargerFilters;
  std::vector<float> neutralPrintFilters;
  std::vector<float> academyPrinterDensityData;
  std::vector<float> paperScanDensityData;
  std::vector<float> scanIlluminantsAndCmfs;
  std::vector<float> scanToOutputRgbData;
  std::vector<float> colorEncodeLut;

  bool validFor(const RenderParams &params) const;
  void reset();
};

struct EffectiveRenderWindow {
  int32_t x1 = 0;
  int32_t y1 = 0;
  int32_t x2 = 0;
  int32_t y2 = 0;

  int32_t width() const { return x2 - x1; }
  int32_t height() const { return y2 - y1; }
  bool empty() const { return x2 <= x1 || y2 <= y1; }
};

float filmFormatMm(FilmFormat format);
float enlargerScale(const RenderParams &params);
bool enlargerTransformActive(const RenderParams &params);
KernelParams toKernelParams(const RenderParams &params, double time, int32_t width, int32_t height);
std::vector<KernelDiffusionComponent> makeCameraDiffusionComponents(
  const RenderParams &params,
  float pixelSizeUm,
  KernelDiffusionInfo &info,
  float clusterSigmaRatio = 0.10f
);
std::vector<KernelDiffusionComponent> makePrintDiffusionComponents(
  const RenderParams &params,
  float pixelSizeUm,
  KernelDiffusionInfo &info,
  float clusterSigmaRatio = 0.10f
);
KernelDirInfo makeDirInfo(const ProfileCurveSet &filmCurves, const RenderParams &params);
KernelGaussianBlurInfo makeGaussianBlurInfo(float sigma, uint32_t radiusLimit);
std::array<KernelGaussianBlurInfo, 3> makeDirTailBlurInfos(const KernelParams &params);
std::vector<float> makeDirCorrectedDensityCurves(
  const ProfileCurveSet &filmCurves,
  const KernelDirInfo &dirInfo
);
bool buildStaticProfileResourceData(
  const RenderParams &params,
  const std::vector<float> &hanatosSpectraData,
  StaticProfileResourceData &resources,
  std::string &error
);

bool validateRgbaFloatOrHalfImages(
  const ImageView &source,
  const MutableImageView &destination,
  std::string &error
);

EffectiveRenderWindow intersectRenderWindow(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window
);

float halfToFloat(uint16_t value);
uint16_t floatToHalf(float value);

void copySourceToFloatStaging(
  const ImageView &source,
  const EffectiveRenderWindow &window,
  int32_t width,
  int32_t height,
  float *destination
);

void copyFloatStagingToDestination(
  const float *source,
  const MutableImageView &destination,
  const EffectiveRenderWindow &window,
  int32_t width,
  int32_t height
);

} // namespace spektrafilm
