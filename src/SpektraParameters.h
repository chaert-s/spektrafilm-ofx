#pragma once

#include <cstdint>

namespace spektrafilm {

enum class ProcessMode : int32_t {
  PrintSimulation = 0,
  ScanNegative = 1,
  ProcessNegative = 2,
};

//Outdated
enum class RenderOutputMode : int32_t {
  FinalPreview = 0,
  FilmDensityCmy = 1,
  FilmDensityCmyWithGrain = 2,
  FilmLogRaw = 3,
  PrintLogRaw = 4,
  PrintDensityCmy = 5,
};

enum class RgbToRawMethod : int32_t {
  Hanatos2025 = 0,
  Mallett2019 = 1,
  Hanatos2026 = 2,
};

enum class AutoExposureMethod : int32_t {
  CenterWeighted = 0,
  Median = 1,
};

enum class PushPullMode : int32_t {
  Standard = 0,
  Experimental = 1,
};

enum class PrintTimingMode : int32_t {
  FilteredEnlarger = 0,
  ApdPrinterDensity = 1,
};

enum class OutputRole : int32_t {
  DisplaySdr = 0,
  DisplayHdr = 1,
  Rcm = 2,
};

enum class HdrPreset : int32_t {
  Pq1000 = 0,
  Pq4000 = 1,
  Hlg1000 = 2,
  Custom = 3,
};

enum class HdrTransfer : int32_t {
  Pq = 0,
  Hlg = 1,
};

enum class HdrToneMapping : int32_t {
  SoftRolloff = 0,
  HardClip = 1,
};

enum class GrainModel : int32_t {
  Preview = 0,
  Production = 1,
  GrainSynthesis = 2,
};

enum class GpuRenderTilingMode : int32_t {
  LegacyFullFrame = 0,
  Tiled = 1,
};

enum class FilmFormat : int32_t {
  Standard8 = 0,
  Super8 = 1,
  Standard16 = 2,
  Super16 = 3,
  Standard35 = 4,
  Super35 = 5,
  Standard65 = 6,
  Imax70 = 7,
};

enum class DiffusionFilterFamily : int32_t {
  Glimmerglass = 0,
  BlackProMist = 1,
  ProMist = 2,
  CineBloom = 3,
};

enum class ColorSpace : int32_t {
  ArriLogC4 = 0,
  ArriLogC3Ei800 = 1,
  BmdFilmWideGamutGen5 = 2,
  DavinciIntermediateWideGamut = 3,
  RedLog3G10RedWideGamutRgb = 4,
  SonySLog3SGamut3 = 5,
  SonySLog3SGamut3Cine = 6,
  CanonLog2CinemaGamutD55 = 7,
  CanonLog3CinemaGamutD55 = 8,
  PanasonicVLogVGamut = 9,
  Aces2065_1 = 10,
  AcesCg = 11,
  AcesCct = 12,
  AcesCc = 13,
  LinearRec2020 = 14,
  LinearRec709 = 15,
  LinearP3D65 = 16,
  Srgb = 17,
  DisplayP3 = 18,
  ProPhotoRgb = 19,
  AdobeRgb1998 = 20,
  DciP3 = 21,
  P3D65Gamma22 = 22,
  P3D65Gamma26 = 23,
  Rec709Gamma22 = 24,
  Rec709Gamma24 = 25,
};

enum class ImageMemoryDomain : int32_t {
  Host = 0,
  CudaDevice = 1,
};

struct RenderParams {
  ProcessMode process = ProcessMode::PrintSimulation;
  bool scanNegativeInvert = false;
  RenderOutputMode renderOutput = RenderOutputMode::FinalPreview;
  RgbToRawMethod rgbToRawMethod = RgbToRawMethod::Hanatos2026;
  ColorSpace inputColorSpace = ColorSpace::DavinciIntermediateWideGamut;
  ColorSpace outputColorSpace = ColorSpace::Rec709Gamma24;
  OutputRole outputRole = OutputRole::DisplaySdr;
  HdrPreset hdrPreset = HdrPreset::Pq1000;
  HdrTransfer hdrTransfer = HdrTransfer::Pq;
  float hdrReferenceWhiteNits = 203.0f;
  float hdrPeakNits = 1000.0f;
  float hdrExposureEv = 0.0f;
  HdrToneMapping hdrToneMapping = HdrToneMapping::HardClip;
  bool colorAdaptation = false;
  bool colorAdaptationInputCompression = true;
  bool colorAdaptationCurveSmoothing = true;
  bool colorAdaptationOutputLightnessCompression = true;
  bool colorAdaptationOutputChromaCompression = true;
  int32_t film = 2;
  int32_t paper = 3;
  PrintTimingMode printTiming = PrintTimingMode::FilteredEnlarger;

  bool cameraUvFilterEnabled = false;
  float cameraUvCutNm = 410.0f;
  bool cameraIrFilterEnabled = false;
  float cameraIrCutNm = 675.0f;

  float filmExposureEv = 0.0f;
  bool autoExposure = false;
  AutoExposureMethod autoExposureMethod = AutoExposureMethod::CenterWeighted;
  float printExposureEv = 0.0f;
  PushPullMode filmPushPullMode = PushPullMode::Standard;
  float filmPushPullStops = 0.0f;
  float printPushPullStops = 0.0f;
  float negativeBleachBypassAmount = 0.0f;
  float negativeLeucoCyanCoupling = 1.0f;
  float printBleachBypassAmount = 0.0f;
  float filmGamma = 1.0f;
  float printGamma = 1.0f;
  float printShadowShape = 0.0f;
  float printHighlightShape = 0.0f;

  float filterC = 0.0f;
  float filterMShift = 0.0f;
  float filterYShift = 0.0f;
  float enlargerScale = 1.0f;
  float enlargerOffsetXPercent = 0.0f;
  float enlargerOffsetYPercent = 0.0f;
  float preflashExposure = 0.0f;
  float preflashMFilterShift = 0.0f;
  float preflashYFilterShift = 0.0f;
  float printerLightsR = 0.0f;
  float printerLightsG = 0.0f;
  float printerLightsB = 0.0f;
  bool printerLightsGang = false;
  bool printerLightCalibration = true;

  float dirCouplersAmount = 0.0f;
  float dirCouplersDiffusionUm = 20.0f;
  float dirCouplersDiffusionTailUm = 200.0f;
  float dirCouplersDiffusionTailWeight = 0.06f;
  float dirCouplersInhibitionSameLayer = 1.0f;
  float dirCouplersInhibitionInterlayer = 1.0f;
  float dirCouplersGammaSameLayerR = 0.336f;
  float dirCouplersGammaSameLayerG = 0.319f;
  float dirCouplersGammaSameLayerB = 0.273f;
  float dirCouplersGammaRToG = 0.353f;
  float dirCouplersGammaRToB = 0.302f;
  float dirCouplersGammaGToR = 0.154f;
  float dirCouplersGammaGToB = 0.353f;
  float dirCouplersGammaBToR = 0.168f;
  float dirCouplersGammaBToG = 0.226f;

  bool grainEnabled = false;
  GrainModel grainModel = GrainModel::Preview;
  FilmFormat filmFormat = FilmFormat::Standard35;
  float grainAmount = 1.0f;
  float grainSaturation = 1.0f;
  bool grainSublayersEnabled = true;
  int32_t grainSubLayerCount = 1;
  float grainParticleAreaUm2 = 0.1f;
  float grainParticleScaleR = 1.2f;
  float grainParticleScaleG = 1.0f;
  float grainParticleScaleB = 2.5f;
  float grainParticleScaleLayer0 = 6.0f;
  float grainParticleScaleLayer1 = 1.0f;
  float grainParticleScaleLayer2 = 0.4f;
  float grainDensityMinR = 0.04f;
  float grainDensityMinG = 0.05f;
  float grainDensityMinB = 0.06f;
  float grainUniformityR = 0.99f;
  float grainUniformityG = 0.97f;
  float grainUniformityB = 0.98f;
  float grainFinalBlurUm = 11.8f;
  float grainBlurDyeCloudsUm = 1.0f;
  float grainMicroStructureScale = 0.2f;
  float grainMicroStructureSigmaNm = 30.0f;
  uint32_t grainSeed = 1;
  bool grainAnimate = false;
  float grainSynthesisSize = 1.0f;
  float grainSynthesisAmount = 1.0f;
  float grainSynthesisSharpness = 1.0f;
  float grainSynthesisQuality = 1.0f;
  int32_t grainSynthesisSamples = 128;
  float grainSynthesisMeanRadiusUm = 0.25f;
  float grainSynthesisRadiusStdDevRatio = 0.0f;
  float grainSynthesisObservationSigmaUm = 1.0f;
  float grainSynthesisCellSizeRatio = 1.0f;
  float grainSynthesisMaxRadiusQuantile = 0.999f;
  float grainSynthesisCoverageEpsilon = 0.0001f;
  int32_t grainSynthesisMaxGrainsPerCell = 32;
  float grainSynthesisRadiusScaleR = 1.2f;
  float grainSynthesisRadiusScaleG = 1.0f;
  float grainSynthesisRadiusScaleB = 2.5f;
  float grainSynthesisLayerScale0 = 6.0f;
  float grainSynthesisLayerScale1 = 1.0f;
  float grainSynthesisLayerScale2 = 0.4f;
  bool grainSynthesisLayered = true;

  bool halationEnabled = false;
  float scatterAmount = 1.0f;
  float scatterScale = 1.0f;
  float halationAmount = 1.0f;
  float halationScale = 1.0f;
  float halationStrengthR = 0.05f;
  float halationStrengthG = 0.015f;
  float halationStrengthB = 0.0f;
  float halationFirstSigmaUmR = 65.0f;
  float halationFirstSigmaUmG = 65.0f;
  float halationFirstSigmaUmB = 65.0f;
  float halationBoostEv = 0.0f;
  float halationBoostRange = 0.3f;
  float halationProtectEv = 4.0f;

  bool cameraDiffusionEnabled = false;
  DiffusionFilterFamily cameraDiffusionFamily = DiffusionFilterFamily::BlackProMist;
  float cameraDiffusionStrength = 0.5f;
  float cameraDiffusionSpatialScale = 1.0f;
  float cameraDiffusionHaloWarmth = 0.0f;
  float cameraDiffusionCoreIntensity = 1.0f;
  float cameraDiffusionCoreSize = 1.0f;
  float cameraDiffusionHaloIntensity = 1.0f;
  float cameraDiffusionHaloSize = 1.0f;
  float cameraDiffusionBloomIntensity = 1.0f;
  float cameraDiffusionBloomSize = 1.0f;

  bool printDiffusionEnabled = false;
  DiffusionFilterFamily printDiffusionFamily = DiffusionFilterFamily::BlackProMist;
  float printDiffusionStrength = 0.5f;
  float printDiffusionSpatialScale = 1.0f;
  float printDiffusionHaloWarmth = 0.0f;
  float printDiffusionCoreIntensity = 1.0f;
  float printDiffusionCoreSize = 1.0f;
  float printDiffusionHaloIntensity = 1.0f;
  float printDiffusionHaloSize = 1.0f;
  float printDiffusionBloomIntensity = 1.0f;
  float printDiffusionBloomSize = 1.0f;

  bool scannerEnabled = false;
  bool scannerWhiteCorrection = false;
  bool scannerBlackCorrection = false;
  float scannerWhiteLevel = 0.98f;
  float scannerBlackLevel = 0.01f;
  float glarePercent = 0.03f;
  float glareRoughness = 0.7f;
  float glareBlur = 0.5f;
  float scannerMtf50LpMm = 60.0f;
  float scannerUnsharpRadiusUm = 5.0f;
  float scannerUnsharpAmount = 0.7f;

  GpuRenderTilingMode gpuRenderTiling = GpuRenderTilingMode::LegacyFullFrame;
};

enum ColorAdaptationFlag : uint32_t {
  kColorAdaptationInputCompression = 1u << 0u,
  kColorAdaptationCurveSmoothing = 1u << 1u,
  kColorAdaptationOutputLightnessCompression = 1u << 2u,
  kColorAdaptationOutputChromaCompression = 1u << 3u,
};

inline uint32_t colorAdaptationFlags(const RenderParams &params) {
  if (!params.colorAdaptation) {
    return 0u;
  }
  uint32_t flags = 0u;
  flags |= params.colorAdaptationInputCompression ? kColorAdaptationInputCompression : 0u;
  flags |= params.colorAdaptationCurveSmoothing ? kColorAdaptationCurveSmoothing : 0u;
  flags |= params.colorAdaptationOutputLightnessCompression ? kColorAdaptationOutputLightnessCompression : 0u;
  flags |= params.colorAdaptationOutputChromaCompression ? kColorAdaptationOutputChromaCompression : 0u;
  return flags;
}

struct ImageView {
  const void *data = nullptr;
  int32_t x1 = 0;
  int32_t y1 = 0;
  int32_t width = 0;
  int32_t height = 0;
  int32_t rowBytes = 0;
  int32_t components = 4;
  int32_t bytesPerComponent = 4;
  ImageMemoryDomain memoryDomain = ImageMemoryDomain::Host;
};

struct MutableImageView {
  void *data = nullptr;
  int32_t x1 = 0;
  int32_t y1 = 0;
  int32_t width = 0;
  int32_t height = 0;
  int32_t rowBytes = 0;
  int32_t components = 4;
  int32_t bytesPerComponent = 4;
  ImageMemoryDomain memoryDomain = ImageMemoryDomain::Host;
};

struct RenderWindow {
  int32_t x1 = 0;
  int32_t y1 = 0;
  int32_t x2 = 0;
  int32_t y2 = 0;
};

} // namespace spektrafilm
