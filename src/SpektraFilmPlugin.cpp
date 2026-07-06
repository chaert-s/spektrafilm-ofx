#include "SpektraParameters.h"
#include "SpektraProfileCurves.h"
#include "SpektraRenderer.h"
#if defined __APPLE__
#  include "SpektraMetalRenderer.h"
#endif
#include "SpektraTooltips.h"

#include "ofxImageEffect.h"
#include "ofxColour.h"
#include "ofxMemory.h"
#include "ofxMessage.h"
#include "ofxMultiThread.h"
#include "ofxParam.h"
#include "ofxGPURender.h"

#if defined __APPLE__
#  include <ApplicationServices/ApplicationServices.h>
#  include <dlfcn.h>
#elif defined _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#  include <shellapi.h>
#elif defined __linux__
#  include <dlfcn.h>
#  include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <vector>

#if defined __APPLE__
#  define SPEKTRA_EXPORT __attribute__((visibility("default")))
#elif defined _WIN32
#  define SPEKTRA_EXPORT __declspec(dllexport)
#elif defined __linux__
#  define SPEKTRA_EXPORT __attribute__((visibility("default")))
#else
#  define SPEKTRA_EXPORT
#endif

#ifndef kOfxBitDepthHalf
#  define kOfxBitDepthHalf "OfxBitDepthHalf"
#endif

#ifndef SPEKTRAFILM_VERSION_STRING
#  define SPEKTRAFILM_VERSION_STRING "0.2.1"
#endif

#ifndef SPEKTRAFILM_PLUGIN_IDENTIFIER
#  define SPEKTRAFILM_PLUGIN_IDENTIFIER "org.spektrafilm.dev"
#endif

#ifndef SPEKTRAFILM_PLUGIN_LABEL
#  define SPEKTRAFILM_PLUGIN_LABEL "spektrafilm dev"
#endif

#ifndef SPEKTRAFILM_PLUGIN_FLAVOR
#  define SPEKTRAFILM_PLUGIN_FLAVOR 2
#endif

#ifndef SPEKTRAFILM_OFX_METAL_GPU_BUFFERS
#  define SPEKTRAFILM_OFX_METAL_GPU_BUFFERS 0
#endif

namespace {

constexpr const char *kPluginIdentifier = SPEKTRAFILM_PLUGIN_IDENTIFIER;
constexpr const char *kPluginLabel = SPEKTRAFILM_PLUGIN_LABEL;
constexpr int kPluginVersionMajor = 0;
constexpr int kPluginVersionMinor = 2;

OfxHost *gHost = nullptr;
OfxImageEffectSuiteV1 *gEffectHost = nullptr;
OfxPropertySuiteV1 *gPropHost = nullptr;
OfxParameterSuiteV1 *gParamHost = nullptr;
OfxMessageSuiteV1 *gMessageHost = nullptr;
int gPluginImageAnchor = 0;
std::atomic<uint32_t> gHostRenderLogCount{0};

enum class PluginFlavor : int32_t {
  Flow = 0,
  Pro = 1,
  FilmDev = 2,
};

constexpr PluginFlavor kPluginFlavor = static_cast<PluginFlavor>(SPEKTRAFILM_PLUGIN_FLAVOR);

constexpr const char *svgIconFileForFlavor() {
  switch (kPluginFlavor) {
    case PluginFlavor::Flow:
      return "icons/spektrafilm_flow.svg";
    case PluginFlavor::Pro:
      return "icons/spektrafilm.svg";
    case PluginFlavor::FilmDev:
      return "icons/spektrafilm_dev.svg";
  }
  return "icons/spektrafilm_dev.svg";
}

constexpr const char *pngIconFileForFlavor() {
  switch (kPluginFlavor) {
    case PluginFlavor::Flow:
      return "icons/spektrafilm_flow.png";
    case PluginFlavor::Pro:
      return "icons/spektrafilm.png";
    case PluginFlavor::FilmDev:
      return "icons/spektrafilm_dev.png";
  }
  return "icons/spektrafilm_dev.png";
}

enum ParamTag : uint32_t {
  kParamTagNone = 0u,
  kParamTagFlow = 1u << 0u,
  kParamTagDevelopment = 1u << 1u,
};

struct ParamMetadata {
  const char *name;
  const char *parentGroup;
  uint32_t tags;
};

constexpr uint32_t flow() {
  return kParamTagFlow;
}

constexpr uint32_t development() {
  return kParamTagDevelopment;
}

constexpr uint32_t flowDevelopment() {
  return kParamTagFlow | kParamTagDevelopment;
}

inline constexpr ParamMetadata kParamMetadata[] = {
  {"process", "colorGroup", flow()},
  {"scanNegativeInvert", "colorGroup", flow()},
  {"inputColorSpace", "colorGroup", flow()},
  {"rcmInputColorSpace", "colorGroup", flow()},
  {"outputRole", "colorGroup", flow()},
  {"sdrOutputColorSpace", "colorGroup", flow()},
  {"sceneOutputColorSpace", "colorGroup", kParamTagDevelopment},
  {"hdrPreset", "colorGroup", flow()},
  {"hdrTransfer", "colorGroup", flow()},
  {"hdrReferenceWhiteNits", "colorGroup", flow()},
  {"hdrPeakNits", "colorGroup", flow()},
  {"hdrExposureEv", "colorGroup", flow()},
  {"hdrToneMapping", "colorGroup", flow()},
  {"colorAdaptation", "colorGroup", flow()},
  {"colorAdaptationInputCompression", "colorGroup", kParamTagNone},
  {"colorAdaptationCurveSmoothing", "colorGroup", kParamTagNone},
  {"colorAdaptationOutputLightnessCompression", "colorGroup", kParamTagNone},
  {"colorAdaptationOutputChromaCompression", "colorGroup", kParamTagNone},

  {"cameraUvFilterEnabled", "filteringGroup", kParamTagNone},
  {"cameraUvCutNm", "filteringGroup", kParamTagNone},
  {"cameraIrFilterEnabled", "filteringGroup", kParamTagNone},
  {"cameraIrCutNm", "filteringGroup", kParamTagNone},

  {"rgbToRawMethod", "filmGroup", flow()},
  {"film", "filmGroup", flow()},
  {"filmFormat", "filmGroup", flow()},
  {"filmPushPullMode", "filmGroup", flow()},
  {"filmPushPullStops", "filmGroup", flow()},
  {"negativeBleachBypassAmount", "filmGroup", flowDevelopment()},
  {"negativeLeucoCyanCoupling", "filmGroup", development()},
  {"filmExposureEv", "filmGroup", flow()},
  {"autoExposure", "filmGroup", kParamTagNone},
  {"autoExposureMethod", "filmGroup", kParamTagNone},
  {"filmGamma", "filmGroup", development()},

  {"paper", "printGroup", flow()},
  {"printTiming", "printGroup", flow()},
  {"printPushPullStops", "printGroup", flow()},
  {"printBleachBypassAmount", "printGroup", flowDevelopment()},
  {"printExposureEv", "printGroup", flow()},
  {"printGamma", "printGroup", development()},
  {"printShadowShape", "printGroup", flow()},
  {"printHighlightShape", "printGroup", flow()},
  {"filterC", "printGroup", flow()},
  {"filterMShift", "printGroup", flow()},
  {"filterYShift", "printGroup", flow()},
  {"preflashExposure", "printGroup", flow()},
  {"preflashMFilterShift", "printGroup", flow()},
  {"preflashYFilterShift", "printGroup", flow()},
  {"printerLightsGang", "printGroup", flow()},
  {"printerLightsGroup", "printGroup", flow()},
  {"printerLightR", "printGroup", flow()},
  {"printerLightG", "printGroup", flow()},
  {"printerLightB", "printGroup", flow()},
  {"printerLightCalibration", "printGroup", kParamTagNone},

  {"enlargerScale", "enlargerGroup", kParamTagNone},
  {"enlargerOffsetXPercent", "enlargerGroup", kParamTagNone},
  {"enlargerOffsetYPercent", "enlargerGroup", kParamTagNone},

  {"dirAmount", "couplerGroup", flow()},
  {"dirDiffusionUm", "couplerGroup", flow()},
  {"dirDiffusionTailUm", "couplerGroup", kParamTagNone},
  {"dirDiffusionTailWeight", "couplerGroup", kParamTagNone},
  {"dirInhibitionSameLayer", "couplerGroup", flow()},
  {"dirInhibitionInterlayer", "couplerGroup", flow()},
  {"dirGammaSameLayerRgb", "couplerGroup", kParamTagNone},
  {"dirGammaRToGb", "couplerGroup", kParamTagNone},
  {"dirGammaGToRb", "couplerGroup", kParamTagNone},
  {"dirGammaBToRg", "couplerGroup", kParamTagNone},
  {"dirCalibrateToStock", "couplerGroup", kParamTagNone},

  {"grainEnabled", "grainGroup", flow()},
  {"grainModel", "grainGroup", flow()},
  {"grainAmount", "grainGroup", flow()},
  {"grainSaturation", "grainGroup", flow()},
  {"grainSublayersEnabled", "grainGroup", kParamTagNone},
  {"grainSubLayerCount", "grainGroup", kParamTagNone},
  {"grainParticleAreaUm2", "grainGroup", flow()},
  {"grainParticleScale", "grainGroup", kParamTagNone},
  {"grainParticleScaleLayers", "grainGroup", kParamTagNone},
  {"grainDensityMin", "grainGroup", kParamTagNone},
  {"grainUniformity", "grainGroup", kParamTagNone},
  {"grainFinalBlurUm", "grainGroup", kParamTagNone},
  {"grainBlurDyeCloudsUm", "grainGroup", kParamTagNone},
  {"grainMicroStructure", "grainGroup", kParamTagNone},
  {"grainSeed", "grainGroup", kParamTagNone},
  {"grainAnimate", "grainGroup", kParamTagNone},
  {"grainSynthesisSize", "grainGroup", development()},
  {"grainSynthesisAmount", "grainGroup", development()},
  {"grainSynthesisSharpness", "grainGroup", development()},
  {"grainSynthesisQuality", "grainGroup", development()},

  {"grainSynthesisSamples", "grainSynthesisGroup", development()},
  {"grainSynthesisMeanRadiusUm", "grainSynthesisGroup", development()},
  {"grainSynthesisRadiusStdDevRatio", "grainSynthesisGroup", development()},
  {"grainSynthesisObservationSigmaUm", "grainSynthesisGroup", development()},
  {"grainSynthesisCellSizeRatio", "grainSynthesisGroup", development()},
  {"grainSynthesisMaxRadiusQuantile", "grainSynthesisGroup", development()},
  {"grainSynthesisCoverageEpsilon", "grainSynthesisGroup", development()},
  {"grainSynthesisMaxGrainsPerCell", "grainSynthesisGroup", development()},
  {"grainSynthesisRadiusScale", "grainSynthesisGroup", development()},
  {"grainSynthesisLayerScale", "grainSynthesisGroup", development()},
  {"grainSynthesisLayered", "grainSynthesisGroup", development()},

  {"halationEnabled", "halationGroup", flow()},
  {"scatterAmount", "halationGroup", kParamTagNone},
  {"scatterScale", "halationGroup", kParamTagNone},
  {"halationAmount", "halationGroup", flow()},
  {"halationScale", "halationGroup", flow()},
  {"halationBoostEv", "halationGroup", flow()},
  {"halationBoostRange", "halationGroup", kParamTagNone},
  {"halationProtectEv", "halationGroup", kParamTagNone},
  {"halationStrength", "halationGroup", flow()},

  {"cameraDiffusionEnabled", "diffusionGroup", flow()},
  {"cameraDiffusionFamily", "diffusionGroup", flow()},
  {"cameraDiffusionStrength", "diffusionGroup", flow()},
  {"cameraDiffusionSpatialScale", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionHaloWarmth", "diffusionGroup", flow()},
  {"cameraDiffusionCoreIntensity", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionCoreSize", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionHaloIntensity", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionHaloSize", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionBloomIntensity", "diffusionGroup", kParamTagNone},
  {"cameraDiffusionBloomSize", "diffusionGroup", kParamTagNone},
  {"printDiffusionEnabled", "diffusionGroup", flow()},
  {"printDiffusionFamily", "diffusionGroup", flow()},
  {"printDiffusionStrength", "diffusionGroup", flow()},
  {"printDiffusionSpatialScale", "diffusionGroup", kParamTagNone},
  {"printDiffusionHaloWarmth", "diffusionGroup", flow()},
  {"printDiffusionCoreIntensity", "diffusionGroup", kParamTagNone},
  {"printDiffusionCoreSize", "diffusionGroup", kParamTagNone},
  {"printDiffusionHaloIntensity", "diffusionGroup", kParamTagNone},
  {"printDiffusionHaloSize", "diffusionGroup", kParamTagNone},
  {"printDiffusionBloomIntensity", "diffusionGroup", kParamTagNone},
  {"printDiffusionBloomSize", "diffusionGroup", kParamTagNone},

  {"scannerEnabled", "scannerGroup", flow()},
  {"scannerWhiteCorrection", "scannerGroup", flow()},
  {"scannerBlackCorrection", "scannerGroup", flow()},
  {"scannerWhiteLevel", "scannerGroup", flow()},
  {"scannerBlackLevel", "scannerGroup", flow()},
  {"glarePercent", "scannerGroup", kParamTagNone},
  {"glareRoughness", "scannerGroup", kParamTagNone},
  {"glareBlur", "scannerGroup", kParamTagNone},
  {"scannerMtf50LpMm", "scannerGroup", kParamTagNone},
  {"scannerUnsharpRadiusUm", "scannerGroup", kParamTagNone},
  {"scannerUnsharpAmount", "scannerGroup", kParamTagNone},

  {"gpuRenderTiling", "manageGroup", kParamTagNone},

  {"infoVersion", "infoGroup", flow()},
  {"infoCreatedBy", "infoGroup", flow()},
  {"infoBasedOn", "infoGroup", flow()},
};

const ParamMetadata *metadataForParam(const char *name) {
  for (const ParamMetadata &metadata : kParamMetadata) {
    if (std::strcmp(metadata.name, name) == 0) {
      return &metadata;
    }
  }
  return nullptr;
}

bool parameterVisibleInFlavor(const ParamMetadata &metadata) {
  const bool flowTagged = (metadata.tags & kParamTagFlow) != 0u;
  const bool developmentTagged = (metadata.tags & kParamTagDevelopment) != 0u;
  if (kPluginFlavor == PluginFlavor::FilmDev) {
    return true;
  }
  if (kPluginFlavor == PluginFlavor::Pro) {
    return !developmentTagged;
  }
  return flowTagged && !developmentTagged;
}

bool shouldDefineParam(const char *name) {
  (void)name;
  return true;
}

bool groupVisibleInFlavor(const char *name) {
  if (std::strcmp(name, "manageGroup") == 0) {
    return true;
  }
  if (kPluginFlavor == PluginFlavor::FilmDev) {
    return true;
  }
  for (const ParamMetadata &metadata : kParamMetadata) {
    if (std::strcmp(metadata.parentGroup, name) == 0 && parameterVisibleInFlavor(metadata)) {
      return true;
    }
  }
  return false;
}

bool shouldDefineGroup(const char *name) {
  return true;
}

bool parameterVisibleInFlavor(const char *name) {
  const ParamMetadata *metadata = metadataForParam(name);
  return !metadata || parameterVisibleInFlavor(*metadata);
}

bool parameterHiddenInFlavor(const char *name) {
  return !parameterVisibleInFlavor(name);
}

constexpr bool flavorAllowsDevelopmentControls() {
  return kPluginFlavor == PluginFlavor::FilmDev;
}

int grainModelOptionCountForFlavor() {
  return flavorAllowsDevelopmentControls() ? 3 : 2;
}

constexpr int outputRoleOptionCountForFlavor() {
  return 3;
}

spektrafilm::OutputRole outputRoleForFlavor(int value) {
  switch (static_cast<spektrafilm::OutputRole>(value)) {
    case spektrafilm::OutputRole::DisplayHdr:
      return spektrafilm::OutputRole::DisplayHdr;
    case spektrafilm::OutputRole::Rcm:
      return spektrafilm::OutputRole::Rcm;
    case spektrafilm::OutputRole::DisplaySdr:
    default:
      return spektrafilm::OutputRole::DisplaySdr;
  }
}

enum class ParamValueKind : uint8_t {
  Int = 1,
  Bool = 2,
  Double = 3,
  Double2D = 4,
  Double3D = 5,
};

struct ParamDefault {
  const char *name;
  ParamValueKind kind;
  int intDefault;
  double doubleDefault[3];
};

constexpr ParamDefault intDefault(const char *name, int value) {
  return {name, ParamValueKind::Int, value, {0.0, 0.0, 0.0}};
}

constexpr ParamDefault boolDefault(const char *name, bool value) {
  return {name, ParamValueKind::Bool, value ? 1 : 0, {0.0, 0.0, 0.0}};
}

constexpr ParamDefault doubleDefault(const char *name, double value) {
  return {name, ParamValueKind::Double, 0, {value, 0.0, 0.0}};
}

constexpr ParamDefault double2DDefault(const char *name, double x, double y) {
  return {name, ParamValueKind::Double2D, 0, {x, y, 0.0}};
}

constexpr ParamDefault double3DDefault(const char *name, double x, double y, double z) {
  return {name, ParamValueKind::Double3D, 0, {x, y, z}};
}

constexpr const char *kGrainSeedParamName = "grainSeed";
constexpr int kGrainSeedMin = 0;
constexpr int kGrainSeedMax = 1000000;

inline constexpr ParamDefault kParamDefaults[] = {
  intDefault("process", 0),
  boolDefault("scanNegativeInvert", false),
  intDefault("inputColorSpace", 0),
  intDefault("rcmInputColorSpace", 0),
  intDefault("outputRole", 0),
  intDefault("sdrOutputColorSpace", 8),
  intDefault("sceneOutputColorSpace", 0),
  intDefault("hdrPreset", 0),
  intDefault("hdrTransfer", 0),
  doubleDefault("hdrReferenceWhiteNits", 203.0),
  doubleDefault("hdrPeakNits", 1000.0),
  doubleDefault("hdrExposureEv", 0.0),
  intDefault("hdrToneMapping", 1),
  boolDefault("colorAdaptation", false),
  boolDefault("colorAdaptationInputCompression", true),
  boolDefault("colorAdaptationCurveSmoothing", true),
  boolDefault("colorAdaptationOutputLightnessCompression", true),
  boolDefault("colorAdaptationOutputChromaCompression", true),

  boolDefault("cameraUvFilterEnabled", false),
  doubleDefault("cameraUvCutNm", 410.0),
  boolDefault("cameraIrFilterEnabled", false),
  doubleDefault("cameraIrCutNm", 675.0),

  intDefault("rgbToRawMethod", 0),
  intDefault("film", static_cast<int>(spektrafilm::kSpektraDefaultFilmIndex)),
  intDefault("filmPushPullMode", 0),
  doubleDefault("filmPushPullStops", 0.0),
  doubleDefault("negativeBleachBypassAmount", 0.0),
  doubleDefault("negativeLeucoCyanCoupling", 1.0),
  doubleDefault("filmExposureEv", 0.0),
  boolDefault("autoExposure", false),
  intDefault("autoExposureMethod", 0),
  doubleDefault("filmGamma", 1.0),

  intDefault("paper", static_cast<int>(spektrafilm::kSpektraDefaultPaperIndex)),
  intDefault("printTiming", 0),
  doubleDefault("printPushPullStops", 0.0),
  doubleDefault("printBleachBypassAmount", 0.0),
  doubleDefault("printExposureEv", 0.0),
  doubleDefault("printGamma", 1.0),
  doubleDefault("printShadowShape", 0.0),
  doubleDefault("printHighlightShape", 0.0),
  doubleDefault("filterC", 0.0),
  doubleDefault("filterMShift", 0.0),
  doubleDefault("filterYShift", 0.0),
  doubleDefault("preflashExposure", 0.0),
  doubleDefault("preflashMFilterShift", 0.0),
  doubleDefault("preflashYFilterShift", 0.0),
  boolDefault("printerLightsGang", false),
  boolDefault("printerLightsGroup", false),
  doubleDefault("printerLightR", 0.0),
  doubleDefault("printerLightG", 0.0),
  doubleDefault("printerLightB", 0.0),
  boolDefault("printerLightCalibration", true),

  doubleDefault("enlargerScale", 1.0),
  doubleDefault("enlargerOffsetXPercent", 0.0),
  doubleDefault("enlargerOffsetYPercent", 0.0),

  doubleDefault("dirAmount", 0.0),
  doubleDefault("dirDiffusionUm", 20.0),
  doubleDefault("dirDiffusionTailUm", 200.0),
  doubleDefault("dirDiffusionTailWeight", 0.06),
  doubleDefault("dirInhibitionSameLayer", 1.0),
  doubleDefault("dirInhibitionInterlayer", 1.0),
  double3DDefault("dirGammaSameLayerRgb", 0.336, 0.319, 0.273),
  double2DDefault("dirGammaRToGb", 0.353, 0.302),
  double2DDefault("dirGammaGToRb", 0.154, 0.353),
  double2DDefault("dirGammaBToRg", 0.168, 0.226),
  boolDefault("dirUsesStockCalibration", true),

  boolDefault("grainEnabled", false),
  intDefault("grainModel", 0),
  intDefault("filmFormat", 4),
  doubleDefault("grainAmount", 1.0),
  doubleDefault("grainSaturation", 1.0),
  boolDefault("grainSublayersEnabled", true),
  intDefault("grainSubLayerCount", 1),
  doubleDefault("grainParticleAreaUm2", 0.1),
  double3DDefault("grainParticleScale", 1.2, 1.0, 2.5),
  double3DDefault("grainParticleScaleLayers", 6.0, 1.0, 0.4),
  double3DDefault("grainDensityMin", 0.04, 0.05, 0.06),
  double3DDefault("grainUniformity", 0.99, 0.97, 0.98),
  doubleDefault("grainFinalBlurUm", 7.17),
  doubleDefault("grainBlurDyeCloudsUm", 1.0),
  double2DDefault("grainMicroStructure", 0.2, 30.0),
  intDefault("grainSeed", 0),
  boolDefault("grainAnimate", true),
  doubleDefault("grainSynthesisSize", 1.0),
  doubleDefault("grainSynthesisAmount", 1.0),
  doubleDefault("grainSynthesisSharpness", 1.0),
  doubleDefault("grainSynthesisQuality", 1.0),

  intDefault("grainSynthesisSamples", 128),
  doubleDefault("grainSynthesisMeanRadiusUm", 0.25),
  doubleDefault("grainSynthesisRadiusStdDevRatio", 0.0),
  doubleDefault("grainSynthesisObservationSigmaUm", 1.0),
  doubleDefault("grainSynthesisCellSizeRatio", 1.0),
  doubleDefault("grainSynthesisMaxRadiusQuantile", 0.999),
  doubleDefault("grainSynthesisCoverageEpsilon", 0.0001),
  intDefault("grainSynthesisMaxGrainsPerCell", 32),
  double3DDefault("grainSynthesisRadiusScale", 1.2, 1.0, 2.5),
  double3DDefault("grainSynthesisLayerScale", 6.0, 1.0, 0.4),
  boolDefault("grainSynthesisLayered", true),

  boolDefault("halationEnabled", false),
  doubleDefault("scatterAmount", 1.0),
  doubleDefault("scatterScale", 1.0),
  doubleDefault("halationAmount", 1.0),
  doubleDefault("halationScale", 1.0),
  double3DDefault("halationStrength", 0.05, 0.015, 0.0),
  doubleDefault("halationBoostEv", 0.0),
  doubleDefault("halationBoostRange", 0.3),
  doubleDefault("halationProtectEv", 4.0),

  boolDefault("cameraDiffusionEnabled", false),
  intDefault("cameraDiffusionFamily", 1),
  doubleDefault("cameraDiffusionStrength", 0.5),
  doubleDefault("cameraDiffusionSpatialScale", 1.0),
  doubleDefault("cameraDiffusionHaloWarmth", 0.0),
  doubleDefault("cameraDiffusionCoreIntensity", 1.0),
  doubleDefault("cameraDiffusionCoreSize", 1.0),
  doubleDefault("cameraDiffusionHaloIntensity", 1.0),
  doubleDefault("cameraDiffusionHaloSize", 1.0),
  doubleDefault("cameraDiffusionBloomIntensity", 1.0),
  doubleDefault("cameraDiffusionBloomSize", 1.0),
  boolDefault("printDiffusionEnabled", false),
  intDefault("printDiffusionFamily", 1),
  doubleDefault("printDiffusionStrength", 0.5),
  doubleDefault("printDiffusionSpatialScale", 1.0),
  doubleDefault("printDiffusionHaloWarmth", 0.0),
  doubleDefault("printDiffusionCoreIntensity", 1.0),
  doubleDefault("printDiffusionCoreSize", 1.0),
  doubleDefault("printDiffusionHaloIntensity", 1.0),
  doubleDefault("printDiffusionHaloSize", 1.0),
  doubleDefault("printDiffusionBloomIntensity", 1.0),
  doubleDefault("printDiffusionBloomSize", 1.0),

  boolDefault("scannerEnabled", false),
  boolDefault("scannerWhiteCorrection", false),
  boolDefault("scannerBlackCorrection", false),
  doubleDefault("scannerWhiteLevel", 0.98),
  doubleDefault("scannerBlackLevel", 0.01),
  doubleDefault("glarePercent", 0.03),
  doubleDefault("glareRoughness", 0.7),
  doubleDefault("glareBlur", 0.5),
  doubleDefault("scannerMtf50LpMm", 60.0),
  doubleDefault("scannerUnsharpRadiusUm", 5.0),
  doubleDefault("scannerUnsharpAmount", 0.7),

  intDefault("gpuRenderTiling", 0),
};

struct StoredParamValue {
  ParamValueKind kind = ParamValueKind::Int;
  int intValue[3] = {0, 0, 0};
  double doubleValue[3] = {0.0, 0.0, 0.0};
};

using DefaultsSnapshot = std::unordered_map<std::string, StoredParamValue>;

const DefaultsSnapshot *gDescribeDefaults = nullptr;
int gDescriptorGrainSeedDefault = 0;

bool isGrainSeedParam(const char *name) {
  return name && std::strcmp(name, kGrainSeedParamName) == 0;
}

std::mt19937 makeGrainSeedRng() {
  uint32_t seedData[] = {
    static_cast<uint32_t>(std::time(nullptr)),
    static_cast<uint32_t>(std::clock()),
    0x9e3779b9u,
    0x85ebca6bu,
    0xc2b2ae35u,
    0x27d4eb2fu,
  };
  try {
    std::random_device device;
    for (uint32_t &word : seedData) {
      word ^= device();
    }
  } catch (...) {
  }
  std::seed_seq seed(std::begin(seedData), std::end(seedData));
  return std::mt19937(seed);
}

int randomGrainSeed() {
  static std::mutex mutex;
  static std::mt19937 rng = makeGrainSeedRng();
  std::lock_guard<std::mutex> lock(mutex);
  std::uniform_int_distribution<int> distribution(kGrainSeedMin, kGrainSeedMax);
  return distribution(rng);
}

int descriptorGrainSeedDefault() {
  gDescriptorGrainSeedDefault = randomGrainSeed();
  return gDescriptorGrainSeedDefault;
}

int paramComponentCount(ParamValueKind kind) {
  switch (kind) {
    case ParamValueKind::Double2D:
      return 2;
    case ParamValueKind::Double3D:
      return 3;
    case ParamValueKind::Int:
    case ParamValueKind::Bool:
    case ParamValueKind::Double:
    default:
      return 1;
  }
}

bool paramKindUsesDouble(ParamValueKind kind) {
  return kind == ParamValueKind::Double ||
    kind == ParamValueKind::Double2D ||
    kind == ParamValueKind::Double3D;
}

const ParamDefault *defaultForParam(const char *name) {
  for (const ParamDefault &entry : kParamDefaults) {
    if (std::strcmp(entry.name, name) == 0) {
      return &entry;
    }
  }
  return nullptr;
}

StoredParamValue factoryStoredValue(const ParamDefault &entry) {
  StoredParamValue value{};
  value.kind = entry.kind;
  if (paramKindUsesDouble(entry.kind)) {
    for (int i = 0; i < paramComponentCount(entry.kind); ++i) {
      value.doubleValue[i] = entry.doubleDefault[i];
    }
  } else if (std::strcmp(entry.name, kGrainSeedParamName) == 0) {
    value.intValue[0] = randomGrainSeed();
  } else {
    value.intValue[0] = entry.intDefault;
  }
  return value;
}

bool storedValueForDefault(const char *name, StoredParamValue &value) {
  if (isGrainSeedParam(name)) {
    return false;
  }
  if (!gDescribeDefaults) {
    return false;
  }
  const ParamDefault *entry = defaultForParam(name);
  if (!entry) {
    return false;
  }
  const auto found = gDescribeDefaults->find(name);
  if (found == gDescribeDefaults->end() || found->second.kind != entry->kind) {
    return false;
  }
  value = found->second;
  return true;
}

struct InstanceData {
  OfxImageClipHandle sourceClip = nullptr;
  OfxImageClipHandle outputClip = nullptr;

  OfxParamHandle filteringGroup = nullptr;
  OfxParamHandle enlargerGroup = nullptr;
  OfxParamHandle filmGroup = nullptr;
  OfxParamHandle printGroup = nullptr;
  OfxParamHandle couplerGroup = nullptr;
  OfxParamHandle grainGroup = nullptr;
  OfxParamHandle grainSynthesisGroup = nullptr;
  OfxParamHandle halationGroup = nullptr;
  OfxParamHandle process = nullptr;
  OfxParamHandle scanNegativeInvert = nullptr;
  OfxParamHandle rgbToRawMethod = nullptr;
  OfxParamHandle inputColorSpace = nullptr;
  OfxParamHandle rcmInputColorSpace = nullptr;
  OfxParamHandle outputRole = nullptr;
  OfxParamHandle sdrOutputColorSpace = nullptr;
  OfxParamHandle sceneOutputColorSpace = nullptr;
  OfxParamHandle hdrPreset = nullptr;
  OfxParamHandle hdrTransfer = nullptr;
  OfxParamHandle hdrReferenceWhiteNits = nullptr;
  OfxParamHandle hdrPeakNits = nullptr;
  OfxParamHandle hdrExposureEv = nullptr;
  OfxParamHandle hdrToneMapping = nullptr;
  OfxParamHandle colorAdaptation = nullptr;
  OfxParamHandle colorAdaptationInputCompression = nullptr;
  OfxParamHandle colorAdaptationCurveSmoothing = nullptr;
  OfxParamHandle colorAdaptationOutputLightnessCompression = nullptr;
  OfxParamHandle colorAdaptationOutputChromaCompression = nullptr;
  OfxParamHandle cameraUvFilterEnabled = nullptr;
  OfxParamHandle cameraUvCutNm = nullptr;
  OfxParamHandle cameraIrFilterEnabled = nullptr;
  OfxParamHandle cameraIrCutNm = nullptr;
  OfxParamHandle film = nullptr;
  OfxParamHandle paper = nullptr;
  OfxParamHandle printTiming = nullptr;
  OfxParamHandle filmExposureEv = nullptr;
  OfxParamHandle autoExposure = nullptr;
  OfxParamHandle autoExposureMethod = nullptr;
  OfxParamHandle printExposureEv = nullptr;
  OfxParamHandle filmPushPullMode = nullptr;
  OfxParamHandle filmPushPullStops = nullptr;
  OfxParamHandle printPushPullStops = nullptr;
  OfxParamHandle negativeBleachBypassAmount = nullptr;
  OfxParamHandle negativeLeucoCyanCoupling = nullptr;
  OfxParamHandle printBleachBypassAmount = nullptr;
  OfxParamHandle filmGamma = nullptr;
  OfxParamHandle printGamma = nullptr;
  OfxParamHandle printShadowShape = nullptr;
  OfxParamHandle printHighlightShape = nullptr;
  OfxParamHandle filterC = nullptr;
  OfxParamHandle filterMShift = nullptr;
  OfxParamHandle filterYShift = nullptr;
  OfxParamHandle enlargerScale = nullptr;
  OfxParamHandle enlargerOffsetXPercent = nullptr;
  OfxParamHandle enlargerOffsetYPercent = nullptr;
  OfxParamHandle preflashExposure = nullptr;
  OfxParamHandle preflashMFilterShift = nullptr;
  OfxParamHandle preflashYFilterShift = nullptr;
  OfxParamHandle printerLightR = nullptr;
  OfxParamHandle printerLightG = nullptr;
  OfxParamHandle printerLightB = nullptr;
  OfxParamHandle printerLightsGang = nullptr;
  OfxParamHandle printerLightsGroup = nullptr;
  OfxParamHandle printerLightCalibration = nullptr;
  OfxParamHandle dirAmount = nullptr;
  OfxParamHandle dirDiffusionUm = nullptr;
  OfxParamHandle dirDiffusionTailUm = nullptr;
  OfxParamHandle dirDiffusionTailWeight = nullptr;
  OfxParamHandle dirInhibitionSameLayer = nullptr;
  OfxParamHandle dirInhibitionInterlayer = nullptr;
  OfxParamHandle dirGammaSameLayerRgb = nullptr;
  OfxParamHandle dirGammaRToGb = nullptr;
  OfxParamHandle dirGammaGToRb = nullptr;
  OfxParamHandle dirGammaBToRg = nullptr;
  OfxParamHandle dirCalibrateToStock = nullptr;
  OfxParamHandle dirUsesStockCalibration = nullptr;
  OfxParamHandle grainEnabled = nullptr;
  OfxParamHandle grainModel = nullptr;
  OfxParamHandle filmFormat = nullptr;
  OfxParamHandle grainAmount = nullptr;
  OfxParamHandle grainSaturation = nullptr;
  OfxParamHandle grainSublayersEnabled = nullptr;
  OfxParamHandle grainSubLayerCount = nullptr;
  OfxParamHandle grainParticleAreaUm2 = nullptr;
  OfxParamHandle grainParticleScale = nullptr;
  OfxParamHandle grainParticleScaleLayers = nullptr;
  OfxParamHandle grainDensityMin = nullptr;
  OfxParamHandle grainUniformity = nullptr;
  OfxParamHandle grainFinalBlurUm = nullptr;
  OfxParamHandle grainBlurDyeCloudsUm = nullptr;
  OfxParamHandle grainMicroStructure = nullptr;
  OfxParamHandle grainSeed = nullptr;
  OfxParamHandle grainAnimate = nullptr;
  OfxParamHandle grainSynthesisSize = nullptr;
  OfxParamHandle grainSynthesisAmount = nullptr;
  OfxParamHandle grainSynthesisSharpness = nullptr;
  OfxParamHandle grainSynthesisQuality = nullptr;
  OfxParamHandle grainSynthesisSamples = nullptr;
  OfxParamHandle grainSynthesisMeanRadiusUm = nullptr;
  OfxParamHandle grainSynthesisRadiusStdDevRatio = nullptr;
  OfxParamHandle grainSynthesisObservationSigmaUm = nullptr;
  OfxParamHandle grainSynthesisCellSizeRatio = nullptr;
  OfxParamHandle grainSynthesisMaxRadiusQuantile = nullptr;
  OfxParamHandle grainSynthesisCoverageEpsilon = nullptr;
  OfxParamHandle grainSynthesisMaxGrainsPerCell = nullptr;
  OfxParamHandle grainSynthesisRadiusScale = nullptr;
  OfxParamHandle grainSynthesisLayerScale = nullptr;
  OfxParamHandle grainSynthesisLayered = nullptr;
  OfxParamHandle halationEnabled = nullptr;
  OfxParamHandle scatterAmount = nullptr;
  OfxParamHandle scatterScale = nullptr;
  OfxParamHandle halationAmount = nullptr;
  OfxParamHandle halationScale = nullptr;
  OfxParamHandle halationStrength = nullptr;
  OfxParamHandle halationBoostEv = nullptr;
  OfxParamHandle halationBoostRange = nullptr;
  OfxParamHandle halationProtectEv = nullptr;
  OfxParamHandle cameraDiffusionEnabled = nullptr;
  OfxParamHandle cameraDiffusionFamily = nullptr;
  OfxParamHandle cameraDiffusionStrength = nullptr;
  OfxParamHandle cameraDiffusionSpatialScale = nullptr;
  OfxParamHandle cameraDiffusionHaloWarmth = nullptr;
  OfxParamHandle cameraDiffusionCoreIntensity = nullptr;
  OfxParamHandle cameraDiffusionCoreSize = nullptr;
  OfxParamHandle cameraDiffusionHaloIntensity = nullptr;
  OfxParamHandle cameraDiffusionHaloSize = nullptr;
  OfxParamHandle cameraDiffusionBloomIntensity = nullptr;
  OfxParamHandle cameraDiffusionBloomSize = nullptr;
  OfxParamHandle printDiffusionEnabled = nullptr;
  OfxParamHandle printDiffusionFamily = nullptr;
  OfxParamHandle printDiffusionStrength = nullptr;
  OfxParamHandle printDiffusionSpatialScale = nullptr;
  OfxParamHandle printDiffusionHaloWarmth = nullptr;
  OfxParamHandle printDiffusionCoreIntensity = nullptr;
  OfxParamHandle printDiffusionCoreSize = nullptr;
  OfxParamHandle printDiffusionHaloIntensity = nullptr;
  OfxParamHandle printDiffusionHaloSize = nullptr;
  OfxParamHandle printDiffusionBloomIntensity = nullptr;
  OfxParamHandle printDiffusionBloomSize = nullptr;
  OfxParamHandle scannerGroup = nullptr;
  OfxParamHandle scannerEnabled = nullptr;
  OfxParamHandle scannerWhiteCorrection = nullptr;
  OfxParamHandle scannerBlackCorrection = nullptr;
  OfxParamHandle scannerWhiteLevel = nullptr;
  OfxParamHandle scannerBlackLevel = nullptr;
  OfxParamHandle glarePercent = nullptr;
  OfxParamHandle glareRoughness = nullptr;
  OfxParamHandle glareBlur = nullptr;
  OfxParamHandle scannerMtf50LpMm = nullptr;
  OfxParamHandle scannerUnsharpRadiusUm = nullptr;
  OfxParamHandle scannerUnsharpAmount = nullptr;
  OfxParamHandle gpuRenderTiling = nullptr;
  OfxParamHandle lutSize = nullptr;
  OfxParamHandle lutDestination = nullptr;
  OfxParamHandle lutIdentifier = nullptr;
  OfxParamHandle exportLut = nullptr;
  OfxParamHandle presetName = nullptr;
  OfxParamHandle presetSelection = nullptr;
  OfxParamHandle savePreset = nullptr;
  OfxParamHandle loadPreset = nullptr;

  double lastPrinterLights[3] = {0.0, 0.0, 0.0};
  bool lastPrinterLightsInitialized = false;
  bool syncingPrinterLights = false;
  bool syncingDirCalibration = false;
  bool metalGpuBufferProbeLogged = false;

  std::mutex rendererMutex;
  std::unique_ptr<spektrafilm::Renderer> renderer;
};

InstanceData *getInstanceData(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffectHost->getPropertySet(effect, &props);
  InstanceData *data = nullptr;
  gPropHost->propGetPointer(props, kOfxPropInstanceData, 0, reinterpret_cast<void **>(&data));
  return data;
}

spektrafilm::Renderer *ensureRenderer(InstanceData *data) {
  if (!data) {
    return nullptr;
  }
  if (!data->renderer) {
    data->renderer = spektrafilm::createNativeRenderer();
  }
  return data->renderer.get();
}

void releaseInstanceRendererResources(InstanceData *data, bool resetRenderer);

int mapPixelDepth(const char *depth) {
  if (!depth) {
    return 0;
  }
  if (std::strcmp(depth, kOfxBitDepthHalf) == 0) {
    return 16;
  }
  if (std::strcmp(depth, kOfxBitDepthFloat) == 0) {
    return 32;
  }
  return 0;
}

int componentsForString(const char *components) {
  if (!components) {
    return 0;
  }
  if (std::strcmp(components, kOfxImageComponentRGBA) == 0) {
    return 4;
  }
  if (std::strcmp(components, kOfxImageComponentRGB) == 0) {
    return 3;
  }
  if (std::strcmp(components, kOfxImageComponentAlpha) == 0) {
    return 1;
  }
  return 0;
}

// host render notes: lives near image fetch so CUDA/CPU pointer state is easy to audit
std::filesystem::path hostDebugLogPath() {
#if defined _WIN32
  char tempPath[MAX_PATH]{};
  const DWORD length = GetTempPathA(MAX_PATH, tempPath);
  if (length > 0 && length < MAX_PATH) {
    return std::filesystem::path(tempPath) / "SpektraFilmOFX.log";
  }
#endif
  return std::filesystem::temp_directory_path() / "SpektraFilmOFX.log";
}

bool hostDebugLogVerbose() {
  const char *value = std::getenv("SPEKTRAFILM_DEBUG_LOG");
  if (!value || !*value) {
    return false;
  }
  std::string text(value);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  return text != "0" && text != "off" && text != "false" && text != "no";
}

void appendHostLog(const std::string &message, bool important = false) {
  if (!important && !hostDebugLogVerbose()) {
    return;
  }
  try {
    std::ofstream out(hostDebugLogPath(), std::ios::app);
    if (!out) {
      return;
    }
    std::time_t now = std::time(nullptr);
    std::tm localTime{};
#if defined _WIN32
    localtime_s(&localTime, &now);
#else
    localtime_r(&localTime, &now);
#endif
    out << std::put_time(&localTime, "%Y-%m-%d %H:%M:%S")
        << " [" << kPluginLabel << "] " << message << "\n";
  } catch (...) {
  }
}

const char *statusName(OfxStatus status) {
  switch (status) {
    case kOfxStatOK:
      return "OK";
    case kOfxStatFailed:
      return "Failed";
    case kOfxStatErrFatal:
      return "Fatal";
    case kOfxStatErrUnknown:
      return "Unknown";
    case kOfxStatErrFormat:
      return "Format";
    case kOfxStatErrMemory:
      return "Memory";
    default:
      return "Other";
  }
}

std::string imageViewSummary(const spektrafilm::ImageView &view) {
  std::ostringstream out;
  out << "bounds=(" << view.x1 << "," << view.y1 << " "
      << (view.x1 + view.width) << "," << (view.y1 + view.height) << ")"
      << " size=" << view.width << "x" << view.height
      << " rowBytes=" << view.rowBytes
      << " components=" << view.components
      << " bytesPerComponent=" << view.bytesPerComponent
      << " memoryDomain="
      << (view.memoryDomain == spektrafilm::ImageMemoryDomain::CudaDevice ? "cuda-device" : "host")
      << " data=" << (view.data ? "yes" : "no");
  return out.str();
}

std::string imageViewSummary(const spektrafilm::MutableImageView &view) {
  spektrafilm::ImageView immutable{};
  immutable.data = view.data;
  immutable.x1 = view.x1;
  immutable.y1 = view.y1;
  immutable.width = view.width;
  immutable.height = view.height;
  immutable.rowBytes = view.rowBytes;
  immutable.components = view.components;
  immutable.bytesPerComponent = view.bytesPerComponent;
  immutable.memoryDomain = view.memoryDomain;
  return imageViewSummary(immutable);
}

std::string renderWindowSummary(const spektrafilm::RenderWindow &window) {
  std::ostringstream out;
  out << "(" << window.x1 << "," << window.y1 << " "
      << window.x2 << "," << window.y2 << ")";
  return out.str();
}

std::string rendererDiagnosticsSummary(const spektrafilm::RendererDiagnostics &diagnostics) {
  std::ostringstream out;
  out << "backend=" << diagnostics.backendName
      << " device=\"" << diagnostics.deviceName << "\""
      << " passes=" << diagnostics.passCount
      << " commandMs=" << diagnostics.commandBufferMs
      << " sourceCopyMs=" << diagnostics.sourceCopyMs
      << " outputCopyMs=" << diagnostics.outputCopyMs
      << " cudaH2DMs=" << diagnostics.cudaHostToDeviceMs
      << " cudaD2HMs=" << diagnostics.cudaDeviceToHostMs
      << " cudaKernelMs=" << diagnostics.cudaKernelMs
      << " uploadBytes=" << diagnostics.uploadBytes
      << " cudaPinnedBytes=" << diagnostics.cudaPinnedStagingBytes
      << " cudaDeviceBytes=" << diagnostics.cudaDeviceScratchBytes
      << " sourceNoCopy=" << (diagnostics.sourceNoCopy ? "yes" : "no")
      << " destinationNoCopy=" << (diagnostics.destinationNoCopy ? "yes" : "no")
      << " cudaMappedHostMemory=" << (diagnostics.cudaMappedHostMemory ? "yes" : "no")
      << " cudaTransferMode=" << diagnostics.cudaTransferMode
      << " passTiming=" << diagnostics.passTimingMode
      << " paths={halation:" << (diagnostics.halationPath ? "yes" : "no")
      << ",cameraDiffusion:" << (diagnostics.cameraDiffusionPath ? "yes" : "no")
      << ",printDiffusion:" << (diagnostics.printDiffusionPath ? "yes" : "no")
      << ",dir:" << (diagnostics.dirPath ? "yes" : "no")
      << ",grain:" << (diagnostics.productionGrainPath ? "yes" : "no")
      << ",grainSynthesis:" << (diagnostics.grainSynthesisPath ? "yes" : "no")
      << ",scanner:" << (diagnostics.finalPostProcessPath ? "yes" : "no") << "}";
  if (!diagnostics.passes.empty()) {
    out << " passNames=";
    const size_t count = std::min<size_t>(diagnostics.passes.size(), 16);
    for (size_t i = 0; i < count; ++i) {
      if (i != 0) {
        out << ",";
      }
      out << diagnostics.passes[i].name;
    }
    if (diagnostics.passes.size() > count) {
      out << ",...";
    }
  }
  if (!diagnostics.backendFallbackReason.empty()) {
    out << " fallback=\"" << diagnostics.backendFallbackReason << "\"";
  }
  return out.str();
}

OfxStatus fetchImageView(
  OfxImageClipHandle clip,
  OfxTime time,
  OfxPropertySetHandle *image,
  spektrafilm::ImageView &view
) {
  if (gEffectHost->clipGetImage(clip, time, nullptr, image) != kOfxStatOK || !*image) {
    return kOfxStatFailed;
  }

  OfxRectI bounds{};
  char *depth = nullptr;
  char *components = nullptr;
  void *data = nullptr;
  int rowBytes = 0;
  gPropHost->propGetIntN(*image, kOfxImagePropBounds, 4, &bounds.x1);
  gPropHost->propGetString(*image, kOfxImageEffectPropPixelDepth, 0, &depth);
  gPropHost->propGetString(*image, kOfxImageEffectPropComponents, 0, &components);
  gPropHost->propGetInt(*image, kOfxImagePropRowBytes, 0, &rowBytes);
  gPropHost->propGetPointer(*image, kOfxImagePropData, 0, &data);

  const int bitDepth = mapPixelDepth(depth);
  view.data = data;
  view.x1 = bounds.x1;
  view.y1 = bounds.y1;
  view.width = bounds.x2 - bounds.x1;
  view.height = bounds.y2 - bounds.y1;
  view.rowBytes = rowBytes;
  view.components = componentsForString(components);
  view.bytesPerComponent = bitDepth / 8;
  return data && view.components == 4 && view.bytesPerComponent > 0 ? kOfxStatOK : kOfxStatErrFormat;
}

OfxStatus fetchMutableImageView(
  OfxImageClipHandle clip,
  OfxTime time,
  OfxPropertySetHandle *image,
  spektrafilm::MutableImageView &view
) {
  spektrafilm::ImageView immutable{};
  OfxStatus status = fetchImageView(clip, time, image, immutable);
  if (status != kOfxStatOK) {
    return status;
  }
  view.data = const_cast<void *>(immutable.data);
  view.x1 = immutable.x1;
  view.y1 = immutable.y1;
  view.width = immutable.width;
  view.height = immutable.height;
  view.rowBytes = immutable.rowBytes;
  view.components = immutable.components;
  view.bytesPerComponent = immutable.bytesPerComponent;
  return kOfxStatOK;
}

void releaseImage(OfxPropertySetHandle image) {
  if (image) {
    gEffectHost->clipReleaseImage(image);
  }
}

#if SPEKTRAFILM_OFX_METAL_GPU_BUFFERS
struct MetalGpuImageProbe {
  OfxRectI bounds{};
  char *depth = nullptr;
  char *components = nullptr;
  void *buffer = nullptr;
  int rowBytes = 0;
};

spektrafilm::MetalBufferImageView makeMetalBufferImageView(const MetalGpuImageProbe &probe) {
  spektrafilm::MetalBufferImageView view{};
  view.buffer = probe.buffer;
  view.x1 = probe.bounds.x1;
  view.y1 = probe.bounds.y1;
  view.width = probe.bounds.x2 - probe.bounds.x1;
  view.height = probe.bounds.y2 - probe.bounds.y1;
  view.rowBytes = probe.rowBytes;
  view.components = componentsForString(probe.components);
  view.bytesPerComponent = mapPixelDepth(probe.depth) / 8;
  return view;
}

bool metalGpuBuffersEnabled(OfxPropertySetHandle inArgs) {
  int enabled = 0;
  return inArgs &&
         gPropHost->propGetInt(inArgs, kOfxImageEffectPropMetalEnabled, 0, &enabled) == kOfxStatOK &&
         enabled != 0;
}

void *metalCommandQueueFromArgs(OfxPropertySetHandle inArgs) {
  void *queue = nullptr;
  if (inArgs) {
    gPropHost->propGetPointer(inArgs, kOfxImageEffectPropMetalCommandQueue, 0, &queue);
  }
  return queue;
}

OfxStatus fetchMetalGpuImageProbe(
  OfxImageClipHandle clip,
  OfxTime time,
  OfxPropertySetHandle *image,
  MetalGpuImageProbe &probe
) {
  if (gEffectHost->clipGetImage(clip, time, nullptr, image) != kOfxStatOK || !*image) {
    return kOfxStatFailed;
  }

  gPropHost->propGetIntN(*image, kOfxImagePropBounds, 4, &probe.bounds.x1);
  gPropHost->propGetString(*image, kOfxImageEffectPropPixelDepth, 0, &probe.depth);
  gPropHost->propGetString(*image, kOfxImageEffectPropComponents, 0, &probe.components);
  gPropHost->propGetInt(*image, kOfxImagePropRowBytes, 0, &probe.rowBytes);
  gPropHost->propGetPointer(*image, kOfxImagePropData, 0, &probe.buffer);
  return probe.buffer ? kOfxStatOK : kOfxStatErrFormat;
}

OfxStatus probeMetalGpuBufferRender(
  OfxImageEffectHandle effect,
  InstanceData *data,
  OfxTime time,
  OfxPropertySetHandle inArgs,
  const spektrafilm::RenderWindow &window,
  const spektrafilm::RenderParams &params
) {
  OfxPropertySetHandle sourceImage = nullptr;
  OfxPropertySetHandle outputImage = nullptr;
  MetalGpuImageProbe source{};
  MetalGpuImageProbe output{};
  OfxStatus status = kOfxStatOK;

  try {
    status = fetchMetalGpuImageProbe(data->sourceClip, time, &sourceImage, source);
    if (status != kOfxStatOK) {
      throw status;
    }
    status = fetchMetalGpuImageProbe(data->outputClip, time, &outputImage, output);
    if (status != kOfxStatOK) {
      throw status;
    }

    if (!data->metalGpuBufferProbeLogged && gMessageHost) {
      data->metalGpuBufferProbeLogged = true;
      gMessageHost->message(
        effect,
        kOfxMessageLog,
        "spektrafilmMetalGpuBufferProbe",
        "OFX Metal GPU buffers enabled by host. queue=%p sourceBuffer=%p sourceBounds=[%d,%d,%d,%d] sourceRowBytes=%d sourceDepth=%s sourceComponents=%s outputBuffer=%p outputBounds=[%d,%d,%d,%d] outputRowBytes=%d outputDepth=%s outputComponents=%s",
        metalCommandQueueFromArgs(inArgs),
        source.buffer,
        source.bounds.x1,
        source.bounds.y1,
        source.bounds.x2,
        source.bounds.y2,
        source.rowBytes,
        source.depth ? source.depth : "",
        source.components ? source.components : "",
        output.buffer,
        output.bounds.x1,
        output.bounds.y1,
        output.bounds.x2,
        output.bounds.y2,
        output.rowBytes,
        output.depth ? output.depth : "",
        output.components ? output.components : ""
      );
    }

    auto *metalRenderer = dynamic_cast<spektrafilm::MetalRenderer *>(data->renderer.get());
    if (!metalRenderer) {
      status = static_cast<OfxStatus>(kOfxStatGPURenderFailed);
    } else if (!metalRenderer->renderMetalBuffers(
                 makeMetalBufferImageView(source),
                 makeMetalBufferImageView(output),
                 window,
                 params,
                 time,
                 metalCommandQueueFromArgs(inArgs))) {
      if (gMessageHost) {
        gMessageHost->message(
          effect,
          kOfxMessageLog,
          "spektrafilmMetalGpuBufferFallback",
          "OFX Metal GPU buffer render fell back to CPU: %s",
          metalRenderer->lastError().c_str()
        );
      }
      status = static_cast<OfxStatus>(kOfxStatGPURenderFailed);
    }
  } catch (OfxStatus caught) {
    status = caught;
  } catch (...) {
    status = kOfxStatErrUnknown;
  }

  releaseImage(sourceImage);
  releaseImage(outputImage);
  if (status != kOfxStatOK) {
    return status;
  }

  return kOfxStatOK;
}
#endif

double getDoubleAtTime(OfxParamHandle handle, OfxTime time, double fallback = 0.0) {
  if (!handle) {
    return fallback;
  }
  double value = fallback;
  gParamHost->paramGetValueAtTime(handle, time, &value);
  return value;
}

int getIntAtTime(OfxParamHandle handle, OfxTime time, int fallback = 0) {
  if (!handle) {
    return fallback;
  }
  int value = fallback;
  gParamHost->paramGetValueAtTime(handle, time, &value);
  return value;
}

bool getBoolAtTime(OfxParamHandle handle, OfxTime time, bool fallback = false) {
  return getIntAtTime(handle, time, fallback ? 1 : 0) != 0;
}

bool getBoolValue(OfxParamHandle handle, bool fallback = false) {
  if (!handle) {
    return fallback;
  }
  int value = fallback ? 1 : 0;
  gParamHost->paramGetValue(handle, &value);
  return value != 0;
}

int getIntValue(OfxParamHandle handle, int fallback = 0) {
  if (!handle) {
    return fallback;
  }
  int value = fallback;
  gParamHost->paramGetValue(handle, &value);
  return value;
}

void setParamSecret(OfxParamHandle handle, bool secret) {
  if (!handle || !gParamHost || !gPropHost) {
    return;
  }
  OfxPropertySetHandle props = nullptr;
  if (gParamHost->paramGetPropertySet(handle, &props) != kOfxStatOK || !props) {
    return;
  }
  gPropHost->propSetInt(props, kOfxParamPropSecret, 0, secret ? 1 : 0);
  gPropHost->propSetInt(props, kOfxParamPropEnabled, 0, secret ? 0 : 1);
}

void setParamEnabled(OfxParamHandle handle, bool enabled) {
  if (!handle || !gParamHost || !gPropHost) {
    return;
  }
  OfxPropertySetHandle props = nullptr;
  if (gParamHost->paramGetPropertySet(handle, &props) != kOfxStatOK || !props) {
    return;
  }
  gPropHost->propSetInt(props, kOfxParamPropEnabled, 0, enabled ? 1 : 0);
}

void setParamSecretForFlavor(OfxParamHandle handle, const char *name, bool secret) {
  setParamSecret(handle, secret || parameterHiddenInFlavor(name));
}

void syncConditionalParamVisibility(InstanceData *data) {
  if (!data) {
    return;
  }

  const spektrafilm::OutputRole outputRole = outputRoleForFlavor(
    getIntValue(data->outputRole, static_cast<int>(spektrafilm::OutputRole::DisplaySdr))
  );
  const bool sdrOutput = outputRole == spektrafilm::OutputRole::DisplaySdr;
  const bool hdrOutput = outputRole == spektrafilm::OutputRole::DisplayHdr;
  const bool rcmOutput = outputRole == spektrafilm::OutputRole::Rcm;
  const int processValue = getIntValue(data->process, 0);
  const bool scanNegative = processValue == static_cast<int>(spektrafilm::ProcessMode::ScanNegative);
  const bool processNegative = processValue == static_cast<int>(spektrafilm::ProcessMode::ProcessNegative);
  setParamSecretForFlavor(data->scanNegativeInvert, "scanNegativeInvert", !scanNegative);
  setParamSecretForFlavor(data->inputColorSpace, "inputColorSpace", rcmOutput);
  setParamSecretForFlavor(data->rcmInputColorSpace, "rcmInputColorSpace", !rcmOutput);
  setParamSecretForFlavor(data->sdrOutputColorSpace, "sdrOutputColorSpace", !sdrOutput);
  setParamSecretForFlavor(data->sceneOutputColorSpace, "sceneOutputColorSpace", true);
  setParamSecretForFlavor(data->hdrPreset, "hdrPreset", !hdrOutput);
  setParamSecretForFlavor(data->hdrTransfer, "hdrTransfer", !hdrOutput);
  setParamSecretForFlavor(data->hdrReferenceWhiteNits, "hdrReferenceWhiteNits", !hdrOutput);
  setParamSecretForFlavor(data->hdrPeakNits, "hdrPeakNits", !hdrOutput);
  setParamSecretForFlavor(data->hdrExposureEv, "hdrExposureEv", !hdrOutput);
  setParamSecretForFlavor(data->hdrToneMapping, "hdrToneMapping", !hdrOutput);

  const bool colorAdaptationEnabled = getBoolValue(data->colorAdaptation, false);
  setParamSecretForFlavor(data->colorAdaptationInputCompression, "colorAdaptationInputCompression", !colorAdaptationEnabled);
  setParamSecretForFlavor(data->colorAdaptationCurveSmoothing, "colorAdaptationCurveSmoothing", !colorAdaptationEnabled);
  setParamSecretForFlavor(data->colorAdaptationOutputLightnessCompression, "colorAdaptationOutputLightnessCompression", !colorAdaptationEnabled || rcmOutput);
  setParamSecretForFlavor(data->colorAdaptationOutputChromaCompression, "colorAdaptationOutputChromaCompression", !colorAdaptationEnabled || rcmOutput);

  setParamSecretForFlavor(data->filteringGroup, "filteringGroup", processNegative);
  setParamSecretForFlavor(data->cameraUvFilterEnabled, "cameraUvFilterEnabled", processNegative);
  setParamSecretForFlavor(data->cameraUvCutNm, "cameraUvCutNm", processNegative);
  setParamSecretForFlavor(data->cameraIrFilterEnabled, "cameraIrFilterEnabled", processNegative);
  setParamSecretForFlavor(data->cameraIrCutNm, "cameraIrCutNm", processNegative);

  const bool printStageHidden = scanNegative;
  setParamSecretForFlavor(data->enlargerGroup, "enlargerGroup", processNegative || printStageHidden);
  setParamSecretForFlavor(data->enlargerScale, "enlargerScale", processNegative || printStageHidden);
  setParamSecretForFlavor(data->enlargerOffsetXPercent, "enlargerOffsetXPercent", processNegative || printStageHidden);
  setParamSecretForFlavor(data->enlargerOffsetYPercent, "enlargerOffsetYPercent", processNegative || printStageHidden);

  setParamSecretForFlavor(data->filmGroup, "filmGroup", processNegative);
  setParamSecretForFlavor(data->rgbToRawMethod, "rgbToRawMethod", processNegative);
  setParamSecretForFlavor(data->film, "film", processNegative);
  setParamSecretForFlavor(data->filmFormat, "filmFormat", processNegative);
  setParamSecretForFlavor(data->filmExposureEv, "filmExposureEv", processNegative);
  setParamSecretForFlavor(data->autoExposure, "autoExposure", processNegative);
  setParamSecretForFlavor(data->autoExposureMethod, "autoExposureMethod", processNegative);
  setParamSecretForFlavor(data->filmPushPullMode, "filmPushPullMode", processNegative);
  setParamSecretForFlavor(data->filmPushPullStops, "filmPushPullStops", processNegative);
  setParamSecretForFlavor(data->negativeBleachBypassAmount, "negativeBleachBypassAmount", processNegative);
  setParamSecretForFlavor(data->negativeLeucoCyanCoupling, "negativeLeucoCyanCoupling", processNegative);
  setParamSecretForFlavor(data->filmGamma, "filmGamma", processNegative);

  setParamSecretForFlavor(data->printGroup, "printGroup", printStageHidden);
  setParamSecretForFlavor(data->paper, "paper", printStageHidden);
  setParamSecretForFlavor(data->printTiming, "printTiming", printStageHidden);
  setParamSecretForFlavor(data->printPushPullStops, "printPushPullStops", printStageHidden);
  setParamSecretForFlavor(data->printBleachBypassAmount, "printBleachBypassAmount", printStageHidden);
  setParamSecretForFlavor(data->printExposureEv, "printExposureEv", printStageHidden);
  setParamSecretForFlavor(data->printGamma, "printGamma", printStageHidden);
  setParamSecretForFlavor(data->printShadowShape, "printShadowShape", printStageHidden);
  setParamSecretForFlavor(data->printHighlightShape, "printHighlightShape", printStageHidden);

  setParamSecretForFlavor(data->couplerGroup, "couplerGroup", processNegative);
  setParamSecretForFlavor(data->dirAmount, "dirCouplersAmount", processNegative);
  setParamSecretForFlavor(data->dirDiffusionUm, "dirCouplersDiffusionUm", processNegative);
  setParamSecretForFlavor(data->dirDiffusionTailUm, "dirCouplersDiffusionTailUm", processNegative);
  setParamSecretForFlavor(data->dirDiffusionTailWeight, "dirCouplersDiffusionTailWeight", processNegative);
  setParamSecretForFlavor(data->dirInhibitionSameLayer, "dirCouplersInhibitionSameLayer", processNegative);
  setParamSecretForFlavor(data->dirInhibitionInterlayer, "dirCouplersInhibitionInterlayer", processNegative);
  setParamSecretForFlavor(data->dirGammaSameLayerRgb, "dirGammaSameLayerRgb", processNegative);
  setParamSecretForFlavor(data->dirGammaRToGb, "dirGammaRToGb", processNegative);
  setParamSecretForFlavor(data->dirGammaGToRb, "dirGammaGToRb", processNegative);
  setParamSecretForFlavor(data->dirGammaBToRg, "dirGammaBToRg", processNegative);
  setParamSecretForFlavor(data->dirCalibrateToStock, "dirCalibrateToStock", processNegative);
  setParamSecretForFlavor(data->dirUsesStockCalibration, "dirUsesStockCalibration", true);

  setParamSecretForFlavor(data->grainGroup, "grainGroup", processNegative);
  setParamSecretForFlavor(data->grainEnabled, "grainEnabled", processNegative);
  setParamSecretForFlavor(data->grainModel, "grainModel", processNegative);
  setParamSecretForFlavor(data->grainAmount, "grainAmount", processNegative);
  setParamSecretForFlavor(data->grainSaturation, "grainSaturation", processNegative);
  setParamSecretForFlavor(data->grainSublayersEnabled, "grainSublayersEnabled", processNegative);
  setParamSecretForFlavor(data->grainSubLayerCount, "grainSubLayerCount", processNegative);
  setParamSecretForFlavor(data->grainParticleAreaUm2, "grainParticleAreaUm2", processNegative);
  setParamSecretForFlavor(data->grainParticleScale, "grainParticleScale", processNegative);
  setParamSecretForFlavor(data->grainParticleScaleLayers, "grainParticleScaleLayers", processNegative);
  setParamSecretForFlavor(data->grainDensityMin, "grainDensityMin", processNegative);
  setParamSecretForFlavor(data->grainUniformity, "grainUniformity", processNegative);
  setParamSecretForFlavor(data->grainFinalBlurUm, "grainFinalBlurUm", processNegative);
  setParamSecretForFlavor(data->grainBlurDyeCloudsUm, "grainBlurDyeCloudsUm", processNegative);
  setParamSecretForFlavor(data->grainMicroStructure, "grainMicroStructure", processNegative);
  setParamSecretForFlavor(data->grainSeed, "grainSeed", processNegative);
  setParamSecretForFlavor(data->grainAnimate, "grainAnimate", processNegative);

  setParamSecretForFlavor(data->grainSynthesisGroup, "grainSynthesisGroup", processNegative || !flavorAllowsDevelopmentControls());
  setParamSecretForFlavor(data->grainSynthesisSamples, "grainSynthesisSamples", processNegative);
  setParamSecretForFlavor(data->grainSynthesisMeanRadiusUm, "grainSynthesisMeanRadiusUm", processNegative);
  setParamSecretForFlavor(data->grainSynthesisRadiusStdDevRatio, "grainSynthesisRadiusStdDevRatio", processNegative);
  setParamSecretForFlavor(data->grainSynthesisObservationSigmaUm, "grainSynthesisObservationSigmaUm", processNegative);
  setParamSecretForFlavor(data->grainSynthesisCellSizeRatio, "grainSynthesisCellSizeRatio", processNegative);
  setParamSecretForFlavor(data->grainSynthesisMaxRadiusQuantile, "grainSynthesisMaxRadiusQuantile", processNegative);
  setParamSecretForFlavor(data->grainSynthesisCoverageEpsilon, "grainSynthesisCoverageEpsilon", processNegative);
  setParamSecretForFlavor(data->grainSynthesisMaxGrainsPerCell, "grainSynthesisMaxGrainsPerCell", processNegative);
  setParamSecretForFlavor(data->grainSynthesisRadiusScale, "grainSynthesisRadiusScale", processNegative);
  setParamSecretForFlavor(data->grainSynthesisLayerScale, "grainSynthesisLayerScale", processNegative);
  setParamSecretForFlavor(data->grainSynthesisLayered, "grainSynthesisLayered", processNegative);

  setParamSecretForFlavor(data->halationGroup, "halationGroup", processNegative);
  setParamSecretForFlavor(data->halationEnabled, "halationEnabled", processNegative);
  setParamSecretForFlavor(data->scatterAmount, "scatterAmount", processNegative);
  setParamSecretForFlavor(data->scatterScale, "scatterScale", processNegative);
  setParamSecretForFlavor(data->halationAmount, "halationAmount", processNegative);
  setParamSecretForFlavor(data->halationScale, "halationScale", processNegative);
  setParamSecretForFlavor(data->halationStrength, "halationStrength", processNegative);
  setParamSecretForFlavor(data->halationBoostEv, "halationBoostEv", processNegative);
  setParamSecretForFlavor(data->halationBoostRange, "halationBoostRange", processNegative);
  setParamSecretForFlavor(data->halationProtectEv, "halationProtectEv", processNegative);

  setParamSecretForFlavor(data->cameraDiffusionEnabled, "cameraDiffusionEnabled", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionFamily, "cameraDiffusionFamily", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionStrength, "cameraDiffusionStrength", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionSpatialScale, "cameraDiffusionSpatialScale", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionHaloWarmth, "cameraDiffusionHaloWarmth", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionCoreIntensity, "cameraDiffusionCoreIntensity", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionCoreSize, "cameraDiffusionCoreSize", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionHaloIntensity, "cameraDiffusionHaloIntensity", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionHaloSize, "cameraDiffusionHaloSize", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionBloomIntensity, "cameraDiffusionBloomIntensity", processNegative);
  setParamSecretForFlavor(data->cameraDiffusionBloomSize, "cameraDiffusionBloomSize", processNegative);

  setParamSecretForFlavor(data->printDiffusionEnabled, "printDiffusionEnabled", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionFamily, "printDiffusionFamily", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionStrength, "printDiffusionStrength", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionSpatialScale, "printDiffusionSpatialScale", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionHaloWarmth, "printDiffusionHaloWarmth", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionCoreIntensity, "printDiffusionCoreIntensity", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionCoreSize, "printDiffusionCoreSize", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionHaloIntensity, "printDiffusionHaloIntensity", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionHaloSize, "printDiffusionHaloSize", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionBloomIntensity, "printDiffusionBloomIntensity", printStageHidden);
  setParamSecretForFlavor(data->printDiffusionBloomSize, "printDiffusionBloomSize", printStageHidden);

  const bool scannerHidden = rcmOutput;
  const bool printScanGlareHidden = scannerHidden || scanNegative;
  setParamSecretForFlavor(data->scannerGroup, "scannerGroup", scannerHidden);
  setParamSecretForFlavor(data->scannerEnabled, "scannerEnabled", scannerHidden);
  setParamSecretForFlavor(data->scannerWhiteCorrection, "scannerWhiteCorrection", scannerHidden);
  setParamSecretForFlavor(data->scannerBlackCorrection, "scannerBlackCorrection", scannerHidden);
  setParamSecretForFlavor(data->scannerWhiteLevel, "scannerWhiteLevel", scannerHidden);
  setParamSecretForFlavor(data->scannerBlackLevel, "scannerBlackLevel", scannerHidden);
  setParamSecretForFlavor(data->glarePercent, "glarePercent", printScanGlareHidden);
  setParamSecretForFlavor(data->glareRoughness, "glareRoughness", printScanGlareHidden);
  setParamSecretForFlavor(data->glareBlur, "glareBlur", printScanGlareHidden);
  setParamSecretForFlavor(data->scannerMtf50LpMm, "scannerMtf50LpMm", scannerHidden);
  setParamSecretForFlavor(data->scannerUnsharpRadiusUm, "scannerUnsharpRadiusUm", scannerHidden);
  setParamSecretForFlavor(data->scannerUnsharpAmount, "scannerUnsharpAmount", scannerHidden);

  const bool synthesisModel = !processNegative && getIntValue(data->grainModel, static_cast<int>(spektrafilm::GrainModel::Preview)) ==
    static_cast<int>(spektrafilm::GrainModel::GrainSynthesis);
  setParamSecretForFlavor(data->grainSynthesisSize, "grainSynthesisSize", !synthesisModel);
  setParamSecretForFlavor(data->grainSynthesisAmount, "grainSynthesisAmount", !synthesisModel);
  setParamSecretForFlavor(data->grainSynthesisSharpness, "grainSynthesisSharpness", !synthesisModel);
  setParamSecretForFlavor(data->grainSynthesisQuality, "grainSynthesisQuality", !synthesisModel);

  const bool sublayersEnabled = getBoolValue(data->grainSublayersEnabled, true);
  setParamSecretForFlavor(data->grainSubLayerCount, "grainSubLayerCount", processNegative || sublayersEnabled);

  const bool apdPrintTiming = spektrafilm::kSpektraAcademyPrinterDensityEnabled &&
    getIntValue(data->printTiming, static_cast<int>(spektrafilm::PrintTimingMode::FilteredEnlarger)) ==
    static_cast<int>(spektrafilm::PrintTimingMode::ApdPrinterDensity);
  setParamSecretForFlavor(data->filterC, "filterC", scanNegative || apdPrintTiming);
  setParamSecretForFlavor(data->filterMShift, "filterMShift", scanNegative || apdPrintTiming);
  setParamSecretForFlavor(data->filterYShift, "filterYShift", scanNegative || apdPrintTiming);
  setParamSecretForFlavor(data->preflashExposure, "preflashExposure", scanNegative || processNegative);
  setParamSecretForFlavor(data->preflashMFilterShift, "preflashMFilterShift", scanNegative || processNegative || apdPrintTiming);
  setParamSecretForFlavor(data->preflashYFilterShift, "preflashYFilterShift", scanNegative || processNegative || apdPrintTiming);
  setParamSecretForFlavor(data->printerLightsGang, "printerLightsGang", scanNegative || !apdPrintTiming);
  setParamSecretForFlavor(data->printerLightsGroup, "printerLightsGroup", scanNegative || !apdPrintTiming);
  setParamSecretForFlavor(data->printerLightR, "printerLightR", scanNegative || !apdPrintTiming);
  setParamSecretForFlavor(data->printerLightG, "printerLightG", scanNegative || !apdPrintTiming);
  setParamSecretForFlavor(data->printerLightB, "printerLightB", scanNegative || !apdPrintTiming);
  setParamSecretForFlavor(data->printerLightCalibration, "printerLightCalibration", scanNegative || !apdPrintTiming);

  const bool lutExportAllowed = outputRole == spektrafilm::OutputRole::DisplaySdr;
  setParamEnabled(data->exportLut, lutExportAllowed);
}

bool readCurrentPrinterLights(InstanceData *data, double (&current)[3]) {
  if (!data || !data->printerLightR || !data->printerLightG || !data->printerLightB) {
    return false;
  }
  gParamHost->paramGetValue(data->printerLightR, &current[0]);
  gParamHost->paramGetValue(data->printerLightG, &current[1]);
  gParamHost->paramGetValue(data->printerLightB, &current[2]);
  return true;
}

void rememberCurrentPrinterLights(InstanceData *data, const double (&current)[3]) {
  if (!data) {
    return;
  }
  data->lastPrinterLights[0] = current[0];
  data->lastPrinterLights[1] = current[1];
  data->lastPrinterLights[2] = current[2];
  data->lastPrinterLightsInitialized = true;
}

void rememberCurrentPrinterLights(InstanceData *data) {
  double current[3] = {0.0, 0.0, 0.0};
  if (readCurrentPrinterLights(data, current)) {
    rememberCurrentPrinterLights(data, current);
  }
}

const spektrafilm::ProfileCurveSet *currentFilmCurves(InstanceData *data) {
  int filmIndex = static_cast<int>(spektrafilm::kSpektraDefaultFilmIndex);
  if (data && data->film) {
    gParamHost->paramGetValue(data->film, &filmIndex);
  }
  const spektrafilm::ProfileCurveSet *curves = spektrafilm::filmProfileCurves(filmIndex);
  return curves ? curves : spektrafilm::filmProfileCurves(static_cast<int32_t>(spektrafilm::kSpektraDefaultFilmIndex));
}

bool dirUsesStockCalibration(InstanceData *data) {
  return data && data->dirUsesStockCalibration &&
    getBoolValue(data->dirUsesStockCalibration, true);
}

bool applyDirStockCalibration(InstanceData *data, bool resetMultipliers) {
  if (!data || !data->dirGammaSameLayerRgb || !data->dirGammaRToGb || !data->dirGammaGToRb ||
      !data->dirGammaBToRg || !data->dirUsesStockCalibration) {
    return false;
  }
  const spektrafilm::ProfileCurveSet *curves = currentFilmCurves(data);
  if (!curves || !curves->dirGammaSameLayerRgb || !curves->dirGammaRToGb ||
      !curves->dirGammaGToRb || !curves->dirGammaBToRg) {
    return false;
  }

  data->syncingDirCalibration = true;
  if (resetMultipliers) {
    if (data->dirInhibitionSameLayer) {
      gParamHost->paramSetValue(data->dirInhibitionSameLayer, 1.0);
    }
    if (data->dirInhibitionInterlayer) {
      gParamHost->paramSetValue(data->dirInhibitionInterlayer, 1.0);
    }
  }
  gParamHost->paramSetValue(
    data->dirGammaSameLayerRgb,
    curves->dirGammaSameLayerRgb[0],
    curves->dirGammaSameLayerRgb[1],
    curves->dirGammaSameLayerRgb[2]
  );
  gParamHost->paramSetValue(data->dirGammaRToGb, curves->dirGammaRToGb[0], curves->dirGammaRToGb[1]);
  gParamHost->paramSetValue(data->dirGammaGToRb, curves->dirGammaGToRb[0], curves->dirGammaGToRb[1]);
  gParamHost->paramSetValue(data->dirGammaBToRg, curves->dirGammaBToRg[0], curves->dirGammaBToRg[1]);
  gParamHost->paramSetValue(data->dirUsesStockCalibration, 1);
  data->syncingDirCalibration = false;
  return true;
}

struct HdrPresetValues {
  int transfer = 0;
  double referenceWhiteNits = 203.0;
  double peakNits = 1000.0;
  int toneMapping = 1;
};

HdrPresetValues hdrPresetValues(int preset) {
  switch (preset) {
    case 1:
      return {0, 203.0, 4000.0, 1};
    case 2:
      return {1, 203.0, 1000.0, 1};
    default:
      return {0, 203.0, 1000.0, 1};
  }
}

spektrafilm::RgbToRawMethod rgbToRawMethodFromChoice(int choice) {
  switch (choice) {
    case 1:
      return spektrafilm::RgbToRawMethod::Hanatos2025;
    case 2:
      return spektrafilm::RgbToRawMethod::Mallett2019;
    case 0:
    default:
      return spektrafilm::RgbToRawMethod::Hanatos2026;
  }
}

spektrafilm::RenderParams readParams(InstanceData *data, OfxTime time) {
  constexpr spektrafilm::ColorSpace kSdrOutputColorSpaces[] = {
    spektrafilm::ColorSpace::Srgb,
    spektrafilm::ColorSpace::DisplayP3,
    spektrafilm::ColorSpace::ProPhotoRgb,
    spektrafilm::ColorSpace::AdobeRgb1998,
    spektrafilm::ColorSpace::DciP3,
    spektrafilm::ColorSpace::P3D65Gamma22,
    spektrafilm::ColorSpace::P3D65Gamma26,
    spektrafilm::ColorSpace::Rec709Gamma22,
    spektrafilm::ColorSpace::Rec709Gamma24,
  };
  constexpr spektrafilm::ColorSpace kRcmInputColorSpaces[] = {
    spektrafilm::ColorSpace::DavinciIntermediateWideGamut,
    spektrafilm::ColorSpace::BmdFilmWideGamutGen5,
    spektrafilm::ColorSpace::Aces2065_1,
    spektrafilm::ColorSpace::AcesCg,
    spektrafilm::ColorSpace::AcesCct,
    spektrafilm::ColorSpace::AcesCc,
    spektrafilm::ColorSpace::LinearRec2020,
    spektrafilm::ColorSpace::LinearRec709,
    spektrafilm::ColorSpace::LinearP3D65,
    spektrafilm::ColorSpace::Rec709Gamma24,
    spektrafilm::ColorSpace::Rec709Gamma22,
    spektrafilm::ColorSpace::Srgb,
    spektrafilm::ColorSpace::DisplayP3,
    spektrafilm::ColorSpace::P3D65Gamma22,
    spektrafilm::ColorSpace::P3D65Gamma26,
    spektrafilm::ColorSpace::DciP3,
  };

  spektrafilm::RenderParams params{};
  switch (getIntAtTime(data->process, time, 0)) {
    case 1:
      params.process = spektrafilm::ProcessMode::ScanNegative;
      break;
    case 2:
      params.process = spektrafilm::ProcessMode::ProcessNegative;
      break;
    case 0:
    default:
      params.process = spektrafilm::ProcessMode::PrintSimulation;
      break;
  }
  params.rgbToRawMethod = rgbToRawMethodFromChoice(getIntAtTime(data->rgbToRawMethod, time, 0));
  if (params.process == spektrafilm::ProcessMode::ProcessNegative) {
    params.rgbToRawMethod = spektrafilm::RgbToRawMethod::Hanatos2026;
  }
  params.inputColorSpace = static_cast<spektrafilm::ColorSpace>(getIntAtTime(data->inputColorSpace, time, 0));
  params.outputRole = outputRoleForFlavor(getIntAtTime(data->outputRole, time, 0));
  if (params.outputRole == spektrafilm::OutputRole::Rcm) {
    const int rcmIndex = std::clamp(
      getIntAtTime(data->rcmInputColorSpace, time, 0),
      0,
      static_cast<int>(std::size(kRcmInputColorSpaces) - 1u)
    );
    params.inputColorSpace = kRcmInputColorSpaces[rcmIndex];
    params.outputColorSpace = kRcmInputColorSpaces[rcmIndex];
  } else {
    const int sdrIndex = std::clamp(
      getIntAtTime(data->sdrOutputColorSpace, time, 8),
      0,
      static_cast<int>(std::size(kSdrOutputColorSpaces) - 1u)
    );
    params.outputColorSpace = kSdrOutputColorSpaces[sdrIndex];
  }
  params.scanNegativeInvert = getBoolAtTime(data->scanNegativeInvert, time, false);
  params.hdrPreset = static_cast<spektrafilm::HdrPreset>(getIntAtTime(data->hdrPreset, time, 0));
  params.hdrTransfer = static_cast<spektrafilm::HdrTransfer>(getIntAtTime(data->hdrTransfer, time, 0));
  params.hdrReferenceWhiteNits = static_cast<float>(getDoubleAtTime(data->hdrReferenceWhiteNits, time, 203.0));
  params.hdrPeakNits = static_cast<float>(getDoubleAtTime(data->hdrPeakNits, time, 1000.0));
  params.hdrExposureEv = static_cast<float>(getDoubleAtTime(data->hdrExposureEv, time, 0.0));
  params.hdrToneMapping = static_cast<spektrafilm::HdrToneMapping>(getIntAtTime(data->hdrToneMapping, time, 1));
  params.colorAdaptation = getBoolAtTime(data->colorAdaptation, time, false);
  if (params.colorAdaptation && kPluginFlavor == PluginFlavor::Flow) {
    params.colorAdaptationInputCompression = true;
    params.colorAdaptationCurveSmoothing = true;
    params.colorAdaptationOutputLightnessCompression = true;
    params.colorAdaptationOutputChromaCompression = true;
  } else {
    params.colorAdaptationInputCompression = getBoolAtTime(data->colorAdaptationInputCompression, time, true);
    params.colorAdaptationCurveSmoothing = getBoolAtTime(data->colorAdaptationCurveSmoothing, time, true);
    params.colorAdaptationOutputLightnessCompression = getBoolAtTime(data->colorAdaptationOutputLightnessCompression, time, true);
    params.colorAdaptationOutputChromaCompression = getBoolAtTime(data->colorAdaptationOutputChromaCompression, time, true);
  }
  params.cameraUvFilterEnabled = getBoolAtTime(data->cameraUvFilterEnabled, time, false);
  params.cameraUvCutNm = static_cast<float>(getDoubleAtTime(data->cameraUvCutNm, time, 410.0));
  params.cameraIrFilterEnabled = getBoolAtTime(data->cameraIrFilterEnabled, time, false);
  params.cameraIrCutNm = static_cast<float>(getDoubleAtTime(data->cameraIrCutNm, time, 675.0));
  params.film = getIntAtTime(data->film, time, static_cast<int>(spektrafilm::kSpektraDefaultFilmIndex));
  params.paper = getIntAtTime(data->paper, time, static_cast<int>(spektrafilm::kSpektraDefaultPaperIndex));
  const int printTiming = spektrafilm::kSpektraAcademyPrinterDensityEnabled
    ? getIntAtTime(data->printTiming, time, 0)
    : static_cast<int>(spektrafilm::PrintTimingMode::FilteredEnlarger);
  params.printTiming = static_cast<spektrafilm::PrintTimingMode>(printTiming);
  params.filmExposureEv = static_cast<float>(getDoubleAtTime(data->filmExposureEv, time, 0.0));
  params.autoExposure = getBoolAtTime(data->autoExposure, time, false);
  params.autoExposureMethod = static_cast<spektrafilm::AutoExposureMethod>(getIntAtTime(data->autoExposureMethod, time, 0));
  params.printExposureEv = static_cast<float>(getDoubleAtTime(data->printExposureEv, time, 0.0));
  params.filmPushPullMode = static_cast<spektrafilm::PushPullMode>(getIntAtTime(data->filmPushPullMode, time, 0));
  params.filmPushPullStops = static_cast<float>(getDoubleAtTime(data->filmPushPullStops, time, 0.0));
  params.printPushPullStops = static_cast<float>(getDoubleAtTime(data->printPushPullStops, time, 0.0));
  params.negativeBleachBypassAmount = static_cast<float>(getDoubleAtTime(data->negativeBleachBypassAmount, time, 0.0));
  params.negativeLeucoCyanCoupling = static_cast<float>(getDoubleAtTime(data->negativeLeucoCyanCoupling, time, 1.0));
  params.printBleachBypassAmount = static_cast<float>(getDoubleAtTime(data->printBleachBypassAmount, time, 0.0));
  params.filmGamma = static_cast<float>(getDoubleAtTime(data->filmGamma, time, 1.0));
  params.printGamma = static_cast<float>(getDoubleAtTime(data->printGamma, time, 1.0));
  params.printShadowShape = static_cast<float>(getDoubleAtTime(data->printShadowShape, time, 0.0));
  params.printHighlightShape = static_cast<float>(getDoubleAtTime(data->printHighlightShape, time, 0.0));
  params.filterC = static_cast<float>(getDoubleAtTime(data->filterC, time, 0.0));
  params.filterMShift = static_cast<float>(getDoubleAtTime(data->filterMShift, time, 0.0));
  params.filterYShift = static_cast<float>(getDoubleAtTime(data->filterYShift, time, 0.0));
  params.enlargerScale = static_cast<float>(getDoubleAtTime(data->enlargerScale, time, 1.0));
  params.enlargerOffsetXPercent = static_cast<float>(getDoubleAtTime(data->enlargerOffsetXPercent, time, 0.0));
  params.enlargerOffsetYPercent = static_cast<float>(getDoubleAtTime(data->enlargerOffsetYPercent, time, 0.0));
  params.preflashExposure = static_cast<float>(getDoubleAtTime(data->preflashExposure, time, 0.0));
  params.preflashMFilterShift = static_cast<float>(getDoubleAtTime(data->preflashMFilterShift, time, 0.0));
  params.preflashYFilterShift = static_cast<float>(getDoubleAtTime(data->preflashYFilterShift, time, 0.0));
  params.printerLightsR = static_cast<float>(getDoubleAtTime(data->printerLightR, time, 0.0));
  params.printerLightsG = static_cast<float>(getDoubleAtTime(data->printerLightG, time, 0.0));
  params.printerLightsB = static_cast<float>(getDoubleAtTime(data->printerLightB, time, 0.0));
  params.printerLightsGang = getBoolAtTime(data->printerLightsGang, time, false);
  params.printerLightCalibration = getBoolAtTime(data->printerLightCalibration, time, true);
  params.dirCouplersAmount = static_cast<float>(getDoubleAtTime(data->dirAmount, time, 0.0));
  params.dirCouplersDiffusionUm = static_cast<float>(getDoubleAtTime(data->dirDiffusionUm, time, 20.0));
  params.dirCouplersDiffusionTailUm = static_cast<float>(getDoubleAtTime(data->dirDiffusionTailUm, time, 200.0));
  params.dirCouplersDiffusionTailWeight = static_cast<float>(getDoubleAtTime(data->dirDiffusionTailWeight, time, 0.06));
  params.dirCouplersInhibitionSameLayer = static_cast<float>(getDoubleAtTime(data->dirInhibitionSameLayer, time, 1.0));
  params.dirCouplersInhibitionInterlayer = static_cast<float>(getDoubleAtTime(data->dirInhibitionInterlayer, time, 1.0));
  double dirGammaSameLayerRgb[3] = {0.336, 0.319, 0.273};
  if (data->dirGammaSameLayerRgb) {
    gParamHost->paramGetValueAtTime(
      data->dirGammaSameLayerRgb,
      time,
      &dirGammaSameLayerRgb[0],
      &dirGammaSameLayerRgb[1],
      &dirGammaSameLayerRgb[2]
    );
  }
  double dirGammaRToGb[2] = {0.353, 0.302};
  if (data->dirGammaRToGb) {
    gParamHost->paramGetValueAtTime(data->dirGammaRToGb, time, &dirGammaRToGb[0], &dirGammaRToGb[1]);
  }
  double dirGammaGToRb[2] = {0.154, 0.353};
  if (data->dirGammaGToRb) {
    gParamHost->paramGetValueAtTime(data->dirGammaGToRb, time, &dirGammaGToRb[0], &dirGammaGToRb[1]);
  }
  double dirGammaBToRg[2] = {0.168, 0.226};
  if (data->dirGammaBToRg) {
    gParamHost->paramGetValueAtTime(data->dirGammaBToRg, time, &dirGammaBToRg[0], &dirGammaBToRg[1]);
  }
  params.dirCouplersGammaSameLayerR = static_cast<float>(dirGammaSameLayerRgb[0]);
  params.dirCouplersGammaSameLayerG = static_cast<float>(dirGammaSameLayerRgb[1]);
  params.dirCouplersGammaSameLayerB = static_cast<float>(dirGammaSameLayerRgb[2]);
  params.dirCouplersGammaRToG = static_cast<float>(dirGammaRToGb[0]);
  params.dirCouplersGammaRToB = static_cast<float>(dirGammaRToGb[1]);
  params.dirCouplersGammaGToR = static_cast<float>(dirGammaGToRb[0]);
  params.dirCouplersGammaGToB = static_cast<float>(dirGammaGToRb[1]);
  params.dirCouplersGammaBToR = static_cast<float>(dirGammaBToRg[0]);
  params.dirCouplersGammaBToG = static_cast<float>(dirGammaBToRg[1]);
  params.grainEnabled = getBoolAtTime(data->grainEnabled, time, false);
  params.grainModel = static_cast<spektrafilm::GrainModel>(getIntAtTime(data->grainModel, time, 0));
  if (!flavorAllowsDevelopmentControls() && params.grainModel == spektrafilm::GrainModel::GrainSynthesis) {
    params.grainModel = spektrafilm::GrainModel::Preview;
  }
  params.filmFormat = static_cast<spektrafilm::FilmFormat>(getIntAtTime(data->filmFormat, time, 4));
  params.grainAmount = static_cast<float>(getDoubleAtTime(data->grainAmount, time, 1.0));
  params.grainSaturation = static_cast<float>(getDoubleAtTime(data->grainSaturation, time, 1.0));
  params.grainSublayersEnabled = getBoolAtTime(data->grainSublayersEnabled, time, true);
  params.grainSubLayerCount = getIntAtTime(data->grainSubLayerCount, time, 1);
  params.grainParticleAreaUm2 = static_cast<float>(getDoubleAtTime(data->grainParticleAreaUm2, time, 0.1));
  double grainParticleScale[3] = {1.2, 1.0, 2.5};
  if (data->grainParticleScale) {
    gParamHost->paramGetValueAtTime(data->grainParticleScale, time, &grainParticleScale[0], &grainParticleScale[1], &grainParticleScale[2]);
  }
  params.grainParticleScaleR = static_cast<float>(grainParticleScale[0]);
  params.grainParticleScaleG = static_cast<float>(grainParticleScale[1]);
  params.grainParticleScaleB = static_cast<float>(grainParticleScale[2]);
  double grainParticleScaleLayers[3] = {6.0, 1.0, 0.4};
  if (data->grainParticleScaleLayers) {
    gParamHost->paramGetValueAtTime(data->grainParticleScaleLayers, time, &grainParticleScaleLayers[0], &grainParticleScaleLayers[1], &grainParticleScaleLayers[2]);
  }
  params.grainParticleScaleLayer0 = static_cast<float>(grainParticleScaleLayers[0]);
  params.grainParticleScaleLayer1 = static_cast<float>(grainParticleScaleLayers[1]);
  params.grainParticleScaleLayer2 = static_cast<float>(grainParticleScaleLayers[2]);
  double grainDensityMin[3] = {0.04, 0.05, 0.06};
  if (data->grainDensityMin) {
    gParamHost->paramGetValueAtTime(data->grainDensityMin, time, &grainDensityMin[0], &grainDensityMin[1], &grainDensityMin[2]);
  }
  params.grainDensityMinR = static_cast<float>(grainDensityMin[0]);
  params.grainDensityMinG = static_cast<float>(grainDensityMin[1]);
  params.grainDensityMinB = static_cast<float>(grainDensityMin[2]);
  double grainUniformity[3] = {0.99, 0.97, 0.98};
  if (data->grainUniformity) {
    gParamHost->paramGetValueAtTime(data->grainUniformity, time, &grainUniformity[0], &grainUniformity[1], &grainUniformity[2]);
  }
  params.grainUniformityR = static_cast<float>(grainUniformity[0]);
  params.grainUniformityG = static_cast<float>(grainUniformity[1]);
  params.grainUniformityB = static_cast<float>(grainUniformity[2]);
  params.grainFinalBlurUm = static_cast<float>(getDoubleAtTime(data->grainFinalBlurUm, time, 7.17));
  params.grainBlurDyeCloudsUm = static_cast<float>(getDoubleAtTime(data->grainBlurDyeCloudsUm, time, 1.0));
  double microStructure[2] = {0.2, 30.0};
  if (data->grainMicroStructure) {
    gParamHost->paramGetValueAtTime(data->grainMicroStructure, time, &microStructure[0], &microStructure[1]);
  }
  params.grainMicroStructureScale = static_cast<float>(microStructure[0]);
  params.grainMicroStructureSigmaNm = static_cast<float>(microStructure[1]);
  params.grainSeed = static_cast<uint32_t>(getIntAtTime(data->grainSeed, time, 1));
  params.grainAnimate = getBoolAtTime(data->grainAnimate, time, false);
  params.grainSynthesisSize = static_cast<float>(getDoubleAtTime(data->grainSynthesisSize, time, 1.0));
  params.grainSynthesisAmount = static_cast<float>(getDoubleAtTime(data->grainSynthesisAmount, time, 1.0));
  params.grainSynthesisSharpness = static_cast<float>(getDoubleAtTime(data->grainSynthesisSharpness, time, 1.0));
  params.grainSynthesisQuality = static_cast<float>(getDoubleAtTime(data->grainSynthesisQuality, time, 1.0));
  params.grainSynthesisSamples = getIntAtTime(data->grainSynthesisSamples, time, 128);
  params.grainSynthesisMeanRadiusUm = static_cast<float>(getDoubleAtTime(data->grainSynthesisMeanRadiusUm, time, 0.25));
  params.grainSynthesisRadiusStdDevRatio = static_cast<float>(getDoubleAtTime(data->grainSynthesisRadiusStdDevRatio, time, 0.0));
  params.grainSynthesisObservationSigmaUm = static_cast<float>(getDoubleAtTime(data->grainSynthesisObservationSigmaUm, time, 1.0));
  params.grainSynthesisCellSizeRatio = static_cast<float>(getDoubleAtTime(data->grainSynthesisCellSizeRatio, time, 1.0));
  params.grainSynthesisMaxRadiusQuantile = static_cast<float>(getDoubleAtTime(data->grainSynthesisMaxRadiusQuantile, time, 0.999));
  params.grainSynthesisCoverageEpsilon = static_cast<float>(getDoubleAtTime(data->grainSynthesisCoverageEpsilon, time, 0.0001));
  params.grainSynthesisMaxGrainsPerCell = getIntAtTime(data->grainSynthesisMaxGrainsPerCell, time, 32);
  double grainSynthesisRadiusScale[3] = {1.2, 1.0, 2.5};
  if (data->grainSynthesisRadiusScale) {
    gParamHost->paramGetValueAtTime(data->grainSynthesisRadiusScale, time, &grainSynthesisRadiusScale[0], &grainSynthesisRadiusScale[1], &grainSynthesisRadiusScale[2]);
  }
  params.grainSynthesisRadiusScaleR = static_cast<float>(grainSynthesisRadiusScale[0]);
  params.grainSynthesisRadiusScaleG = static_cast<float>(grainSynthesisRadiusScale[1]);
  params.grainSynthesisRadiusScaleB = static_cast<float>(grainSynthesisRadiusScale[2]);
  double grainSynthesisLayerScale[3] = {6.0, 1.0, 0.4};
  if (data->grainSynthesisLayerScale) {
    gParamHost->paramGetValueAtTime(data->grainSynthesisLayerScale, time, &grainSynthesisLayerScale[0], &grainSynthesisLayerScale[1], &grainSynthesisLayerScale[2]);
  }
  params.grainSynthesisLayerScale0 = static_cast<float>(grainSynthesisLayerScale[0]);
  params.grainSynthesisLayerScale1 = static_cast<float>(grainSynthesisLayerScale[1]);
  params.grainSynthesisLayerScale2 = static_cast<float>(grainSynthesisLayerScale[2]);
  params.grainSynthesisLayered = getBoolAtTime(data->grainSynthesisLayered, time, true);
  params.halationEnabled = getBoolAtTime(data->halationEnabled, time, false);
  params.scatterAmount = static_cast<float>(getDoubleAtTime(data->scatterAmount, time, 1.0));
  params.scatterScale = static_cast<float>(getDoubleAtTime(data->scatterScale, time, 1.0));
  params.halationAmount = static_cast<float>(getDoubleAtTime(data->halationAmount, time, 1.0));
  params.halationScale = static_cast<float>(getDoubleAtTime(data->halationScale, time, 1.0));
  double strength[3] = {0.05, 0.015, 0.0};
  if (data->halationStrength) {
    gParamHost->paramGetValueAtTime(data->halationStrength, time, &strength[0], &strength[1], &strength[2]);
  }
  params.halationStrengthR = static_cast<float>(strength[0]);
  params.halationStrengthG = static_cast<float>(strength[1]);
  params.halationStrengthB = static_cast<float>(strength[2]);
  params.halationBoostEv = static_cast<float>(getDoubleAtTime(data->halationBoostEv, time, 0.0));
  params.halationBoostRange = static_cast<float>(getDoubleAtTime(data->halationBoostRange, time, 0.3));
  params.halationProtectEv = static_cast<float>(getDoubleAtTime(data->halationProtectEv, time, 4.0));
  params.cameraDiffusionEnabled = getBoolAtTime(data->cameraDiffusionEnabled, time, false);
  params.cameraDiffusionFamily = static_cast<spektrafilm::DiffusionFilterFamily>(getIntAtTime(data->cameraDiffusionFamily, time, 1));
  params.cameraDiffusionStrength = static_cast<float>(getDoubleAtTime(data->cameraDiffusionStrength, time, 0.5));
  params.cameraDiffusionSpatialScale = static_cast<float>(getDoubleAtTime(data->cameraDiffusionSpatialScale, time, 1.0));
  params.cameraDiffusionHaloWarmth = static_cast<float>(getDoubleAtTime(data->cameraDiffusionHaloWarmth, time, 0.0));
  params.cameraDiffusionCoreIntensity = static_cast<float>(getDoubleAtTime(data->cameraDiffusionCoreIntensity, time, 1.0));
  params.cameraDiffusionCoreSize = static_cast<float>(getDoubleAtTime(data->cameraDiffusionCoreSize, time, 1.0));
  params.cameraDiffusionHaloIntensity = static_cast<float>(getDoubleAtTime(data->cameraDiffusionHaloIntensity, time, 1.0));
  params.cameraDiffusionHaloSize = static_cast<float>(getDoubleAtTime(data->cameraDiffusionHaloSize, time, 1.0));
  params.cameraDiffusionBloomIntensity = static_cast<float>(getDoubleAtTime(data->cameraDiffusionBloomIntensity, time, 1.0));
  params.cameraDiffusionBloomSize = static_cast<float>(getDoubleAtTime(data->cameraDiffusionBloomSize, time, 1.0));
  params.printDiffusionEnabled = getBoolAtTime(data->printDiffusionEnabled, time, false);
  params.printDiffusionFamily = static_cast<spektrafilm::DiffusionFilterFamily>(getIntAtTime(data->printDiffusionFamily, time, 1));
  params.printDiffusionStrength = static_cast<float>(getDoubleAtTime(data->printDiffusionStrength, time, 0.5));
  params.printDiffusionSpatialScale = static_cast<float>(getDoubleAtTime(data->printDiffusionSpatialScale, time, 1.0));
  params.printDiffusionHaloWarmth = static_cast<float>(getDoubleAtTime(data->printDiffusionHaloWarmth, time, 0.0));
  params.printDiffusionCoreIntensity = static_cast<float>(getDoubleAtTime(data->printDiffusionCoreIntensity, time, 1.0));
  params.printDiffusionCoreSize = static_cast<float>(getDoubleAtTime(data->printDiffusionCoreSize, time, 1.0));
  params.printDiffusionHaloIntensity = static_cast<float>(getDoubleAtTime(data->printDiffusionHaloIntensity, time, 1.0));
  params.printDiffusionHaloSize = static_cast<float>(getDoubleAtTime(data->printDiffusionHaloSize, time, 1.0));
  params.printDiffusionBloomIntensity = static_cast<float>(getDoubleAtTime(data->printDiffusionBloomIntensity, time, 1.0));
  params.printDiffusionBloomSize = static_cast<float>(getDoubleAtTime(data->printDiffusionBloomSize, time, 1.0));
  params.scannerEnabled = getBoolAtTime(data->scannerEnabled, time, false);
  params.scannerWhiteCorrection = getBoolAtTime(data->scannerWhiteCorrection, time, false);
  params.scannerBlackCorrection = getBoolAtTime(data->scannerBlackCorrection, time, false);
  params.scannerWhiteLevel = static_cast<float>(getDoubleAtTime(data->scannerWhiteLevel, time, 0.98));
  params.scannerBlackLevel = static_cast<float>(getDoubleAtTime(data->scannerBlackLevel, time, 0.01));
  params.glarePercent = static_cast<float>(getDoubleAtTime(data->glarePercent, time, 0.03));
  params.glareRoughness = static_cast<float>(getDoubleAtTime(data->glareRoughness, time, 0.7));
  params.glareBlur = static_cast<float>(getDoubleAtTime(data->glareBlur, time, 0.5));
  params.scannerMtf50LpMm = static_cast<float>(getDoubleAtTime(data->scannerMtf50LpMm, time, 60.0));
  params.scannerUnsharpRadiusUm = static_cast<float>(getDoubleAtTime(data->scannerUnsharpRadiusUm, time, 5.0));
  params.scannerUnsharpAmount = static_cast<float>(getDoubleAtTime(data->scannerUnsharpAmount, time, 0.7));
  params.gpuRenderTiling = getIntAtTime(data->gpuRenderTiling, time, 0) == 1
    ? spektrafilm::GpuRenderTilingMode::Tiled
    : spektrafilm::GpuRenderTilingMode::LegacyFullFrame;
  if (params.process == spektrafilm::ProcessMode::ProcessNegative) {
    params.rgbToRawMethod = spektrafilm::RgbToRawMethod::Hanatos2026;
    params.film = static_cast<int32_t>(spektrafilm::kSpektraDefaultFilmIndex);
    params.filmFormat = spektrafilm::FilmFormat::Standard35;
    params.cameraUvFilterEnabled = false;
    params.cameraIrFilterEnabled = false;
    params.filmExposureEv = 0.0f;
    params.autoExposure = false;
    params.filmPushPullMode = spektrafilm::PushPullMode::Standard;
    params.filmPushPullStops = 0.0f;
    params.negativeBleachBypassAmount = 0.0f;
    params.negativeLeucoCyanCoupling = 1.0f;
    params.filmGamma = 1.0f;
    params.enlargerScale = 1.0f;
    params.enlargerOffsetXPercent = 0.0f;
    params.enlargerOffsetYPercent = 0.0f;
    params.dirCouplersAmount = 0.0f;
    params.grainEnabled = false;
    params.halationEnabled = false;
    params.cameraDiffusionEnabled = false;
  }
  return params;
}

bool validStoredKind(ParamValueKind kind) {
  switch (kind) {
    case ParamValueKind::Int:
    case ParamValueKind::Bool:
    case ParamValueKind::Double:
    case ParamValueKind::Double2D:
    case ParamValueKind::Double3D:
      return true;
  }
  return false;
}

std::string encodeDefaultsSnapshot(const DefaultsSnapshot &snapshot) {
  std::ostringstream out;
  out << "SPKDFLT2\n";
  for (const auto &item : snapshot) {
    const std::string &name = item.first;
    const StoredParamValue &value = item.second;
    const int components = paramComponentCount(value.kind);
    out << name << ' ' << static_cast<int>(value.kind) << ' ' << components;
    if (paramKindUsesDouble(value.kind)) {
      out << std::setprecision(std::numeric_limits<double>::max_digits10);
      for (int i = 0; i < components; ++i) {
        out << ' ' << value.doubleValue[i];
      }
    } else {
      for (int i = 0; i < components; ++i) {
        out << ' ' << value.intValue[i];
      }
    }
    out << '\n';
  }
  return out.str();
}

bool decodeDefaultsSnapshot(const std::string &text, DefaultsSnapshot &snapshot) {
  std::istringstream in(text);
  std::string line;
  if (!std::getline(in, line) || line != "SPKDFLT2") {
    return false;
  }
  DefaultsSnapshot decoded;
  while (std::getline(in, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream row(line);
    std::string name;
    int kindRaw = 0;
    int components = 0;
    if (!(row >> name >> kindRaw >> components)) {
      continue;
    }
    StoredParamValue value{};
    value.kind = static_cast<ParamValueKind>(kindRaw);
    if (!validStoredKind(value.kind) || components != paramComponentCount(value.kind)) {
      continue;
    }
    const ParamDefault *factory = defaultForParam(name.c_str());
    if (!factory || factory->kind != value.kind) {
      continue;
    }
    bool ok = true;
    if (paramKindUsesDouble(value.kind)) {
      for (int c = 0; c < components; ++c) {
        if (!(row >> value.doubleValue[c])) {
          ok = false;
          break;
        }
      }
    } else {
      for (int c = 0; c < components; ++c) {
        if (!(row >> value.intValue[c])) {
          ok = false;
          break;
        }
      }
    }
    if (!ok) {
      continue;
    }
    decoded[name] = value;
  }
  snapshot = std::move(decoded);
  return true;
}

std::filesystem::path userDefaultsPath() {
#if defined _WIN32
  const char *base = std::getenv("APPDATA");
  if (base && base[0]) {
    return std::filesystem::path(base) / "spektrafilm" / "ofx-defaults-v1.spkdefaults";
  }
  const char *home = std::getenv("USERPROFILE");
  return std::filesystem::path(home && home[0] ? home : ".") / "AppData" / "Roaming" / "spektrafilm" / "ofx-defaults-v1.spkdefaults";
#elif defined __APPLE__
  const char *home = std::getenv("HOME");
  return std::filesystem::path(home && home[0] ? home : ".") /
    "Library" / "Application Support" / "spektrafilm" / "ofx-defaults-v1.spkdefaults";
#else
  const char *home = std::getenv("HOME");
  const char *xdgConfigHome = std::getenv("XDG_CONFIG_HOME");
  const std::filesystem::path configRoot =
    xdgConfigHome && xdgConfigHome[0]
      ? std::filesystem::path(xdgConfigHome)
      : std::filesystem::path(home && home[0] ? home : ".") / ".config";
  return configRoot / "spektrafilm" / "ofx-defaults-v1.spkdefaults";
#endif
}

void obfuscateDefaultsText(std::string &text) {
  constexpr uint8_t key[] = {
    0x53, 0x70, 0x65, 0x6b, 0x74, 0x72, 0x61, 0x46,
    0x69, 0x6c, 0x6d, 0x4f, 0x46, 0x58, 0x31, 0x21
  };
  for (size_t i = 0; i < text.size(); ++i) {
    const uint8_t stream = static_cast<uint8_t>(key[i % sizeof(key)] + static_cast<uint8_t>((i * 37u) & 0xffu));
    text[i] = static_cast<char>(static_cast<uint8_t>(text[i]) ^ stream);
  }
}

bool loadSnapshotFromFile(const std::filesystem::path &path, DefaultsSnapshot &snapshot, bool &found, std::string &error) {
  found = false;
  error.clear();
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) {
    return true;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "Could not open spektrafilm defaults file for reading: " + path.string();
    return false;
  }
  std::string text{
    std::istreambuf_iterator<char>(input),
    std::istreambuf_iterator<char>()
  };
  if (!input.good() && !input.eof()) {
    error = "Could not read spektrafilm defaults file: " + path.string();
    return false;
  }
  obfuscateDefaultsText(text);
  DefaultsSnapshot decoded;
  if (!decodeDefaultsSnapshot(text, decoded)) {
    error = "spektrafilm defaults file is not a recognized defaults snapshot: " + path.string();
    return false;
  }
  snapshot = std::move(decoded);
  found = true;
  return true;
}

bool saveSnapshotToFile(const std::filesystem::path &path, const DefaultsSnapshot &snapshot, std::string &error) {
  error.clear();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    error = "Could not create spektrafilm defaults folder: " + path.parent_path().string();
    return false;
  }
  std::string text = encodeDefaultsSnapshot(snapshot);
  obfuscateDefaultsText(text);
  const std::filesystem::path tempPath = path.string() + ".tmp";
  {
    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output) {
      error = "Could not open spektrafilm defaults file for writing: " + tempPath.string();
      return false;
    }
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) {
      error = "Could not write spektrafilm defaults file: " + tempPath.string();
      return false;
    }
  }
  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
  }
  if (ec) {
    error = "Could not replace spektrafilm defaults file: " + path.string();
    return false;
  }
  return true;
}

bool deleteSnapshotFile(const std::filesystem::path &path, std::string &error) {
  error.clear();
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (ec) {
    error = "Could not delete spektrafilm defaults file: " + path.string();
    return false;
  }
  return true;
}

bool loadDefaultsFromFile(DefaultsSnapshot &snapshot, bool &found, std::string &error) {
  return loadSnapshotFromFile(userDefaultsPath(), snapshot, found, error);
}

bool saveDefaultsToFile(const DefaultsSnapshot &snapshot, std::string &error) {
  return saveSnapshotToFile(userDefaultsPath(), snapshot, error);
}

bool deleteDefaultsFile(std::string &error) {
  return deleteSnapshotFile(userDefaultsPath(), error);
}

#if defined __APPLE__
std::string clipboardStatusMessage(const char *prefix, OSStatus status) {
  return std::string(prefix) + " (clipboard status " + std::to_string(static_cast<long long>(status)) + ").";
}

bool writeTextToClipboard(const std::string &text, std::string &error) {
  error.clear();
  PasteboardRef pasteboard = nullptr;
  OSStatus status = PasteboardCreate(kPasteboardClipboard, &pasteboard);
  if (status != noErr || !pasteboard) {
    error = clipboardStatusMessage("Could not open system clipboard", status);
    return false;
  }
  status = PasteboardClear(pasteboard);
  if (status == noErr) {
    CFDataRef data = CFDataCreate(nullptr, reinterpret_cast<const UInt8 *>(text.data()), static_cast<CFIndex>(text.size()));
    if (data) {
      PasteboardItemID itemId = reinterpret_cast<PasteboardItemID>(1);
      status = PasteboardPutItemFlavor(pasteboard, itemId, CFSTR("com.spektrafilm.ofx-params"), data, 0);
      if (status == noErr) {
        status = PasteboardPutItemFlavor(pasteboard, itemId, CFSTR("public.utf8-plain-text"), data, 0);
      }
      CFRelease(data);
    } else {
      status = memFullErr;
    }
  }
  CFRelease(pasteboard);
  if (status != noErr) {
    error = clipboardStatusMessage("Could not write spektrafilm params to system clipboard", status);
    return false;
  }
  return true;
}

bool copyClipboardFlavor(PasteboardRef pasteboard, PasteboardItemID item, CFStringRef flavor, std::string &text) {
  CFDataRef data = nullptr;
  const OSStatus status = PasteboardCopyItemFlavorData(pasteboard, item, flavor, &data);
  if (status != noErr || !data) {
    return false;
  }
  const UInt8 *bytes = CFDataGetBytePtr(data);
  const CFIndex length = CFDataGetLength(data);
  text.assign(reinterpret_cast<const char *>(bytes), static_cast<size_t>(length));
  CFRelease(data);
  return true;
}

bool readTextFromClipboard(std::string &text, bool &found, std::string &error) {
  text.clear();
  found = false;
  error.clear();
  PasteboardRef pasteboard = nullptr;
  OSStatus status = PasteboardCreate(kPasteboardClipboard, &pasteboard);
  if (status != noErr || !pasteboard) {
    error = clipboardStatusMessage("Could not open system clipboard", status);
    return false;
  }
  PasteboardSynchronize(pasteboard);
  ItemCount itemCount = 0;
  status = PasteboardGetItemCount(pasteboard, &itemCount);
  if (status != noErr) {
    CFRelease(pasteboard);
    error = clipboardStatusMessage("Could not inspect system clipboard", status);
    return false;
  }
  for (ItemCount index = 1; index <= itemCount; ++index) {
    PasteboardItemID item = nullptr;
    if (PasteboardGetItemIdentifier(pasteboard, static_cast<CFIndex>(index), &item) != noErr || !item) {
      continue;
    }
    if (copyClipboardFlavor(pasteboard, item, CFSTR("com.spektrafilm.ofx-params"), text) ||
        copyClipboardFlavor(pasteboard, item, CFSTR("public.utf8-plain-text"), text)) {
      found = true;
      break;
    }
  }
  CFRelease(pasteboard);
  return true;
}
#elif defined _WIN32
std::string clipboardStatusMessage(const char *prefix, DWORD status) {
  return std::string(prefix) + " (Win32 error " + std::to_string(static_cast<unsigned long>(status)) + ").";
}

UINT spektraClipboardFormat() {
  return RegisterClipboardFormatA("com.spektrafilm.ofx-params");
}

bool writeClipboardBytes(UINT format, const std::string &text, std::string &error) {
  if (format == 0) {
    error = clipboardStatusMessage("Could not register spektrafilm clipboard format", GetLastError());
    return false;
  }

  constexpr uint64_t headerSize = sizeof(uint64_t);
  const uint64_t textSize = static_cast<uint64_t>(text.size());
  if (textSize > static_cast<uint64_t>(std::numeric_limits<SIZE_T>::max() - headerSize)) {
    error = "spektrafilm params are too large for the system clipboard.";
    return false;
  }

  const SIZE_T allocationSize = static_cast<SIZE_T>(headerSize + textSize);
  HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, allocationSize);
  if (!memory) {
    error = clipboardStatusMessage("Could not allocate system clipboard memory", GetLastError());
    return false;
  }

  void *locked = GlobalLock(memory);
  if (!locked) {
    error = clipboardStatusMessage("Could not lock system clipboard memory", GetLastError());
    GlobalFree(memory);
    return false;
  }
  std::memcpy(locked, &textSize, sizeof(textSize));
  if (!text.empty()) {
    std::memcpy(static_cast<uint8_t *>(locked) + headerSize, text.data(), text.size());
  }
  GlobalUnlock(memory);

  if (!SetClipboardData(format, memory)) {
    error = clipboardStatusMessage("Could not write spektrafilm params to system clipboard", GetLastError());
    GlobalFree(memory);
    return false;
  }
  return true;
}

void writeClipboardTextFallback(const std::string &text) {
  HGLOBAL memory = GlobalAlloc(GMEM_MOVEABLE, static_cast<SIZE_T>(text.size() + 1u));
  if (!memory) {
    return;
  }
  void *locked = GlobalLock(memory);
  if (!locked) {
    GlobalFree(memory);
    return;
  }
  std::memcpy(locked, text.data(), text.size());
  static_cast<char *>(locked)[text.size()] = '\0';
  GlobalUnlock(memory);
  if (!SetClipboardData(CF_TEXT, memory)) {
    GlobalFree(memory);
  }
}

bool writeTextToClipboard(const std::string &text, std::string &error) {
  error.clear();
  if (!OpenClipboard(nullptr)) {
    error = clipboardStatusMessage("Could not open system clipboard", GetLastError());
    return false;
  }

  if (!EmptyClipboard()) {
    error = clipboardStatusMessage("Could not clear system clipboard", GetLastError());
    CloseClipboard();
    return false;
  }

  const bool wroteBytes = writeClipboardBytes(spektraClipboardFormat(), text, error);
  if (wroteBytes) {
    writeClipboardTextFallback(text);
  }
  CloseClipboard();
  return wroteBytes;
}

bool readClipboardBytes(UINT format, std::string &text) {
  if (format == 0 || !IsClipboardFormatAvailable(format)) {
    return false;
  }
  HGLOBAL memory = GetClipboardData(format);
  if (!memory) {
    return false;
  }
  const SIZE_T allocationSize = GlobalSize(memory);
  if (allocationSize < sizeof(uint64_t)) {
    return false;
  }
  void *locked = GlobalLock(memory);
  if (!locked) {
    return false;
  }

  uint64_t textSize = 0;
  std::memcpy(&textSize, locked, sizeof(textSize));
  const SIZE_T payloadSize = allocationSize - sizeof(uint64_t);
  if (textSize > static_cast<uint64_t>(payloadSize)) {
    GlobalUnlock(memory);
    return false;
  }

  const auto *bytes = static_cast<const char *>(locked) + sizeof(uint64_t);
  text.assign(bytes, static_cast<size_t>(textSize));
  GlobalUnlock(memory);
  return true;
}

bool readClipboardTextFallback(std::string &text) {
  if (!IsClipboardFormatAvailable(CF_TEXT)) {
    return false;
  }
  HGLOBAL memory = GetClipboardData(CF_TEXT);
  if (!memory) {
    return false;
  }
  const char *locked = static_cast<const char *>(GlobalLock(memory));
  if (!locked) {
    return false;
  }
  text.assign(locked);
  GlobalUnlock(memory);
  return true;
}

bool readTextFromClipboard(std::string &text, bool &found, std::string &error) {
  text.clear();
  found = false;
  error.clear();
  if (!OpenClipboard(nullptr)) {
    error = clipboardStatusMessage("Could not open system clipboard", GetLastError());
    return false;
  }

  found = readClipboardBytes(spektraClipboardFormat(), text) || readClipboardTextFallback(text);
  CloseClipboard();
  return true;
}
#else
bool writeTextToClipboard(const std::string &, std::string &error) {
  error = "Writing spektrafilm params to the system clipboard is not implemented on this platform.";
  return false;
}

bool readTextFromClipboard(std::string &text, bool &found, std::string &error) {
  text.clear();
  found = false;
  error = "Reading spektrafilm params from the system clipboard is not implemented on this platform.";
  return false;
}
#endif

void showMessage(OfxImageEffectHandle effect, const char *type, const char *id, const std::string &message) {
  if (gMessageHost) {
    gMessageHost->message(effect, type, id, "%s", message.c_str());
  }
}

bool getParamValueAtTime(OfxParamHandle handle, OfxTime time, const ParamDefault &entry, StoredParamValue &value) {
  if (!handle) {
    return false;
  }
  value.kind = entry.kind;
  switch (entry.kind) {
    case ParamValueKind::Int:
    case ParamValueKind::Bool: {
      int current = entry.intDefault;
      if (gParamHost->paramGetValueAtTime(handle, time, &current) != kOfxStatOK) {
        return false;
      }
      value.intValue[0] = current;
      return true;
    }
    case ParamValueKind::Double: {
      double current = entry.doubleDefault[0];
      if (gParamHost->paramGetValueAtTime(handle, time, &current) != kOfxStatOK) {
        return false;
      }
      value.doubleValue[0] = current;
      return true;
    }
    case ParamValueKind::Double2D: {
      double x = entry.doubleDefault[0];
      double y = entry.doubleDefault[1];
      if (gParamHost->paramGetValueAtTime(handle, time, &x, &y) != kOfxStatOK) {
        return false;
      }
      value.doubleValue[0] = x;
      value.doubleValue[1] = y;
      return true;
    }
    case ParamValueKind::Double3D: {
      double x = entry.doubleDefault[0];
      double y = entry.doubleDefault[1];
      double z = entry.doubleDefault[2];
      if (gParamHost->paramGetValueAtTime(handle, time, &x, &y, &z) != kOfxStatOK) {
        return false;
      }
      value.doubleValue[0] = x;
      value.doubleValue[1] = y;
      value.doubleValue[2] = z;
      return true;
    }
  }
  return false;
}

bool setParamValue(OfxParamHandle handle, const StoredParamValue &value) {
  if (!handle) {
    return false;
  }
  switch (value.kind) {
    case ParamValueKind::Int:
    case ParamValueKind::Bool:
      return gParamHost->paramSetValue(handle, value.intValue[0]) == kOfxStatOK;
    case ParamValueKind::Double:
      return gParamHost->paramSetValue(handle, value.doubleValue[0]) == kOfxStatOK;
    case ParamValueKind::Double2D:
      return gParamHost->paramSetValue(handle, value.doubleValue[0], value.doubleValue[1]) == kOfxStatOK;
    case ParamValueKind::Double3D:
      return gParamHost->paramSetValue(handle, value.doubleValue[0], value.doubleValue[1], value.doubleValue[2]) == kOfxStatOK;
  }
  return false;
}

void setParamParent(OfxPropertySetHandle props, const char *parent) {
  if (parent && parent[0]) {
    gPropHost->propSetString(props, kOfxParamPropParent, 0, parent);
  }
}

void setParamDescriptorHidden(OfxPropertySetHandle props, bool hidden) {
  gPropHost->propSetInt(props, kOfxParamPropSecret, 0, hidden ? 1 : 0);
  gPropHost->propSetInt(props, kOfxParamPropEnabled, 0, hidden ? 0 : 1);
}

void setParamHint(OfxPropertySetHandle props, const char *name) {
  const char *hint = spektrafilm::tooltipForParam(name);
  if (hint && hint[0]) {
    gPropHost->propSetString(props, kOfxParamPropHint, 0, hint);
  }
}

void defineGroup(OfxParamSetHandle paramSet, const char *name, const char *label, bool openByDefault) {
  if (!shouldDefineGroup(name)) {
    return;
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeGroup, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamDescriptorHidden(props, !groupVisibleInFlavor(name));
  gPropHost->propSetInt(props, kOfxParamPropGroupOpen, 0, openByDefault ? 1 : 0);
}

void defineChoice(OfxParamSetHandle paramSet, const char *name, const char *label, const char **options, int optionCount, int defaultValue, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  if (optionCount <= 0) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    defaultValue = stored.intValue[0];
  }
  defaultValue = std::clamp(defaultValue, 0, optionCount - 1);
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeChoice, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  for (int i = 0; i < optionCount; ++i) {
    gPropHost->propSetString(props, kOfxParamPropChoiceOption, i, options[i]);
  }
  gPropHost->propSetInt(props, kOfxParamPropDefault, 0, defaultValue);
}

void defineDouble(OfxParamSetHandle paramSet, const char *name, const char *label, double defaultValue, double min, double max, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    defaultValue = stored.doubleValue[0];
  }
  defaultValue = std::clamp(defaultValue, min, max);
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeDouble, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, defaultValue);
  gPropHost->propSetDouble(props, kOfxParamPropMin, 0, min);
  gPropHost->propSetDouble(props, kOfxParamPropMax, 0, max);
  gPropHost->propSetDouble(props, kOfxParamPropDisplayMin, 0, min);
  gPropHost->propSetDouble(props, kOfxParamPropDisplayMax, 0, max);
}

void defineInt(OfxParamSetHandle paramSet, const char *name, const char *label, int defaultValue, int min, int max, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    defaultValue = stored.intValue[0];
  }
  defaultValue = std::clamp(defaultValue, min, max);
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeInteger, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetInt(props, kOfxParamPropDefault, 0, defaultValue);
  gPropHost->propSetInt(props, kOfxParamPropMin, 0, min);
  gPropHost->propSetInt(props, kOfxParamPropMax, 0, max);
  gPropHost->propSetInt(props, kOfxParamPropDisplayMin, 0, min);
  gPropHost->propSetInt(props, kOfxParamPropDisplayMax, 0, max);
}

void defineBool(OfxParamSetHandle paramSet, const char *name, const char *label, bool defaultValue, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    defaultValue = stored.intValue[0] != 0;
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetInt(props, kOfxParamPropDefault, 0, defaultValue ? 1 : 0);
}

void defineLabel(OfxParamSetHandle paramSet, const char *name, const char *descriptor, const char *value, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeString, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, descriptor);
  gPropHost->propSetString(props, kOfxPropShortLabel, 0, descriptor);
  gPropHost->propSetString(props, kOfxPropLongLabel, 0, descriptor);
  gPropHost->propSetString(props, kOfxParamPropDefault, 0, value);
  gPropHost->propSetString(props, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);
  gPropHost->propSetInt(props, kOfxParamPropEnabled, 0, 0);
  gPropHost->propSetInt(props, kOfxParamPropPersistant, 0, 0);
  gPropHost->propSetInt(props, kOfxParamPropEvaluateOnChange, 0, 0);
  setParamParent(props, parent);
}

void defineSingleLineString(OfxParamSetHandle paramSet, const char *name, const char *label, const char *defaultValue, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeString, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetString(props, kOfxParamPropStringMode, 0, kOfxParamStringIsSingleLine);
  gPropHost->propSetString(props, kOfxParamPropDefault, 0, defaultValue);
  gPropHost->propSetInt(props, kOfxParamPropEvaluateOnChange, 0, 0);
}

void definePushButton(OfxParamSetHandle paramSet, const char *name, const char *label, const char *parent = nullptr) {
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypePushButton, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetInt(props, kOfxParamPropPersistant, 0, 0);
  gPropHost->propSetInt(props, kOfxParamPropEvaluateOnChange, 0, 1);
}

void defineHiddenBool(OfxParamSetHandle paramSet, const char *name, bool value) {
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    value = stored.intValue[0] != 0;
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeBoolean, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, name);
  setParamDescriptorHidden(props, true);
  gPropHost->propSetInt(props, kOfxParamPropDefault, 0, value ? 1 : 0);
}

void defineRGB(OfxParamSetHandle paramSet, const char *name, const char *label, double r, double g, double b, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    r = stored.doubleValue[0];
    g = stored.doubleValue[1];
    b = stored.doubleValue[2];
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeRGB, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, r);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 1, g);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 2, b);
}

void defineDouble3D(OfxParamSetHandle paramSet, const char *name, const char *label, double x, double y, double z, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    x = stored.doubleValue[0];
    y = stored.doubleValue[1];
    z = stored.doubleValue[2];
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeDouble3D, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, x);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 1, y);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 2, z);
}

void defineDouble3DRange(
  OfxParamSetHandle paramSet,
  const char *name,
  const char *label,
  double x,
  double y,
  double z,
  double min,
  double max,
  const char *parent = nullptr
) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    x = stored.doubleValue[0];
    y = stored.doubleValue[1];
    z = stored.doubleValue[2];
  }
  x = std::clamp(x, min, max);
  y = std::clamp(y, min, max);
  z = std::clamp(z, min, max);
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeDouble3D, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, x);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 1, y);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 2, z);
  for (int i = 0; i < 3; ++i) {
    gPropHost->propSetDouble(props, kOfxParamPropMin, i, min);
    gPropHost->propSetDouble(props, kOfxParamPropMax, i, max);
    gPropHost->propSetDouble(props, kOfxParamPropDisplayMin, i, min);
    gPropHost->propSetDouble(props, kOfxParamPropDisplayMax, i, max);
  }
}

void defineDouble2D(OfxParamSetHandle paramSet, const char *name, const char *label, double x, double y, const char *parent = nullptr) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    x = stored.doubleValue[0];
    y = stored.doubleValue[1];
  }
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeDouble2D, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, x);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 1, y);
}

void defineDouble2DRange(
  OfxParamSetHandle paramSet,
  const char *name,
  const char *label,
  double x,
  double y,
  double min,
  double max,
  const char *parent = nullptr
) {
  if (!shouldDefineParam(name)) {
    return;
  }
  StoredParamValue stored{};
  if (storedValueForDefault(name, stored)) {
    x = stored.doubleValue[0];
    y = stored.doubleValue[1];
  }
  x = std::clamp(x, min, max);
  y = std::clamp(y, min, max);
  OfxPropertySetHandle props = nullptr;
  gParamHost->paramDefine(paramSet, kOfxParamTypeDouble2D, name, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, label);
  setParamHint(props, name);
  setParamParent(props, parent);
  setParamDescriptorHidden(props, parameterHiddenInFlavor(name));
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 0, x);
  gPropHost->propSetDouble(props, kOfxParamPropDefault, 1, y);
  for (int i = 0; i < 2; ++i) {
    gPropHost->propSetDouble(props, kOfxParamPropMin, i, min);
    gPropHost->propSetDouble(props, kOfxParamPropMax, i, max);
    gPropHost->propSetDouble(props, kOfxParamPropDisplayMin, i, min);
    gPropHost->propSetDouble(props, kOfxParamPropDisplayMax, i, max);
  }
}

void cacheParam(OfxParamSetHandle paramSet, const char *name, OfxParamHandle &handle) {
  gParamHost->paramGetHandle(paramSet, name, &handle, nullptr);
}

OfxParamHandle paramHandleForName(OfxParamSetHandle paramSet, const char *name) {
  OfxParamHandle handle = nullptr;
  if (gParamHost->paramGetHandle(paramSet, name, &handle, nullptr) != kOfxStatOK) {
    return nullptr;
  }
  return handle;
}

void randomizeGrainSeedForNewInstance(InstanceData *data) {
  if (!data || !data->grainSeed) {
    return;
  }
  int current = 0;
  if (gParamHost->paramGetValue(data->grainSeed, &current) != kOfxStatOK) {
    return;
  }
  if (current == gDescriptorGrainSeedDefault || current == 0 || current == 1) {
    gParamHost->paramSetValue(data->grainSeed, randomGrainSeed());
  }
}

bool applySnapshotToParamSet(OfxParamSetHandle paramSet, const DefaultsSnapshot &snapshot, bool includeGrainSeed = true) {
  bool appliedAny = false;
  for (const ParamDefault &entry : kParamDefaults) {
    if (!shouldDefineParam(entry.name)) {
      continue;
    }
    if (!includeGrainSeed && isGrainSeedParam(entry.name)) {
      continue;
    }
    const auto found = snapshot.find(entry.name);
    if (found == snapshot.end() || found->second.kind != entry.kind) {
      continue;
    }
    OfxParamHandle handle = paramHandleForName(paramSet, entry.name);
    if (handle && setParamValue(handle, found->second)) {
      appliedAny = true;
    }
  }
  return appliedAny;
}

bool resetParamSetToFactory(OfxParamSetHandle paramSet) {
  bool resetAny = false;
  for (const ParamDefault &entry : kParamDefaults) {
    if (!shouldDefineParam(entry.name)) {
      continue;
    }
    OfxParamHandle handle = paramHandleForName(paramSet, entry.name);
    if (handle && setParamValue(handle, factoryStoredValue(entry))) {
      resetAny = true;
    }
  }
  return resetAny;
}

void captureParamSetSnapshot(OfxParamSetHandle paramSet, OfxTime time, DefaultsSnapshot &snapshot, bool includeGrainSeed = true) {
  for (const ParamDefault &entry : kParamDefaults) {
    if (!shouldDefineParam(entry.name)) {
      continue;
    }
    if (!includeGrainSeed && isGrainSeedParam(entry.name)) {
      snapshot.erase(entry.name);
      continue;
    }
    OfxParamHandle handle = paramHandleForName(paramSet, entry.name);
    StoredParamValue value{};
    if (handle && getParamValueAtTime(handle, time, entry, value)) {
      snapshot[entry.name] = value;
    }
  }
}

bool saveVisibleDefaults(OfxParamSetHandle paramSet, OfxTime time, std::string &error) {
  DefaultsSnapshot snapshot;
  bool found = false;
  if (!loadDefaultsFromFile(snapshot, found, error)) {
    return false;
  }
  captureParamSetSnapshot(paramSet, time, snapshot, false);
  return saveDefaultsToFile(snapshot, error);
}

bool copyVisibleParams(OfxParamSetHandle paramSet, OfxTime time, std::string &error) {
  DefaultsSnapshot snapshot;
  captureParamSetSnapshot(paramSet, time, snapshot);
  std::string text = encodeDefaultsSnapshot(snapshot);
  obfuscateDefaultsText(text);
  return writeTextToClipboard(text, error);
}

const char *pluginFlavorName() {
  switch (kPluginFlavor) {
    case PluginFlavor::Flow:
      return "spektrafilm flow";
    case PluginFlavor::Pro:
      return "spektrafilm";
    case PluginFlavor::FilmDev:
      return "spektrafilm dev";
  }
  return "spektrafilm dev";
}

const char *processName(spektrafilm::ProcessMode process) {
  switch (process) {
    case spektrafilm::ProcessMode::ScanNegative:
      return "Scan negative";
    case spektrafilm::ProcessMode::ProcessNegative:
      return "Process negative";
    case spektrafilm::ProcessMode::PrintSimulation:
    default:
      return "Print simulation";
  }
}

const char *outputRoleName(spektrafilm::OutputRole role) {
  switch (role) {
    case spektrafilm::OutputRole::DisplayHdr:
      return "Display Out HDR";
    case spektrafilm::OutputRole::Rcm:
      return "RCM/ACES (Beta)";
    case spektrafilm::OutputRole::DisplaySdr:
    default:
      return "Display Out SDR";
  }
}

const char *colorSpaceName(spektrafilm::ColorSpace colorSpace) {
  switch (colorSpace) {
    case spektrafilm::ColorSpace::ArriLogC4:
      return "ARRI LogC4";
    case spektrafilm::ColorSpace::ArriLogC3Ei800:
      return "ARRI LogC3 EI800";
    case spektrafilm::ColorSpace::BmdFilmWideGamutGen5:
      return "BMDFilm WideGamut Gen5";
    case spektrafilm::ColorSpace::DavinciIntermediateWideGamut:
      return "DaVinci Intermediate WideGamut";
    case spektrafilm::ColorSpace::RedLog3G10RedWideGamutRgb:
      return "RED Log3G10 REDWideGamutRGB";
    case spektrafilm::ColorSpace::SonySLog3SGamut3:
      return "Sony S-Log3 S-Gamut3";
    case spektrafilm::ColorSpace::SonySLog3SGamut3Cine:
      return "Sony S-Log3 S-Gamut3.Cine";
    case spektrafilm::ColorSpace::CanonLog2CinemaGamutD55:
      return "Canon Log2 CinemaGamut D55";
    case spektrafilm::ColorSpace::CanonLog3CinemaGamutD55:
      return "Canon Log3 CinemaGamut D55";
    case spektrafilm::ColorSpace::PanasonicVLogVGamut:
      return "Panasonic V-Log V-Gamut";
    case spektrafilm::ColorSpace::Aces2065_1:
      return "ACES2065-1";
    case spektrafilm::ColorSpace::AcesCg:
      return "ACEScg";
    case spektrafilm::ColorSpace::AcesCct:
      return "ACEScct";
    case spektrafilm::ColorSpace::AcesCc:
      return "ACEScc";
    case spektrafilm::ColorSpace::LinearRec2020:
      return "Linear Rec.2020";
    case spektrafilm::ColorSpace::LinearRec709:
      return "Linear Rec.709";
    case spektrafilm::ColorSpace::LinearP3D65:
      return "Linear P3-D65";
    case spektrafilm::ColorSpace::Srgb:
      return "sRGB";
    case spektrafilm::ColorSpace::DisplayP3:
      return "Display P3";
    case spektrafilm::ColorSpace::ProPhotoRgb:
      return "ProPhoto RGB";
    case spektrafilm::ColorSpace::AdobeRgb1998:
      return "Adobe RGB (1998)";
    case spektrafilm::ColorSpace::DciP3:
      return "DCI-P3";
    case spektrafilm::ColorSpace::P3D65Gamma22:
      return "P3-D65 Gamma 2.2";
    case spektrafilm::ColorSpace::P3D65Gamma26:
      return "P3-D65 Gamma 2.6";
    case spektrafilm::ColorSpace::Rec709Gamma22:
      return "Rec.709 Gamma 2.2";
    case spektrafilm::ColorSpace::Rec709Gamma24:
      return "Rec.709 Gamma 2.4";
  }
  return "Unknown";
}

std::string sanitizePathComponent(std::string value) {
  for (char &ch : value) {
    const unsigned char c = static_cast<unsigned char>(ch);
    if (std::isalnum(c)) {
      continue;
    }
    ch = '_';
  }
  while (value.find("__") != std::string::npos) {
    value.replace(value.find("__"), 2, "_");
  }
  while (!value.empty() && value.front() == '_') {
    value.erase(value.begin());
  }
  while (!value.empty() && value.back() == '_') {
    value.pop_back();
  }
  return value.empty() ? "unknown" : value;
}

std::string getProfileName(const spektrafilm::ProfileCurveSet *profile, const char *fallback) {
  return profile && profile->name && profile->name[0] ? profile->name : fallback;
}

std::string currentDatePrefix() {
  std::time_t now = std::time(nullptr);
  std::tm local{};
#if defined _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  std::ostringstream out;
  out << std::put_time(&local, "%y%m%d");
  return out.str();
}

std::string currentTimestampSuffix() {
  std::time_t now = std::time(nullptr);
  std::tm local{};
#if defined _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  std::ostringstream out;
  out << std::put_time(&local, "%y%m%d_%H%M%S");
  return out.str();
}

std::string currentReadableTimestamp() {
  std::time_t now = std::time(nullptr);
  std::tm local{};
#if defined _WIN32
  localtime_s(&local, &now);
#else
  localtime_r(&now, &local);
#endif
  std::ostringstream out;
  out << std::put_time(&local, "%Y-%m-%d %H:%M:%S");
  return out.str();
}

std::string randomExportCode() {
  static std::mt19937 generator{std::random_device{}()};
  static std::uniform_int_distribution<int> distribution(0, 0xffffff);
  std::ostringstream out;
  out << std::uppercase << std::hex << std::setw(6) << std::setfill('0') << distribution(generator);
  return out.str();
}

std::filesystem::path envPath(const char *name, const std::filesystem::path &fallback) {
  const char *value = std::getenv(name);
  return std::filesystem::path(value && value[0] ? value : fallback.string());
}

bool envFlagEnabledOrDefault(const char *name, bool defaultValue) {
  const char *value = std::getenv(name);
  if (!value || value[0] == '\0') {
    return defaultValue;
  }
  if (std::strcmp(value, "0") == 0 ||
      std::strcmp(value, "false") == 0 ||
      std::strcmp(value, "FALSE") == 0 ||
      std::strcmp(value, "no") == 0 ||
      std::strcmp(value, "NO") == 0 ||
      std::strcmp(value, "off") == 0 ||
      std::strcmp(value, "OFF") == 0) {
    return false;
  }
  return true;
}

bool resetRendererAfterEndSequence() {
  return envFlagEnabledOrDefault("SPEKTRAFILM_OFX_RESET_RENDERER_AFTER_END_SEQUENCE", false);
}

bool releaseTransientAfterEndSequence() {
#if defined(__APPLE__)
  constexpr bool defaultValue = true;
#else
  constexpr bool defaultValue = false;
#endif
  return envFlagEnabledOrDefault("SPEKTRAFILM_OFX_RELEASE_TRANSIENT_AFTER_END_SEQUENCE", defaultValue);
}

bool resetRendererOnPurgeCaches() {
  return envFlagEnabledOrDefault("SPEKTRAFILM_OFX_RESET_RENDERER_ON_PURGE", true);
}

bool resetRendererAfterLutExport() {
#if defined(__APPLE__)
  constexpr bool defaultValue = false;
#else
  constexpr bool defaultValue = true;
#endif
  return envFlagEnabledOrDefault("SPEKTRAFILM_OFX_RESET_RENDERER_AFTER_LUT_EXPORT", defaultValue);
}

std::filesystem::path homeFolder() {
#if defined _WIN32
  return envPath("USERPROFILE", ".");
#else
  return envPath("HOME", ".");
#endif
}

std::filesystem::path userLutFolder() {
#if defined _WIN32
  return homeFolder() / "Documents" / "spektrafilm";
#elif defined __APPLE__
  return homeFolder() / "Movies" / "spektrafilm";
#else
  return homeFolder() / "spektrafilm";
#endif
}

std::filesystem::path userDocumentsFolder() {
#if defined _WIN32
  return homeFolder() / "Documents";
#else
  return homeFolder() / "Documents";
#endif
}

std::filesystem::path lutDestinationFolder(int destination) {
  const std::filesystem::path homePath = homeFolder();
  switch (destination) {
    case 1:
#if defined _WIN32
      return envPath("PROGRAMDATA", "C:\\ProgramData") /
        "Blackmagic Design" / "DaVinci Resolve" / "Support" / "LUT" / "spektrafilm";
#elif defined __APPLE__
      return std::filesystem::path("/") / "Library" / "Application Support" /
        "Blackmagic Design" / "DaVinci Resolve" / "LUT" / "spektrafilm";
#else
      return userLutFolder();
#endif
    case 2:
      return homePath / ".nuke" / "spektrafilm";
    case 3:
#if defined _WIN32
      return envPath("PROGRAMFILES", "C:\\Program Files") /
        "Adobe" / "Common" / "LUTs" / "Creative" / "spektrafilm";
#elif defined __APPLE__
      return std::filesystem::path("/") / "Library" / "Application Support" /
        "Adobe" / "Common" / "LUTs" / "Creative" / "spektrafilm";
#else
      return userLutFolder();
#endif
    case 4:
#if defined _WIN32
      return userLutFolder();
#elif defined __APPLE__
      return homePath / "Library" / "Application Support" /
        "ProApps" / "Custom LUTs" / "spektrafilm";
#else
      return userLutFolder();
#endif
    case 0:
    default:
      return userLutFolder();
  }
}

std::filesystem::path generatedLutExportPath(int destination, const std::string &identifier) {
  const std::filesystem::path folder = lutDestinationFolder(destination);
  const std::string cleanIdentifier = sanitizePathComponent(identifier.empty() ? "spektrafilm" : identifier);
  const std::string date = currentDatePrefix();
  for (int attempt = 0; attempt < 64; ++attempt) {
    std::filesystem::path path = folder / (date + "_" + cleanIdentifier + "_" + randomExportCode() + ".cube");
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      return path;
    }
  }
  return folder / (date + "_" + cleanIdentifier + "_" + randomExportCode() + ".cube");
}

#if defined __linux__
std::string shellQuote(const std::string &value) {
  std::string quoted = "'";
  for (char ch : value) {
    if (ch == '\'') {
      quoted += "'\\''";
    } else {
      quoted += ch;
    }
  }
  quoted += "'";
  return quoted;
}

std::filesystem::path executableOnPath(const char *name) {
  const char *pathEnv = std::getenv("PATH");
  if (!pathEnv || !pathEnv[0]) {
    return {};
  }

  std::string pathList(pathEnv);
  size_t start = 0;
  while (start <= pathList.size()) {
    const size_t end = pathList.find(':', start);
    const std::string directory = pathList.substr(
      start,
      end == std::string::npos ? std::string::npos : end - start
    );
    const std::filesystem::path candidate =
      std::filesystem::path(directory.empty() ? "." : directory) / name;
    const std::string candidateString = candidate.string();
    if (access(candidateString.c_str(), X_OK) == 0) {
      return candidate;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1u;
  }
  return {};
}
#endif

std::filesystem::path bundledUserManualPath() {
#if defined __APPLE__
  Dl_info imageInfo{};
  if (dladdr(&gPluginImageAnchor, &imageInfo) == 0 || !imageInfo.dli_fname) {
    return {};
  }
  const std::filesystem::path imagePath(imageInfo.dli_fname);
  const std::filesystem::path contentsPath = imagePath.parent_path().parent_path();
  if (contentsPath.empty()) {
    return {};
  }
  return contentsPath / "Resources" / "manual.pdf";
#elif defined _WIN32
  HMODULE module = nullptr;
  if (!GetModuleHandleExW(
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
        reinterpret_cast<LPCWSTR>(&gPluginImageAnchor),
        &module
      )) {
    return {};
  }

  std::vector<wchar_t> buffer(MAX_PATH);
  for (;;) {
    const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0) {
      return {};
    }
    if (length < buffer.size() - 1) {
      const std::filesystem::path imagePath(std::wstring(buffer.data(), length));
      const std::filesystem::path contentsPath = imagePath.parent_path().parent_path();
      if (contentsPath.empty()) {
        return {};
      }
      return contentsPath / "Resources" / "manual.pdf";
    }
    buffer.resize(buffer.size() * 2u);
  }
  return {};
#elif defined __linux__
  Dl_info imageInfo{};
  if (dladdr(&gPluginImageAnchor, &imageInfo) == 0 || !imageInfo.dli_fname) {
    return {};
  }
  const std::filesystem::path imagePath(imageInfo.dli_fname);
  const std::filesystem::path contentsPath = imagePath.parent_path().parent_path();
  if (contentsPath.empty()) {
    return {};
  }
  return contentsPath / "Resources" / "manual.pdf";
#else
  return {};
#endif
}

bool openBundledUserManual(std::string &error) {
  const std::filesystem::path manualPath = bundledUserManualPath();
  if (manualPath.empty() || !std::filesystem::is_regular_file(manualPath)) {
    error = "Could not find manual.pdf in the OFX bundle resources.";
    return false;
  }

#if defined __APPLE__
  const std::string pathString = manualPath.string();
  CFURLRef manualUrl = CFURLCreateFromFileSystemRepresentation(
    kCFAllocatorDefault,
    reinterpret_cast<const UInt8 *>(pathString.c_str()),
    static_cast<CFIndex>(pathString.size()),
    false
  );
  if (!manualUrl) {
    error = "Could not create a file URL for the spektrafilm user manual.";
    return false;
  }

  const OSStatus status = LSOpenCFURLRef(manualUrl, nullptr);
  CFRelease(manualUrl);
  if (status != noErr) {
    error = "Could not open the spektrafilm user manual. LaunchServices returned " + std::to_string(status) + ".";
    return false;
  }
  return true;
#elif defined _WIN32
  const HINSTANCE result = ShellExecuteW(
    nullptr,
    L"open",
    manualPath.wstring().c_str(),
    nullptr,
    nullptr,
    SW_SHOWNORMAL
  );
  if (reinterpret_cast<intptr_t>(result) <= 32) {
    error = "Could not open the spektrafilm user manual. ShellExecute returned " +
      std::to_string(reinterpret_cast<intptr_t>(result)) + ".";
    return false;
  }
  return true;
#elif defined __linux__
  const std::filesystem::path xdgOpen = executableOnPath("xdg-open");
  if (xdgOpen.empty()) {
    error = "Could not open the spektrafilm user manual because xdg-open was not found on PATH.";
    return false;
  }

  const std::string command =
    shellQuote(xdgOpen.string()) + " " + shellQuote(manualPath.string()) + " >/dev/null 2>&1 &";
  if (std::system(command.c_str()) != 0) {
    error = "Could not open the spektrafilm user manual with xdg-open.";
    return false;
  }
  return true;
#else
  error = "Opening the spektrafilm user manual is not implemented on this platform.";
  return false;
#endif
}

std::string getStringValue(OfxParamHandle handle) {
  if (!handle) {
    return {};
  }
  char *value = nullptr;
  if (gParamHost->paramGetValue(handle, &value) != kOfxStatOK || !value) {
    return {};
  }
  return value;
}

constexpr const char *kPresetExtension = ".spkpreset";
constexpr const char *kPresetFormat = "spektrafilm-preset-v1";

struct PresetEntry {
  std::string displayName;
  std::string created;
  std::filesystem::path path;
};

std::filesystem::path presetFolder() {
  return userDocumentsFolder() / "spektrafilm" / "presets";
}

std::string lowercaseAscii(std::string value) {
  for (char &ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

bool isPresetFilePath(const std::filesystem::path &path) {
  return lowercaseAscii(path.extension().string()) == kPresetExtension;
}

std::string trimString(std::string value) {
  const auto isSpace = [](unsigned char ch) {
    return std::isspace(ch) != 0;
  };
  while (!value.empty() && isSpace(static_cast<unsigned char>(value.front()))) {
    value.erase(value.begin());
  }
  while (!value.empty() && isSpace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  return value;
}

std::string jsonEscape(const std::string &value) {
  std::ostringstream out;
  for (unsigned char ch : value) {
    switch (ch) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\b':
        out << "\\b";
        break;
      case '\f':
        out << "\\f";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        if (ch < 0x20u) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch) << std::dec;
        } else {
          out << static_cast<char>(ch);
        }
        break;
    }
  }
  return out.str();
}

bool appendUtf8Codepoint(uint32_t codepoint, std::string &out) {
  if (codepoint <= 0x7fu) {
    out.push_back(static_cast<char>(codepoint));
    return true;
  }
  if (codepoint <= 0x7ffu) {
    out.push_back(static_cast<char>(0xc0u | (codepoint >> 6u)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    return true;
  }
  if (codepoint <= 0xffffu) {
    out.push_back(static_cast<char>(0xe0u | (codepoint >> 12u)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    return true;
  }
  if (codepoint <= 0x10ffffu) {
    out.push_back(static_cast<char>(0xf0u | (codepoint >> 18u)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 12u) & 0x3fu)));
    out.push_back(static_cast<char>(0x80u | ((codepoint >> 6u) & 0x3fu)));
    out.push_back(static_cast<char>(0x80u | (codepoint & 0x3fu)));
    return true;
  }
  return false;
}

int hexNibble(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return 10 + ch - 'a';
  }
  if (ch >= 'A' && ch <= 'F') {
    return 10 + ch - 'A';
  }
  return -1;
}

bool jsonUnescapeString(const std::string &text, size_t &pos, std::string &value) {
  if (pos >= text.size() || text[pos] != '"') {
    return false;
  }
  ++pos;
  std::string decoded;
  while (pos < text.size()) {
    const char ch = text[pos++];
    if (ch == '"') {
      value = std::move(decoded);
      return true;
    }
    if (ch != '\\') {
      decoded.push_back(ch);
      continue;
    }
    if (pos >= text.size()) {
      return false;
    }
    const char escaped = text[pos++];
    switch (escaped) {
      case '"':
      case '\\':
      case '/':
        decoded.push_back(escaped);
        break;
      case 'b':
        decoded.push_back('\b');
        break;
      case 'f':
        decoded.push_back('\f');
        break;
      case 'n':
        decoded.push_back('\n');
        break;
      case 'r':
        decoded.push_back('\r');
        break;
      case 't':
        decoded.push_back('\t');
        break;
      case 'u': {
        if (pos + 4u > text.size()) {
          return false;
        }
        uint32_t codepoint = 0;
        for (int i = 0; i < 4; ++i) {
          const int nibble = hexNibble(text[pos++]);
          if (nibble < 0) {
            return false;
          }
          codepoint = (codepoint << 4u) | static_cast<uint32_t>(nibble);
        }
        if (!appendUtf8Codepoint(codepoint, decoded)) {
          return false;
        }
        break;
      }
      default:
        return false;
    }
  }
  return false;
}

bool extractJsonString(const std::string &text, const char *key, std::string &value) {
  const std::string quotedKey = std::string("\"") + key + "\"";
  size_t pos = text.find(quotedKey);
  if (pos == std::string::npos) {
    return false;
  }
  pos += quotedKey.size();
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  if (pos >= text.size() || text[pos] != ':') {
    return false;
  }
  ++pos;
  while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
  return jsonUnescapeString(text, pos, value);
}

std::string hexEncode(const std::string &value) {
  constexpr char digits[] = "0123456789abcdef";
  std::string encoded;
  encoded.reserve(value.size() * 2u);
  for (unsigned char ch : value) {
    encoded.push_back(digits[ch >> 4u]);
    encoded.push_back(digits[ch & 0x0fu]);
  }
  return encoded;
}

bool hexDecode(const std::string &value, std::string &decoded) {
  if ((value.size() % 2u) != 0u) {
    return false;
  }
  std::string out;
  out.reserve(value.size() / 2u);
  for (size_t i = 0; i < value.size(); i += 2u) {
    const int high = hexNibble(value[i]);
    const int low = hexNibble(value[i + 1u]);
    if (high < 0 || low < 0) {
      return false;
    }
    out.push_back(static_cast<char>((high << 4) | low));
  }
  decoded = std::move(out);
  return true;
}

bool readTextFile(const std::filesystem::path &path, std::string &text, std::string &error) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = "Could not open spektrafilm preset file for reading: " + path.string();
    return false;
  }
  text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  if (!input.good() && !input.eof()) {
    error = "Could not read spektrafilm preset file: " + path.string();
    return false;
  }
  return true;
}

bool writePresetFile(
  const std::filesystem::path &path,
  const std::string &displayName,
  const DefaultsSnapshot &snapshot,
  std::string &error
) {
  error.clear();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    error = "Could not create spektrafilm preset folder: " + path.parent_path().string();
    return false;
  }

  std::string payload = encodeDefaultsSnapshot(snapshot);
  obfuscateDefaultsText(payload);

  std::ostringstream json;
  json << "{\n";
  json << "  \"format\": \"" << kPresetFormat << "\",\n";
  json << "  \"name\": \"" << jsonEscape(displayName) << "\",\n";
  json << "  \"created\": \"" << jsonEscape(currentReadableTimestamp()) << "\",\n";
  json << "  \"plugin\": \"" << jsonEscape(pluginFlavorName()) << "\",\n";
  json << "  \"version\": \"" << jsonEscape(SPEKTRAFILM_VERSION_STRING) << "\",\n";
  json << "  \"payload_encoding\": \"obfuscated-snapshot-hex\",\n";
  json << "  \"payload_hex\": \"" << hexEncode(payload) << "\"\n";
  json << "}\n";

  const std::filesystem::path tempPath = path.string() + ".tmp";
  {
    std::ofstream output(tempPath, std::ios::binary | std::ios::trunc);
    if (!output) {
      error = "Could not open spektrafilm preset file for writing: " + tempPath.string();
      return false;
    }
    const std::string text = json.str();
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) {
      error = "Could not write spektrafilm preset file: " + tempPath.string();
      return false;
    }
  }

  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
  }
  if (ec) {
    error = "Could not replace spektrafilm preset file: " + path.string();
    return false;
  }
  return true;
}

bool readPresetSnapshot(
  const std::filesystem::path &path,
  DefaultsSnapshot &snapshot,
  std::string &displayName,
  std::string &error
) {
  error.clear();
  std::string text;
  if (!readTextFile(path, text, error)) {
    return false;
  }

  std::string format;
  if (!extractJsonString(text, "format", format) || format != kPresetFormat) {
    error = "spektrafilm preset file is not a recognized preset: " + path.string();
    return false;
  }
  if (!extractJsonString(text, "name", displayName)) {
    displayName = path.stem().string();
  }
  std::string payloadEncoding;
  if (!extractJsonString(text, "payload_encoding", payloadEncoding) || payloadEncoding != "obfuscated-snapshot-hex") {
    error = "spektrafilm preset file uses an unsupported payload encoding: " + path.string();
    return false;
  }
  std::string payloadHex;
  if (!extractJsonString(text, "payload_hex", payloadHex)) {
    error = "spektrafilm preset file is missing its snapshot payload: " + path.string();
    return false;
  }
  std::string payload;
  if (!hexDecode(payloadHex, payload)) {
    error = "spektrafilm preset file contains an invalid snapshot payload: " + path.string();
    return false;
  }
  obfuscateDefaultsText(payload);
  DefaultsSnapshot decoded;
  if (!decodeDefaultsSnapshot(payload, decoded)) {
    error = "spektrafilm preset file payload is not a recognized params snapshot: " + path.string();
    return false;
  }
  snapshot = std::move(decoded);
  return true;
}

PresetEntry presetEntryForPath(const std::filesystem::path &path) {
  PresetEntry entry{};
  entry.path = path;
  entry.displayName = path.stem().string();
  std::string text;
  std::string ignoredError;
  if (readTextFile(path, text, ignoredError)) {
    std::string name;
    if (extractJsonString(text, "name", name) && !trimString(name).empty()) {
      entry.displayName = name;
    }
    extractJsonString(text, "created", entry.created);
  }
  return entry;
}

std::vector<PresetEntry> listPresetEntries() {
  std::vector<PresetEntry> entries;
  std::error_code ec;
  const std::filesystem::path folder = presetFolder();
  if (!std::filesystem::exists(folder, ec)) {
    return entries;
  }
  for (std::filesystem::directory_iterator it(folder, ec), end; !ec && it != end; it.increment(ec)) {
    std::error_code fileEc;
    if (!it->is_regular_file(fileEc) || fileEc) {
      continue;
    }
    if (!isPresetFilePath(it->path())) {
      continue;
    }
    entries.push_back(presetEntryForPath(it->path()));
  }
  std::sort(entries.begin(), entries.end(), [](const PresetEntry &a, const PresetEntry &b) {
    const int nameCompare = a.displayName.compare(b.displayName);
    if (nameCompare != 0) {
      return nameCompare < 0;
    }
    const int createdCompare = a.created.compare(b.created);
    if (createdCompare != 0) {
      return createdCompare < 0;
    }
    return a.path.string() < b.path.string();
  });
  return entries;
}

std::vector<std::string> presetChoiceLabels(const std::vector<PresetEntry> &entries) {
  if (entries.empty()) {
    return {"No presets found"};
  }
  std::vector<std::string> labels;
  labels.reserve(entries.size());
  for (const PresetEntry &entry : entries) {
    labels.push_back(entry.displayName.empty() ? entry.path.stem().string() : entry.displayName);
  }
  return labels;
}

bool samePresetPath(const std::filesystem::path &a, const std::filesystem::path &b) {
  return a.lexically_normal().string() == b.lexically_normal().string();
}

int presetIndexForPath(const std::vector<PresetEntry> &entries, const std::filesystem::path &path) {
  for (size_t i = 0; i < entries.size(); ++i) {
    if (samePresetPath(entries[i].path, path)) {
      return static_cast<int>(i);
    }
  }
  return entries.empty() ? 0 : static_cast<int>(entries.size() - 1u);
}

void refreshPresetDropdown(InstanceData *data, const std::filesystem::path &selectedPath = {}) {
  if (!data || !data->presetSelection || !gParamHost || !gPropHost) {
    return;
  }
  const std::vector<PresetEntry> entries = listPresetEntries();
  const std::vector<std::string> labels = presetChoiceLabels(entries);
  std::vector<const char *> labelPointers;
  labelPointers.reserve(labels.size());
  for (const std::string &label : labels) {
    labelPointers.push_back(label.c_str());
  }

  OfxPropertySetHandle props = nullptr;
  if (gParamHost->paramGetPropertySet(data->presetSelection, &props) == kOfxStatOK && props) {
    gPropHost->propReset(props, kOfxParamPropChoiceOption);
    gPropHost->propSetStringN(props, kOfxParamPropChoiceOption, static_cast<int>(labelPointers.size()), labelPointers.data());
  }

  const int selected = !selectedPath.empty() ? presetIndexForPath(entries, selectedPath) : std::clamp(getIntValue(data->presetSelection, 0), 0, static_cast<int>(labels.size() - 1u));
  gParamHost->paramSetValue(data->presetSelection, selected);
}

std::string normalizedPresetName(const std::string &rawName) {
  const std::string trimmed = trimString(rawName);
  return trimmed.empty() ? "spektrafilm_preset" : trimmed;
}

std::filesystem::path generatedPresetPath(const std::string &displayName) {
  const std::filesystem::path folder = presetFolder();
  const std::string cleanName = sanitizePathComponent(displayName.empty() ? "spektrafilm_preset" : displayName);
  const std::string timestamp = currentTimestampSuffix();
  for (int attempt = 0; attempt < 64; ++attempt) {
    const std::string suffix = attempt == 0 ? timestamp : timestamp + "_" + randomExportCode();
    const std::filesystem::path path = folder / (cleanName + "_" + suffix + kPresetExtension);
    std::error_code ec;
    if (!std::filesystem::exists(path, ec)) {
      return path;
    }
  }
  return folder / (cleanName + "_" + timestamp + "_" + randomExportCode() + kPresetExtension);
}

int currentLutSize(InstanceData *data) {
  const int selected = getIntValue(data ? data->lutSize : nullptr, 1);
  return selected == 0 ? 33 : 65;
}

int currentLutDestination(InstanceData *data) {
  const int selected = getIntValue(data ? data->lutDestination : nullptr, 0);
  return std::clamp(selected, 0, 4);
}

std::string joinLabels(const std::vector<std::string> &labels) {
  std::ostringstream out;
  for (size_t i = 0; i < labels.size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << labels[i];
  }
  return out.str();
}

std::vector<std::string> lutDisabledEffectLabels(const spektrafilm::RenderParams &params) {
  std::vector<std::string> labels;
  if (params.autoExposure) {
    labels.push_back("auto exposure");
  }
  if (params.grainEnabled) {
    labels.push_back("grain");
  }
  if (params.halationEnabled && (params.scatterAmount > 0.0f || params.halationAmount > 0.0f)) {
    labels.push_back("halation");
  }
  if (params.cameraDiffusionEnabled && params.cameraDiffusionStrength > 0.0f) {
    labels.push_back("camera diffusion");
  }
  if (params.printDiffusionEnabled && params.printDiffusionStrength > 0.0f) {
    labels.push_back("print diffusion");
  }
  if (params.dirCouplersAmount > 0.0f && params.dirCouplersDiffusionUm > 0.0f) {
    labels.push_back("DIR diffusion");
  }
  if (std::abs(params.enlargerScale - 1.0f) > 1.0e-6f ||
      std::abs(params.enlargerOffsetXPercent) > 1.0e-6f ||
      std::abs(params.enlargerOffsetYPercent) > 1.0e-6f) {
    labels.push_back("film-plane transform");
  }
  if (params.scannerEnabled && params.scannerMtf50LpMm > 0.0f) {
    labels.push_back("scanner blur");
  }
  if (params.scannerEnabled && params.scannerUnsharpRadiusUm > 0.0f && params.scannerUnsharpAmount > 0.0f) {
    labels.push_back("scanner unsharp");
  }
  return labels;
}

spektrafilm::RenderParams lutSafeParams(spektrafilm::RenderParams params) {
  params.autoExposure = false;
  params.grainEnabled = false;
  params.grainModel = spektrafilm::GrainModel::Preview;
  params.grainAnimate = false;
  params.halationEnabled = false;
  params.scatterAmount = 0.0f;
  params.halationAmount = 0.0f;
  params.cameraDiffusionEnabled = false;
  params.printDiffusionEnabled = false;
  params.dirCouplersDiffusionUm = 0.0f;
  params.dirCouplersDiffusionTailUm = 0.0f;
  params.dirCouplersDiffusionTailWeight = 0.0f;
  params.enlargerScale = 1.0f;
  params.enlargerOffsetXPercent = 0.0f;
  params.enlargerOffsetYPercent = 0.0f;
  params.scannerMtf50LpMm = 0.0f;
  params.scannerUnsharpRadiusUm = 0.0f;
  params.scannerUnsharpAmount = 0.0f;
  return params;
}

bool writeCubeLut(
  const std::filesystem::path &path,
  int lutSize,
  const spektrafilm::RenderParams &sourceParams,
  const std::vector<float> &pixels,
  const std::vector<std::string> &disabledEffects,
  std::string &error
) {
  error.clear();
  std::error_code ec;
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path(), ec);
    if (ec) {
      error = "Could not create LUT export folder: " + path.parent_path().string();
      return false;
    }
  }

  const std::filesystem::path tempPath = path.string() + ".tmp";
  std::ofstream out(tempPath, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "Could not open LUT file for writing: " + tempPath.string();
    return false;
  }

  out << "TITLE \"" << pluginFlavorName() << ' ' << lutSize << "pt "
      << colorSpaceName(sourceParams.inputColorSpace) << " to "
      << colorSpaceName(sourceParams.outputColorSpace) << "\"\n";
  out << "# Generated by " << pluginFlavorName() << " OFX " << SPEKTRAFILM_VERSION_STRING << "\n";
  out << "# Process: " << processName(sourceParams.process) << "\n";
  out << "# Output role: " << outputRoleName(sourceParams.outputRole) << "\n";
  out << "# Input color space: " << colorSpaceName(sourceParams.inputColorSpace) << "\n";
  out << "# Output color space: " << colorSpaceName(sourceParams.outputColorSpace) << "\n";
  out << "# Film: " << getProfileName(spektrafilm::filmProfileCurves(sourceParams.film), "Unknown Film") << "\n";
  out << "# Paper: " << getProfileName(spektrafilm::paperProfileCurves(sourceParams.paper), "Unknown Paper") << "\n";
  if (!disabledEffects.empty()) {
    out << "# Disabled for LUT export: " << joinLabels(disabledEffects) << "\n";
  }
  out << "LUT_3D_SIZE " << lutSize << "\n";
  out << "DOMAIN_MIN 0 0 0\n";
  out << "DOMAIN_MAX 1 1 1\n";
  out << std::setprecision(9);

  const size_t sampleCount = static_cast<size_t>(lutSize) * static_cast<size_t>(lutSize) * static_cast<size_t>(lutSize);
  if (pixels.size() < sampleCount * 4u) {
    error = "Rendered LUT buffer is smaller than expected.";
    return false;
  }
  for (size_t i = 0; i < sampleCount; ++i) {
    const float r = pixels[i * 4u];
    const float g = pixels[i * 4u + 1u];
    const float b = pixels[i * 4u + 2u];
    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
      error = "Rendered LUT contains non-finite values.";
      return false;
    }
    out << r << ' ' << g << ' ' << b << '\n';
  }
  if (!out) {
    error = "Could not write LUT file: " + tempPath.string();
    return false;
  }
  out.close();

  std::filesystem::rename(tempPath, path, ec);
  if (ec) {
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tempPath, path, ec);
  }
  if (ec) {
    error = "Could not replace LUT file: " + path.string();
    return false;
  }
  return true;
}

bool exportCurrentLut(InstanceData *data, OfxTime time, std::filesystem::path &path, std::vector<std::string> &disabledEffects, std::string &error) {
  error.clear();
  disabledEffects.clear();
  if (!data) {
    error = "LUT export is not available because the renderer is not initialized.";
    return false;
  }

  const int lutSize = currentLutSize(data);
  spektrafilm::RenderParams params = readParams(data, time);
  if (params.outputRole != spektrafilm::OutputRole::DisplaySdr) {
    error = "LUT export is only available for Display Out SDR. Select an SDR output color space before exporting.";
    return false;
  }

  path = generatedLutExportPath(currentLutDestination(data), getStringValue(data->lutIdentifier));
  disabledEffects = lutDisabledEffectLabels(params);
  spektrafilm::RenderParams renderParams = lutSafeParams(params);

  const int width = lutSize * lutSize;
  const int height = lutSize;
  const size_t pixelCount = static_cast<size_t>(width) * static_cast<size_t>(height);
  std::vector<float> source(pixelCount * 4u, 1.0f);
  std::vector<float> destination(pixelCount * 4u, 0.0f);
  const float denominator = static_cast<float>(std::max(lutSize - 1, 1));
  for (int b = 0; b < lutSize; ++b) {
    for (int g = 0; g < lutSize; ++g) {
      for (int r = 0; r < lutSize; ++r) {
        const size_t index = static_cast<size_t>(b) * static_cast<size_t>(lutSize) * static_cast<size_t>(lutSize) +
          static_cast<size_t>(g) * static_cast<size_t>(lutSize) + static_cast<size_t>(r);
        source[index * 4u] = static_cast<float>(r) / denominator;
        source[index * 4u + 1u] = static_cast<float>(g) / denominator;
        source[index * 4u + 2u] = static_cast<float>(b) / denominator;
        source[index * 4u + 3u] = 1.0f;
      }
    }
  }

  spektrafilm::ImageView sourceView{};
  sourceView.data = source.data();
  sourceView.width = width;
  sourceView.height = height;
  sourceView.rowBytes = width * static_cast<int32_t>(4 * sizeof(float));
  sourceView.components = 4;
  sourceView.bytesPerComponent = 4;

  spektrafilm::MutableImageView destinationView{};
  destinationView.data = destination.data();
  destinationView.width = width;
  destinationView.height = height;
  destinationView.rowBytes = width * static_cast<int32_t>(4 * sizeof(float));
  destinationView.components = 4;
  destinationView.bytesPerComponent = 4;

  spektrafilm::RenderWindow window{0, 0, width, height};
  bool renderOk = false;
  {
    std::lock_guard<std::mutex> rendererLock(data->rendererMutex);
    spektrafilm::Renderer *renderer = ensureRenderer(data);
    if (!renderer) {
      error = "LUT export is not available because the renderer is not initialized.";
      return false;
    }
    renderOk = renderer->render(sourceView, destinationView, window, renderParams, time);
    if (!renderOk) {
      error = renderer->lastError().empty() ? "Could not render LUT samples." : renderer->lastError();
    }
  }
  if (!renderOk) {
    releaseInstanceRendererResources(data, resetRendererAfterLutExport());
    return false;
  }

  const bool wroteLut = writeCubeLut(path, lutSize, params, destination, disabledEffects, error);
  releaseInstanceRendererResources(data, resetRendererAfterLutExport());
  return wroteLut;
}

OfxStatus createInstance(OfxImageEffectHandle effect) {
  auto *data = new InstanceData();
  OfxPropertySetHandle effectProps = nullptr;
  OfxParamSetHandle paramSet = nullptr;
  gEffectHost->getPropertySet(effect, &effectProps);
  gEffectHost->getParamSet(effect, &paramSet);

  gEffectHost->clipGetHandle(effect, kOfxImageEffectSimpleSourceClipName, &data->sourceClip, nullptr);
  gEffectHost->clipGetHandle(effect, kOfxImageEffectOutputClipName, &data->outputClip, nullptr);

  cacheParam(paramSet, "filteringGroup", data->filteringGroup);
  cacheParam(paramSet, "enlargerGroup", data->enlargerGroup);
  cacheParam(paramSet, "filmGroup", data->filmGroup);
  cacheParam(paramSet, "printGroup", data->printGroup);
  cacheParam(paramSet, "couplerGroup", data->couplerGroup);
  cacheParam(paramSet, "grainGroup", data->grainGroup);
  cacheParam(paramSet, "grainSynthesisGroup", data->grainSynthesisGroup);
  cacheParam(paramSet, "halationGroup", data->halationGroup);
  cacheParam(paramSet, "process", data->process);
  cacheParam(paramSet, "scanNegativeInvert", data->scanNegativeInvert);
  cacheParam(paramSet, "rgbToRawMethod", data->rgbToRawMethod);
  cacheParam(paramSet, "inputColorSpace", data->inputColorSpace);
  cacheParam(paramSet, "rcmInputColorSpace", data->rcmInputColorSpace);
  cacheParam(paramSet, "outputRole", data->outputRole);
  cacheParam(paramSet, "sdrOutputColorSpace", data->sdrOutputColorSpace);
  cacheParam(paramSet, "sceneOutputColorSpace", data->sceneOutputColorSpace);
  cacheParam(paramSet, "hdrPreset", data->hdrPreset);
  cacheParam(paramSet, "hdrTransfer", data->hdrTransfer);
  cacheParam(paramSet, "hdrReferenceWhiteNits", data->hdrReferenceWhiteNits);
  cacheParam(paramSet, "hdrPeakNits", data->hdrPeakNits);
  cacheParam(paramSet, "hdrExposureEv", data->hdrExposureEv);
  cacheParam(paramSet, "hdrToneMapping", data->hdrToneMapping);
  cacheParam(paramSet, "colorAdaptation", data->colorAdaptation);
  cacheParam(paramSet, "colorAdaptationInputCompression", data->colorAdaptationInputCompression);
  cacheParam(paramSet, "colorAdaptationCurveSmoothing", data->colorAdaptationCurveSmoothing);
  cacheParam(paramSet, "colorAdaptationOutputLightnessCompression", data->colorAdaptationOutputLightnessCompression);
  cacheParam(paramSet, "colorAdaptationOutputChromaCompression", data->colorAdaptationOutputChromaCompression);
  cacheParam(paramSet, "cameraUvFilterEnabled", data->cameraUvFilterEnabled);
  cacheParam(paramSet, "cameraUvCutNm", data->cameraUvCutNm);
  cacheParam(paramSet, "cameraIrFilterEnabled", data->cameraIrFilterEnabled);
  cacheParam(paramSet, "cameraIrCutNm", data->cameraIrCutNm);
  cacheParam(paramSet, "film", data->film);
  cacheParam(paramSet, "paper", data->paper);
  cacheParam(paramSet, "printTiming", data->printTiming);
  cacheParam(paramSet, "filmExposureEv", data->filmExposureEv);
  cacheParam(paramSet, "autoExposure", data->autoExposure);
  cacheParam(paramSet, "autoExposureMethod", data->autoExposureMethod);
  cacheParam(paramSet, "printExposureEv", data->printExposureEv);
  cacheParam(paramSet, "filmPushPullMode", data->filmPushPullMode);
  cacheParam(paramSet, "filmPushPullStops", data->filmPushPullStops);
  cacheParam(paramSet, "printPushPullStops", data->printPushPullStops);
  cacheParam(paramSet, "negativeBleachBypassAmount", data->negativeBleachBypassAmount);
  cacheParam(paramSet, "negativeLeucoCyanCoupling", data->negativeLeucoCyanCoupling);
  cacheParam(paramSet, "printBleachBypassAmount", data->printBleachBypassAmount);
  cacheParam(paramSet, "filmGamma", data->filmGamma);
  cacheParam(paramSet, "printGamma", data->printGamma);
  cacheParam(paramSet, "printShadowShape", data->printShadowShape);
  cacheParam(paramSet, "printHighlightShape", data->printHighlightShape);
  cacheParam(paramSet, "filterC", data->filterC);
  cacheParam(paramSet, "filterMShift", data->filterMShift);
  cacheParam(paramSet, "filterYShift", data->filterYShift);
  cacheParam(paramSet, "enlargerScale", data->enlargerScale);
  cacheParam(paramSet, "enlargerOffsetXPercent", data->enlargerOffsetXPercent);
  cacheParam(paramSet, "enlargerOffsetYPercent", data->enlargerOffsetYPercent);
  cacheParam(paramSet, "preflashExposure", data->preflashExposure);
  cacheParam(paramSet, "preflashMFilterShift", data->preflashMFilterShift);
  cacheParam(paramSet, "preflashYFilterShift", data->preflashYFilterShift);
  cacheParam(paramSet, "printerLightR", data->printerLightR);
  cacheParam(paramSet, "printerLightG", data->printerLightG);
  cacheParam(paramSet, "printerLightB", data->printerLightB);
  cacheParam(paramSet, "printerLightsGang", data->printerLightsGang);
  cacheParam(paramSet, "printerLightsGroup", data->printerLightsGroup);
  cacheParam(paramSet, "printerLightCalibration", data->printerLightCalibration);
  cacheParam(paramSet, "dirAmount", data->dirAmount);
  cacheParam(paramSet, "dirDiffusionUm", data->dirDiffusionUm);
  cacheParam(paramSet, "dirDiffusionTailUm", data->dirDiffusionTailUm);
  cacheParam(paramSet, "dirDiffusionTailWeight", data->dirDiffusionTailWeight);
  cacheParam(paramSet, "dirInhibitionSameLayer", data->dirInhibitionSameLayer);
  cacheParam(paramSet, "dirInhibitionInterlayer", data->dirInhibitionInterlayer);
  cacheParam(paramSet, "dirGammaSameLayerRgb", data->dirGammaSameLayerRgb);
  cacheParam(paramSet, "dirGammaRToGb", data->dirGammaRToGb);
  cacheParam(paramSet, "dirGammaGToRb", data->dirGammaGToRb);
  cacheParam(paramSet, "dirGammaBToRg", data->dirGammaBToRg);
  cacheParam(paramSet, "dirCalibrateToStock", data->dirCalibrateToStock);
  cacheParam(paramSet, "dirUsesStockCalibration", data->dirUsesStockCalibration);
  cacheParam(paramSet, "grainEnabled", data->grainEnabled);
  cacheParam(paramSet, "grainModel", data->grainModel);
  cacheParam(paramSet, "filmFormat", data->filmFormat);
  cacheParam(paramSet, "grainAmount", data->grainAmount);
  cacheParam(paramSet, "grainSaturation", data->grainSaturation);
  cacheParam(paramSet, "grainSublayersEnabled", data->grainSublayersEnabled);
  cacheParam(paramSet, "grainSubLayerCount", data->grainSubLayerCount);
  cacheParam(paramSet, "grainParticleAreaUm2", data->grainParticleAreaUm2);
  cacheParam(paramSet, "grainParticleScale", data->grainParticleScale);
  cacheParam(paramSet, "grainParticleScaleLayers", data->grainParticleScaleLayers);
  cacheParam(paramSet, "grainDensityMin", data->grainDensityMin);
  cacheParam(paramSet, "grainUniformity", data->grainUniformity);
  cacheParam(paramSet, "grainFinalBlurUm", data->grainFinalBlurUm);
  cacheParam(paramSet, "grainBlurDyeCloudsUm", data->grainBlurDyeCloudsUm);
  cacheParam(paramSet, "grainMicroStructure", data->grainMicroStructure);
  cacheParam(paramSet, "grainSeed", data->grainSeed);
  cacheParam(paramSet, "grainAnimate", data->grainAnimate);
  cacheParam(paramSet, "grainSynthesisSize", data->grainSynthesisSize);
  cacheParam(paramSet, "grainSynthesisAmount", data->grainSynthesisAmount);
  cacheParam(paramSet, "grainSynthesisSharpness", data->grainSynthesisSharpness);
  cacheParam(paramSet, "grainSynthesisQuality", data->grainSynthesisQuality);
  cacheParam(paramSet, "grainSynthesisSamples", data->grainSynthesisSamples);
  cacheParam(paramSet, "grainSynthesisMeanRadiusUm", data->grainSynthesisMeanRadiusUm);
  cacheParam(paramSet, "grainSynthesisRadiusStdDevRatio", data->grainSynthesisRadiusStdDevRatio);
  cacheParam(paramSet, "grainSynthesisObservationSigmaUm", data->grainSynthesisObservationSigmaUm);
  cacheParam(paramSet, "grainSynthesisCellSizeRatio", data->grainSynthesisCellSizeRatio);
  cacheParam(paramSet, "grainSynthesisMaxRadiusQuantile", data->grainSynthesisMaxRadiusQuantile);
  cacheParam(paramSet, "grainSynthesisCoverageEpsilon", data->grainSynthesisCoverageEpsilon);
  cacheParam(paramSet, "grainSynthesisMaxGrainsPerCell", data->grainSynthesisMaxGrainsPerCell);
  cacheParam(paramSet, "grainSynthesisRadiusScale", data->grainSynthesisRadiusScale);
  cacheParam(paramSet, "grainSynthesisLayerScale", data->grainSynthesisLayerScale);
  cacheParam(paramSet, "grainSynthesisLayered", data->grainSynthesisLayered);
  cacheParam(paramSet, "halationEnabled", data->halationEnabled);
  cacheParam(paramSet, "scatterAmount", data->scatterAmount);
  cacheParam(paramSet, "scatterScale", data->scatterScale);
  cacheParam(paramSet, "halationAmount", data->halationAmount);
  cacheParam(paramSet, "halationScale", data->halationScale);
  cacheParam(paramSet, "halationStrength", data->halationStrength);
  cacheParam(paramSet, "halationBoostEv", data->halationBoostEv);
  cacheParam(paramSet, "halationBoostRange", data->halationBoostRange);
  cacheParam(paramSet, "halationProtectEv", data->halationProtectEv);
  cacheParam(paramSet, "cameraDiffusionEnabled", data->cameraDiffusionEnabled);
  cacheParam(paramSet, "cameraDiffusionFamily", data->cameraDiffusionFamily);
  cacheParam(paramSet, "cameraDiffusionStrength", data->cameraDiffusionStrength);
  cacheParam(paramSet, "cameraDiffusionSpatialScale", data->cameraDiffusionSpatialScale);
  cacheParam(paramSet, "cameraDiffusionHaloWarmth", data->cameraDiffusionHaloWarmth);
  cacheParam(paramSet, "cameraDiffusionCoreIntensity", data->cameraDiffusionCoreIntensity);
  cacheParam(paramSet, "cameraDiffusionCoreSize", data->cameraDiffusionCoreSize);
  cacheParam(paramSet, "cameraDiffusionHaloIntensity", data->cameraDiffusionHaloIntensity);
  cacheParam(paramSet, "cameraDiffusionHaloSize", data->cameraDiffusionHaloSize);
  cacheParam(paramSet, "cameraDiffusionBloomIntensity", data->cameraDiffusionBloomIntensity);
  cacheParam(paramSet, "cameraDiffusionBloomSize", data->cameraDiffusionBloomSize);
  cacheParam(paramSet, "printDiffusionEnabled", data->printDiffusionEnabled);
  cacheParam(paramSet, "printDiffusionFamily", data->printDiffusionFamily);
  cacheParam(paramSet, "printDiffusionStrength", data->printDiffusionStrength);
  cacheParam(paramSet, "printDiffusionSpatialScale", data->printDiffusionSpatialScale);
  cacheParam(paramSet, "printDiffusionHaloWarmth", data->printDiffusionHaloWarmth);
  cacheParam(paramSet, "printDiffusionCoreIntensity", data->printDiffusionCoreIntensity);
  cacheParam(paramSet, "printDiffusionCoreSize", data->printDiffusionCoreSize);
  cacheParam(paramSet, "printDiffusionHaloIntensity", data->printDiffusionHaloIntensity);
  cacheParam(paramSet, "printDiffusionHaloSize", data->printDiffusionHaloSize);
  cacheParam(paramSet, "printDiffusionBloomIntensity", data->printDiffusionBloomIntensity);
  cacheParam(paramSet, "printDiffusionBloomSize", data->printDiffusionBloomSize);
  cacheParam(paramSet, "scannerGroup", data->scannerGroup);
  cacheParam(paramSet, "scannerEnabled", data->scannerEnabled);
  cacheParam(paramSet, "scannerWhiteCorrection", data->scannerWhiteCorrection);
  cacheParam(paramSet, "scannerBlackCorrection", data->scannerBlackCorrection);
  cacheParam(paramSet, "scannerWhiteLevel", data->scannerWhiteLevel);
  cacheParam(paramSet, "scannerBlackLevel", data->scannerBlackLevel);
  cacheParam(paramSet, "glarePercent", data->glarePercent);
  cacheParam(paramSet, "glareRoughness", data->glareRoughness);
  cacheParam(paramSet, "glareBlur", data->glareBlur);
  cacheParam(paramSet, "scannerMtf50LpMm", data->scannerMtf50LpMm);
  cacheParam(paramSet, "scannerUnsharpRadiusUm", data->scannerUnsharpRadiusUm);
  cacheParam(paramSet, "scannerUnsharpAmount", data->scannerUnsharpAmount);
  cacheParam(paramSet, "gpuRenderTiling", data->gpuRenderTiling);
  cacheParam(paramSet, "lutSize", data->lutSize);
  cacheParam(paramSet, "lutDestination", data->lutDestination);
  cacheParam(paramSet, "lutIdentifier", data->lutIdentifier);
  cacheParam(paramSet, "exportLut", data->exportLut);
  cacheParam(paramSet, "presetName", data->presetName);
  cacheParam(paramSet, "presetSelection", data->presetSelection);
  cacheParam(paramSet, "savePreset", data->savePreset);
  cacheParam(paramSet, "loadPreset", data->loadPreset);
  refreshPresetDropdown(data);

  DefaultsSnapshot savedDefaults;
  bool defaultsFound = false;
  std::string defaultsError;
  if (loadDefaultsFromFile(savedDefaults, defaultsFound, defaultsError) && defaultsFound) {
    applySnapshotToParamSet(paramSet, savedDefaults, false);
  }
  randomizeGrainSeedForNewInstance(data);
  if (dirUsesStockCalibration(data)) {
    applyDirStockCalibration(data, false);
  }

  {
    std::ostringstream log;
    log << "createInstance version=" << SPEKTRAFILM_VERSION_STRING
        << " flavor=" << pluginFlavorName();
    appendHostLog(log.str(), true);
  }
  rememberCurrentPrinterLights(data);
  syncConditionalParamVisibility(data);
  gPropHost->propSetPointer(effectProps, kOfxPropInstanceData, 0, data);
  return kOfxStatOK;
}

OfxStatus destroyInstance(OfxImageEffectHandle effect) {
  InstanceData *data = getInstanceData(effect);
  delete data;
  return kOfxStatOK;
}

void releaseInstanceRendererResources(InstanceData *data, bool resetRenderer) {
  if (!data) {
    return;
  }
  std::lock_guard<std::mutex> rendererLock(data->rendererMutex);
  if (data->renderer) {
    if (resetRenderer) {
      data->renderer.reset();
    } else {
      data->renderer->releaseTransientResources();
    }
  }
}

OfxStatus releaseInactiveInstanceResources(OfxImageEffectHandle effect, bool resetRenderer) {
  if (!effect) {
    return kOfxStatOK;
  }
  releaseInstanceRendererResources(getInstanceData(effect), resetRenderer);
  return kOfxStatOK;
}

OfxStatus instanceChanged(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs) {
  InstanceData *data = getInstanceData(effect);
  if (!data) {
    return kOfxStatReplyDefault;
  }
  char *changedName = nullptr;
  char *changeReason = nullptr;
  if (inArgs) {
    gPropHost->propGetString(inArgs, kOfxPropName, 0, &changedName);
    gPropHost->propGetString(inArgs, kOfxPropChangeReason, 0, &changeReason);
  }
  if (changeReason && std::strcmp(changeReason, kOfxChangePluginEdited) == 0) {
    return kOfxStatReplyDefault;
  }

  const bool copyParamsChanged = changedName && std::strcmp(changedName, "copyParams") == 0;
  const bool pasteParamsChanged = changedName && std::strcmp(changedName, "pasteParams") == 0;
  const bool saveDefaultsChanged = changedName && std::strcmp(changedName, "saveDefaults") == 0;
  const bool resetDefaultsChanged = changedName && std::strcmp(changedName, "resetDefaults") == 0;
  const bool exportLutChanged = changedName && std::strcmp(changedName, "exportLut") == 0;
  const bool savePresetChanged = changedName && std::strcmp(changedName, "savePreset") == 0;
  const bool loadPresetChanged = changedName && std::strcmp(changedName, "loadPreset") == 0;
  const bool openManualChanged = changedName && std::strcmp(changedName, "openUserManual") == 0;
  if (copyParamsChanged || pasteParamsChanged || saveDefaultsChanged || resetDefaultsChanged ||
      exportLutChanged || savePresetChanged || loadPresetChanged || openManualChanged) {
    OfxParamSetHandle paramSet = nullptr;
    gEffectHost->getParamSet(effect, &paramSet);
    OfxTime time = 0.0;
    if (inArgs) {
      gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
    }
    if (copyParamsChanged) {
      std::string error;
      if (copyVisibleParams(paramSet, time, error)) {
        return kOfxStatOK;
      }
      showMessage(effect, kOfxMessageError, "spektrafilmDefaults", error.empty() ? "Could not copy spektrafilm params." : error);
      return kOfxStatFailed;
    }
    if (pasteParamsChanged) {
      DefaultsSnapshot copiedParams;
      bool copyFound = false;
      std::string error;
      std::string clipboardText;
      if (!readTextFromClipboard(clipboardText, copyFound, error)) {
        showMessage(effect, kOfxMessageError, "spektrafilmDefaults", error.empty() ? "Could not read copied spektrafilm params." : error);
        return kOfxStatFailed;
      }
      if (!copyFound) {
        showMessage(effect, kOfxMessageWarning, "spektrafilmDefaults", "No copied spektrafilm params found.");
        return kOfxStatReplyDefault;
      }
      obfuscateDefaultsText(clipboardText);
      if (!decodeDefaultsSnapshot(clipboardText, copiedParams)) {
        showMessage(effect, kOfxMessageWarning, "spektrafilmDefaults", "Clipboard does not contain spektrafilm params.");
        return kOfxStatReplyDefault;
      }
      gParamHost->paramEditBegin(paramSet, "Paste spektrafilm params");
      applySnapshotToParamSet(paramSet, copiedParams);
      gParamHost->paramEditEnd(paramSet);
      syncConditionalParamVisibility(data);
      return kOfxStatOK;
    }
    if (saveDefaultsChanged) {
      std::string error;
      if (saveVisibleDefaults(paramSet, time, error)) {
        showMessage(effect, kOfxMessageMessage, "spektrafilmDefaults", "spektrafilm defaults saved successfully.");
        return kOfxStatOK;
      }
      showMessage(effect, kOfxMessageError, "spektrafilmDefaults", error.empty() ? "Could not save spektrafilm defaults." : error);
      return kOfxStatFailed;
    }
    if (savePresetChanged) {
      DefaultsSnapshot snapshot;
      captureParamSetSnapshot(paramSet, time, snapshot);
      const std::string displayName = normalizedPresetName(getStringValue(data->presetName));
      const std::filesystem::path path = generatedPresetPath(displayName);
      std::string error;
      if (!writePresetFile(path, displayName, snapshot, error)) {
        showMessage(effect, kOfxMessageError, "spektrafilmPreset", error.empty() ? "Could not save spektrafilm preset." : error);
        return kOfxStatFailed;
      }
      if (data->presetName) {
        gParamHost->paramSetValue(data->presetName, displayName.c_str());
      }
      refreshPresetDropdown(data, path);
      showMessage(
        effect,
        kOfxMessageMessage,
        "spektrafilmPreset",
        "spektrafilm preset saved successfully: " + displayName + "\nPreset folder: " + presetFolder().string()
      );
      return kOfxStatOK;
    }
    if (loadPresetChanged) {
      const std::vector<PresetEntry> entries = listPresetEntries();
      if (entries.empty()) {
        showMessage(effect, kOfxMessageWarning, "spektrafilmPreset", "No spektrafilm presets found.\nPreset folder: " + presetFolder().string());
        refreshPresetDropdown(data);
        return kOfxStatReplyDefault;
      }
      const int selected = getIntValue(data->presetSelection, 0);
      if (selected < 0 || selected >= static_cast<int>(entries.size())) {
        showMessage(effect, kOfxMessageWarning, "spektrafilmPreset", "Selected spektrafilm preset is no longer available.");
        refreshPresetDropdown(data);
        return kOfxStatReplyDefault;
      }

      DefaultsSnapshot presetSnapshot;
      std::string displayName;
      std::string error;
      if (!readPresetSnapshot(entries[static_cast<size_t>(selected)].path, presetSnapshot, displayName, error)) {
        showMessage(effect, kOfxMessageError, "spektrafilmPreset", error.empty() ? "Could not load spektrafilm preset." : error);
        refreshPresetDropdown(data);
        return kOfxStatFailed;
      }
      gParamHost->paramEditBegin(paramSet, "Load spektrafilm preset");
      applySnapshotToParamSet(paramSet, presetSnapshot);
      if (data->presetName && !displayName.empty()) {
        gParamHost->paramSetValue(data->presetName, displayName.c_str());
      }
      gParamHost->paramEditEnd(paramSet);
      syncConditionalParamVisibility(data);
      refreshPresetDropdown(data, entries[static_cast<size_t>(selected)].path);
      showMessage(effect, kOfxMessageMessage, "spektrafilmPreset", "spektrafilm preset loaded: " + (displayName.empty() ? entries[static_cast<size_t>(selected)].displayName : displayName));
      return kOfxStatOK;
    }
    if (exportLutChanged) {
      std::filesystem::path path;
      std::vector<std::string> disabledEffects;
      std::string error;
      if (!exportCurrentLut(data, time, path, disabledEffects, error)) {
        showMessage(effect, kOfxMessageError, "spektrafilmLutExport", error.empty() ? "Could not export spektrafilm LUT." : error);
        return kOfxStatFailed;
      }
      std::string message = "spektrafilm LUT exported: " + path.string();
      if (!disabledEffects.empty()) {
        message += "\n\nThis LUT contains the color-only spectral transform. Disabled for export: " + joinLabels(disabledEffects) + ".";
        showMessage(effect, kOfxMessageWarning, "spektrafilmLutExport", message);
      } else {
        showMessage(effect, kOfxMessageMessage, "spektrafilmLutExport", message);
      }
      return kOfxStatOK;
    }
    if (openManualChanged) {
      std::string error;
      if (openBundledUserManual(error)) {
        return kOfxStatOK;
      }
      showMessage(effect, kOfxMessageError, "spektrafilmUserManual", error.empty() ? "Could not open the spektrafilm user manual." : error);
      return kOfxStatFailed;
    }
    std::string error;
    if (!deleteDefaultsFile(error)) {
      showMessage(
        effect,
        kOfxMessageError,
        "spektrafilmDefaults",
        error.empty() ? "Could not delete spektrafilm defaults file." : error
      );
      return kOfxStatFailed;
    }
    gParamHost->paramEditBegin(paramSet, "Reset spektrafilm factory defaults");
    resetParamSetToFactory(paramSet);
    if (dirUsesStockCalibration(data)) {
      applyDirStockCalibration(data, false);
    }
    gParamHost->paramEditEnd(paramSet);
    syncConditionalParamVisibility(data);
    showMessage(effect, kOfxMessageMessage, "spektrafilmDefaults", "spektrafilm factory defaults restored.");
    return kOfxStatOK;
  }

  syncConditionalParamVisibility(data);

  const bool hdrPresetChanged = changedName && std::strcmp(changedName, "hdrPreset") == 0;
  const bool hdrControlChanged = changedName && (
    std::strcmp(changedName, "hdrTransfer") == 0 ||
    std::strcmp(changedName, "hdrReferenceWhiteNits") == 0 ||
    std::strcmp(changedName, "hdrPeakNits") == 0 ||
    std::strcmp(changedName, "hdrExposureEv") == 0 ||
    std::strcmp(changedName, "hdrToneMapping") == 0
  );
  if (hdrPresetChanged && data->hdrPreset && data->hdrTransfer && data->hdrReferenceWhiteNits &&
      data->hdrPeakNits && data->hdrToneMapping) {
    int preset = 0;
    gParamHost->paramGetValue(data->hdrPreset, &preset);
    if (preset != static_cast<int>(spektrafilm::HdrPreset::Custom)) {
      const HdrPresetValues values = hdrPresetValues(preset);
      gParamHost->paramSetValue(data->hdrTransfer, values.transfer);
      gParamHost->paramSetValue(data->hdrReferenceWhiteNits, values.referenceWhiteNits);
      gParamHost->paramSetValue(data->hdrPeakNits, values.peakNits);
      gParamHost->paramSetValue(data->hdrToneMapping, values.toneMapping);
      return kOfxStatOK;
    }
  } else if (hdrControlChanged && data->hdrPreset) {
    int preset = 0;
    gParamHost->paramGetValue(data->hdrPreset, &preset);
    if (preset != static_cast<int>(spektrafilm::HdrPreset::Custom)) {
      gParamHost->paramSetValue(data->hdrPreset, static_cast<int>(spektrafilm::HdrPreset::Custom));
      return kOfxStatOK;
    }
  }

  const bool filmChanged = changedName && std::strcmp(changedName, "film") == 0;
  const bool dirCalibrateChanged = changedName && std::strcmp(changedName, "dirCalibrateToStock") == 0;
  const bool dirCoefficientChanged = changedName && (
    std::strcmp(changedName, "dirGammaSameLayerRgb") == 0 ||
    std::strcmp(changedName, "dirGammaRToGb") == 0 ||
    std::strcmp(changedName, "dirGammaGToRb") == 0 ||
    std::strcmp(changedName, "dirGammaBToRg") == 0
  );
  if (dirCalibrateChanged) {
    return applyDirStockCalibration(data, true) ? kOfxStatOK : kOfxStatReplyDefault;
  }
  if (dirCoefficientChanged && data->dirUsesStockCalibration && !data->syncingDirCalibration) {
    gParamHost->paramSetValue(data->dirUsesStockCalibration, 0);
    return kOfxStatOK;
  }
  if (filmChanged && dirUsesStockCalibration(data)) {
    return applyDirStockCalibration(data, false) ? kOfxStatOK : kOfxStatReplyDefault;
  }

  if (!data->printerLightR || !data->printerLightG || !data->printerLightB ||
      !data->printerLightsGang || !data->printerLightsGroup) {
    return kOfxStatReplyDefault;
  }
  if (data->syncingPrinterLights) {
    return kOfxStatOK;
  }
  const bool printerLightRChanged = changedName && std::strcmp(changedName, "printerLightR") == 0;
  const bool printerLightGChanged = changedName && std::strcmp(changedName, "printerLightG") == 0;
  const bool printerLightBChanged = changedName && std::strcmp(changedName, "printerLightB") == 0;
  const bool printerLightsChanged = printerLightRChanged || printerLightGChanged || printerLightBChanged;
  const bool gangChanged = changedName && std::strcmp(changedName, "printerLightsGang") == 0;
  const bool groupChanged = changedName && std::strcmp(changedName, "printerLightsGroup") == 0;
  if (!printerLightsChanged && !gangChanged && !groupChanged) {
    return kOfxStatReplyDefault;
  }

  double current[3] = {0.0, 0.0, 0.0};
  if (!readCurrentPrinterLights(data, current)) {
    return kOfxStatReplyDefault;
  }

  bool gangEnabled = getBoolValue(data->printerLightsGang, false);
  bool groupEnabled = getBoolValue(data->printerLightsGroup, false);
  if (gangChanged && gangEnabled && groupEnabled) {
    data->syncingPrinterLights = true;
    gParamHost->paramSetValue(data->printerLightsGroup, 0);
    data->syncingPrinterLights = false;
    groupEnabled = false;
  } else if (groupChanged && groupEnabled && gangEnabled) {
    data->syncingPrinterLights = true;
    gParamHost->paramSetValue(data->printerLightsGang, 0);
    data->syncingPrinterLights = false;
    gangEnabled = false;
  }

  if (!gangEnabled && !groupEnabled) {
    rememberCurrentPrinterLights(data, current);
    return kOfxStatReplyDefault;
  }

  if (gangEnabled) {
    double linkedValue = (current[0] + current[1] + current[2]) / 3.0;
    if (printerLightRChanged) {
      linkedValue = current[0];
    } else if (printerLightGChanged) {
      linkedValue = current[1];
    } else if (printerLightBChanged) {
      linkedValue = current[2];
    }

    double linked[3] = {linkedValue, linkedValue, linkedValue};
    rememberCurrentPrinterLights(data, linked);
    data->syncingPrinterLights = true;
    if (std::abs(current[0] - linkedValue) > 1.0e-9) {
      gParamHost->paramSetValue(data->printerLightR, linkedValue);
    }
    if (std::abs(current[1] - linkedValue) > 1.0e-9) {
      gParamHost->paramSetValue(data->printerLightG, linkedValue);
    }
    if (std::abs(current[2] - linkedValue) > 1.0e-9) {
      gParamHost->paramSetValue(data->printerLightB, linkedValue);
    }
    data->syncingPrinterLights = false;
    return kOfxStatOK;
  }

  if (!printerLightsChanged || !data->lastPrinterLightsInitialized) {
    rememberCurrentPrinterLights(data, current);
    return kOfxStatReplyDefault;
  }

  int changedIndex = 0;
  if (printerLightGChanged) {
    changedIndex = 1;
  } else if (printerLightBChanged) {
    changedIndex = 2;
  }
  const double delta = current[changedIndex] - data->lastPrinterLights[changedIndex];
  if (std::abs(delta) <= 1.0e-9) {
    rememberCurrentPrinterLights(data, current);
    return kOfxStatReplyDefault;
  }

  double grouped[3] = {
    std::clamp(data->lastPrinterLights[0] + delta, -24.0, 24.0),
    std::clamp(data->lastPrinterLights[1] + delta, -24.0, 24.0),
    std::clamp(data->lastPrinterLights[2] + delta, -24.0, 24.0),
  };
  grouped[changedIndex] = current[changedIndex];
  rememberCurrentPrinterLights(data, grouped);
  data->syncingPrinterLights = true;
  if (std::abs(current[0] - grouped[0]) > 1.0e-9) {
    gParamHost->paramSetValue(data->printerLightR, grouped[0]);
  }
  if (std::abs(current[1] - grouped[1]) > 1.0e-9) {
    gParamHost->paramSetValue(data->printerLightG, grouped[1]);
  }
  if (std::abs(current[2] - grouped[2]) > 1.0e-9) {
    gParamHost->paramSetValue(data->printerLightB, grouped[2]);
  }
  data->syncingPrinterLights = false;
  return kOfxStatOK;
}

double filmFormatMm(spektrafilm::FilmFormat format) {
  switch (format) {
    case spektrafilm::FilmFormat::Standard8:
      return 4.8;
    case spektrafilm::FilmFormat::Super8:
      return 5.79;
    case spektrafilm::FilmFormat::Standard16:
      return 10.26;
    case spektrafilm::FilmFormat::Super16:
      return 12.52;
    case spektrafilm::FilmFormat::Super35:
      return 24.89;
    case spektrafilm::FilmFormat::Standard65:
      return 52.48;
    case spektrafilm::FilmFormat::Imax70:
      return 70.41;
    case spektrafilm::FilmFormat::Standard35:
    default:
      return 35.0;
  }
}

double grainFinalBlurFormatScale(spektrafilm::FilmFormat format) {
  return std::pow(std::max(filmFormatMm(format) / 35.0, 1.0e-6), 0.62);
}

double effectiveGrainFinalBlurUm(const spektrafilm::RenderParams &params) {
  return std::max(static_cast<double>(params.grainFinalBlurUm), 0.0) *
    grainFinalBlurFormatScale(params.filmFormat);
}

double enlargerScale(const spektrafilm::RenderParams &params) {
  return std::clamp(static_cast<double>(params.enlargerScale), 1.0, 32.0);
}

bool enlargerTransformActive(const spektrafilm::RenderParams &params) {
  return std::abs(enlargerScale(params) - 1.0) > 1.0e-6;
}

double rectWidth(const OfxRectD &rect) {
  return std::max(rect.x2 - rect.x1, 0.0);
}

double rectHeight(const OfxRectD &rect) {
  return std::max(rect.y2 - rect.y1, 0.0);
}

double sourceLongEdgePixels(InstanceData *data, OfxTime time, const OfxRectD &roi) {
  double longEdge = std::max({rectWidth(roi), rectHeight(roi), 1.0});
  if (data && data->sourceClip && gEffectHost) {
    OfxRectD sourceRod{};
    if (gEffectHost->clipGetRegionOfDefinition(data->sourceClip, time, &sourceRod) == kOfxStatOK) {
      longEdge = std::max({longEdge, rectWidth(sourceRod), rectHeight(sourceRod), 1.0});
    }
  }
  return longEdge;
}

double pixelSizeUmForRender(InstanceData *data, OfxTime time, const spektrafilm::RenderParams &params, const OfxRectD &roi) {
  const double formatLongEdgeMm = filmFormatMm(params.filmFormat);
  const double scale = enlargerScale(params);
  return formatLongEdgeMm * 1000.0 / sourceLongEdgePixels(data, time, roi) / scale;
}

double normalQuantile(double p) {
  p = std::clamp(p, 1.0e-9, 1.0 - 1.0e-9);
  constexpr double a1 = -3.969683028665376e+01;
  constexpr double a2 = 2.209460984245205e+02;
  constexpr double a3 = -2.759285104469687e+02;
  constexpr double a4 = 1.383577518672690e+02;
  constexpr double a5 = -3.066479806614716e+01;
  constexpr double a6 = 2.506628277459239e+00;
  constexpr double b1 = -5.447609879822406e+01;
  constexpr double b2 = 1.615858368580409e+02;
  constexpr double b3 = -1.556989798598866e+02;
  constexpr double b4 = 6.680131188771972e+01;
  constexpr double b5 = -1.328068155288572e+01;
  constexpr double c1 = -7.784894002430293e-03;
  constexpr double c2 = -3.223964580411365e-01;
  constexpr double c3 = -2.400758277161838e+00;
  constexpr double c4 = -2.549732539343734e+00;
  constexpr double c5 = 4.374664141464968e+00;
  constexpr double c6 = 2.938163982698783e+00;
  constexpr double d1 = 7.784695709041462e-03;
  constexpr double d2 = 3.224671290700398e-01;
  constexpr double d3 = 2.445134137142996e+00;
  constexpr double d4 = 3.754408661907416e+00;
  constexpr double pLow = 0.02425;
  constexpr double pHigh = 1.0 - pLow;
  if (p < pLow) {
    const double q = std::sqrt(-2.0 * std::log(p));
    return (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  if (p > pHigh) {
    const double q = std::sqrt(-2.0 * std::log(1.0 - p));
    return -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) /
      ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
  }
  const double q = p - 0.5;
  const double r = q * q;
  return (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q /
    (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
}

double grainSynthesisMaxRadiusUm(const spektrafilm::RenderParams &params) {
  const double mean = std::max(
    static_cast<double>(params.grainSynthesisMeanRadiusUm) *
      std::clamp(static_cast<double>(params.grainSynthesisSize), 0.25, 4.0),
    1.0e-6
  );
  const double ratio = std::max(static_cast<double>(params.grainSynthesisRadiusStdDevRatio), 0.0);
  double radius = mean;
  if (ratio > 1.0e-6) {
    const double varianceRatio = ratio * ratio;
    const double sigma = std::sqrt(std::log(1.0 + varianceRatio));
    const double mu = std::log(mean) - 0.5 * sigma * sigma;
    radius = std::exp(mu + sigma * normalQuantile(params.grainSynthesisMaxRadiusQuantile));
  }
  const double channelScale = std::max({
    static_cast<double>(params.grainSynthesisRadiusScaleR),
    static_cast<double>(params.grainSynthesisRadiusScaleG),
    static_cast<double>(params.grainSynthesisRadiusScaleB),
    1.0e-6
  });
  const double layerScale = params.grainSynthesisLayered
    ? std::max({
        static_cast<double>(params.grainSynthesisLayerScale0),
        static_cast<double>(params.grainSynthesisLayerScale1),
        static_cast<double>(params.grainSynthesisLayerScale2),
        1.0e-6
      })
    : 1.0;
  return radius * channelScale * layerScale;
}

double grainSynthesisObservationSigmaUm(const spektrafilm::RenderParams &params) {
  return std::max(static_cast<double>(params.grainSynthesisObservationSigmaUm), 0.0) /
    std::max(static_cast<double>(params.grainSynthesisSharpness), 0.25);
}

double scannerSigmaUmFromMtf50(double mtf50LpMm) {
  if (!std::isfinite(mtf50LpMm) || mtf50LpMm <= 0.0) {
    return 0.0;
  }
  constexpr double kPi = 3.14159265358979323846;
  return 1000.0 * std::sqrt(std::log(2.0) / (2.0 * kPi * kPi)) / mtf50LpMm;
}

int clampedKernelRadius(double sigmaPixels, int cap) {
  if (!std::isfinite(sigmaPixels) || sigmaPixels <= 0.0) {
    return 0;
  }
  return std::clamp(static_cast<int>(std::ceil(3.0 * sigmaPixels)), 0, cap);
}

double diffusionGroupMaxLambdaUm(spektrafilm::DiffusionFilterFamily family, int group, double size) {
  double lambdaUm = 0.0;
  double spread = 1.0;
  switch (family) {
    case spektrafilm::DiffusionFilterFamily::Glimmerglass:
      lambdaUm = group == 0 ? 10.0 : group == 1 ? 50.0 : 260.0;
      break;
    case spektrafilm::DiffusionFilterFamily::ProMist:
      lambdaUm = group == 0 ? 14.0 : group == 1 ? 150.0 : 650.0;
      break;
    case spektrafilm::DiffusionFilterFamily::CineBloom:
      lambdaUm = group == 0 ? 20.0 : group == 1 ? 200.0 : 1000.0;
      break;
    case spektrafilm::DiffusionFilterFamily::BlackProMist:
    default:
      lambdaUm = group == 0 ? 16.0 : group == 1 ? 95.0 : 380.0;
      break;
  }
  spread = group == 0 ? 1.5 : group == 1 ? 2.0 : 2.5;
  return lambdaUm * spread * std::max(size, 1.0e-6);
}

int diffusionRadiusPixels(
  spektrafilm::DiffusionFilterFamily family,
  double strength,
  double spatialScale,
  double coreIntensity,
  double coreSize,
  double haloIntensity,
  double haloSize,
  double bloomIntensity,
  double bloomSize,
  double pixelSizeUm
) {
  if (strength <= 0.0 || spatialScale <= 0.0 || pixelSizeUm <= 0.0) {
    return 0;
  }
  double maxLambdaUm = 0.0;
  if (coreIntensity > 0.0) {
    maxLambdaUm = std::max(maxLambdaUm, diffusionGroupMaxLambdaUm(family, 0, coreSize));
  }
  if (haloIntensity > 0.0) {
    maxLambdaUm = std::max(maxLambdaUm, diffusionGroupMaxLambdaUm(family, 1, haloSize));
  }
  if (bloomIntensity > 0.0) {
    maxLambdaUm = std::max(maxLambdaUm, diffusionGroupMaxLambdaUm(family, 2, bloomSize));
  }
  constexpr double kMaxExpGaussianFitSigmaScale = 2.7684;
  return clampedKernelRadius(maxLambdaUm * kMaxExpGaussianFitSigmaScale * spatialScale / pixelSizeUm, 256);
}

int halationRadiusPixels(const spektrafilm::RenderParams &params, double pixelSizeUm) {
  if (!params.halationEnabled || pixelSizeUm <= 0.0) {
    return 0;
  }
  double maxSigmaUm = 0.0;
  if (params.scatterAmount > 0.0f && params.scatterScale > 0.0f) {
    constexpr double kMaxScatterTailUm = 9.7;
    constexpr double kMaxExpGaussianFitSigmaScale = 2.7684;
    maxSigmaUm = std::max(maxSigmaUm, kMaxScatterTailUm * kMaxExpGaussianFitSigmaScale * params.scatterScale);
  }
  if (params.halationAmount > 0.0f && params.halationScale > 0.0f) {
    constexpr double kMaxProfileFirstSigmaUm = 65.0;
    constexpr double kThirdBounceSigmaScale = 1.7320508075688772;
    maxSigmaUm = std::max(maxSigmaUm, kMaxProfileFirstSigmaUm * kThirdBounceSigmaScale * params.halationScale);
  }
  return clampedKernelRadius(maxSigmaUm / pixelSizeUm, 256);
}

int dirRadiusPixels(const spektrafilm::RenderParams &params, double pixelSizeUm) {
  if (params.dirCouplersAmount <= 0.0f || params.dirCouplersDiffusionUm <= 0.0f || pixelSizeUm <= 0.0) {
    return 0;
  }
  double maxSigmaUm = params.dirCouplersDiffusionUm;
  if (params.dirCouplersDiffusionTailUm > 0.0f && params.dirCouplersDiffusionTailWeight > 0.0f) {
    constexpr double kMaxExpGaussianFitSigmaScale = 2.7684;
    maxSigmaUm = std::max(maxSigmaUm, static_cast<double>(params.dirCouplersDiffusionTailUm) * kMaxExpGaussianFitSigmaScale);
  }
  return clampedKernelRadius(maxSigmaUm / pixelSizeUm, 256);
}

int grainRadiusPixels(const spektrafilm::RenderParams &params, double pixelSizeUm) {
  if (!params.grainEnabled) {
    return 0;
  }
  if (params.grainModel == spektrafilm::GrainModel::GrainSynthesis && pixelSizeUm > 0.0) {
    const double supportUm = grainSynthesisMaxRadiusUm(params) +
      3.0 * grainSynthesisObservationSigmaUm(params);
    return clampedKernelRadius(std::max(supportUm, effectiveGrainFinalBlurUm(params)) / pixelSizeUm, 256);
  }
  double maxSigmaPixels = pixelSizeUm > 0.0
    ? effectiveGrainFinalBlurUm(params) / pixelSizeUm
    : 0.0;
  if (params.grainModel == spektrafilm::GrainModel::Production && pixelSizeUm > 0.0) {
    maxSigmaPixels = std::max(maxSigmaPixels, static_cast<double>(params.grainBlurDyeCloudsUm) / pixelSizeUm);
    maxSigmaPixels = std::max(maxSigmaPixels, static_cast<double>(params.grainMicroStructureScale) / pixelSizeUm);
  }
  return clampedKernelRadius(maxSigmaPixels, 64);
}

int scannerRadiusPixels(const spektrafilm::RenderParams &params, double pixelSizeUm) {
  if (!params.scannerEnabled || pixelSizeUm <= 0.0) {
    return 0;
  }
  return std::max(
    clampedKernelRadius(scannerSigmaUmFromMtf50(params.scannerMtf50LpMm) / pixelSizeUm, 256),
    clampedKernelRadius(std::max(static_cast<double>(params.scannerUnsharpRadiusUm), 0.0) / pixelSizeUm, 256)
  );
}

int estimateSourceExpansionPixels(InstanceData *data, OfxTime time, const spektrafilm::RenderParams &params, const OfxRectD &roi) {
  const double pixelSizeUm = pixelSizeUmForRender(data, time, params, roi);
  int radius = 0;
  radius = std::max(radius, halationRadiusPixels(params, pixelSizeUm));
  radius = std::max(radius, dirRadiusPixels(params, pixelSizeUm));
  radius = std::max(radius, grainRadiusPixels(params, pixelSizeUm));
  radius = std::max(radius, scannerRadiusPixels(params, pixelSizeUm));
  if (params.cameraDiffusionEnabled) {
    radius = std::max(radius, diffusionRadiusPixels(
      params.cameraDiffusionFamily,
      params.cameraDiffusionStrength,
      params.cameraDiffusionSpatialScale,
      params.cameraDiffusionCoreIntensity,
      params.cameraDiffusionCoreSize,
      params.cameraDiffusionHaloIntensity,
      params.cameraDiffusionHaloSize,
      params.cameraDiffusionBloomIntensity,
      params.cameraDiffusionBloomSize,
      pixelSizeUm
    ));
  }
  if (params.printDiffusionEnabled) {
    radius = std::max(radius, diffusionRadiusPixels(
      params.printDiffusionFamily,
      params.printDiffusionStrength,
      params.printDiffusionSpatialScale,
      params.printDiffusionCoreIntensity,
      params.printDiffusionCoreSize,
      params.printDiffusionHaloIntensity,
      params.printDiffusionHaloSize,
      params.printDiffusionBloomIntensity,
      params.printDiffusionBloomSize,
      pixelSizeUm
    ));
  }
  return radius;
}

OfxRectD mapRoiThroughEnlarger(
  const spektrafilm::RenderParams &params,
  const OfxRectD &roi,
  const OfxRectD &sourceRod
) {
  if (!enlargerTransformActive(params)) {
    return roi;
  }
  const double width = std::max(rectWidth(sourceRod), 1.0);
  const double height = std::max(rectHeight(sourceRod), 1.0);
  const double scale = enlargerScale(params);
  const double offsetX = static_cast<double>(params.enlargerOffsetXPercent) * 0.01 / scale;
  const double offsetY = static_cast<double>(params.enlargerOffsetYPercent) * 0.01 / scale;
  const auto mapX = [&](double x) {
    const double normalized = (x - sourceRod.x1) / width;
    return sourceRod.x1 + (0.5 + (normalized - 0.5) / scale + offsetX) * width;
  };
  const auto mapY = [&](double y) {
    const double normalized = (y - sourceRod.y1) / height;
    return sourceRod.y1 + (0.5 + (normalized - 0.5) / scale + offsetY) * height;
  };
  OfxRectD mapped{};
  mapped.x1 = std::min(mapX(roi.x1), mapX(roi.x2));
  mapped.x2 = std::max(mapX(roi.x1), mapX(roi.x2));
  mapped.y1 = std::min(mapY(roi.y1), mapY(roi.y2));
  mapped.y2 = std::max(mapY(roi.y1), mapY(roi.y2));
  return mapped;
}

OfxStatus getRegionOfDefinition(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  InstanceData *data = getInstanceData(effect);
  OfxTime time = 0.0;
  gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  OfxRectD rod{};
  gEffectHost->clipGetRegionOfDefinition(data->sourceClip, time, &rod);
  gPropHost->propSetDoubleN(outArgs, kOfxImageEffectPropRegionOfDefinition, 4, &rod.x1);
  return kOfxStatOK;
}

OfxStatus getRegionOfInterest(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  InstanceData *data = getInstanceData(effect);
  OfxTime time = 0.0;
  OfxRectD roi{};
  gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  gPropHost->propGetDoubleN(inArgs, kOfxImageEffectPropRegionOfInterest, 4, &roi.x1);

  if (data) {
    const spektrafilm::RenderParams params = readParams(data, time);
    const int expansion = estimateSourceExpansionPixels(data, time, params, roi);
    OfxRectD sourceRod{};
    if (data->sourceClip && gEffectHost->clipGetRegionOfDefinition(data->sourceClip, time, &sourceRod) == kOfxStatOK) {
      roi = mapRoiThroughEnlarger(params, roi, sourceRod);
      const double sourceExpansion = std::ceil(static_cast<double>(expansion) / enlargerScale(params));
      roi.x1 -= sourceExpansion;
      roi.y1 -= sourceExpansion;
      roi.x2 += sourceExpansion;
      roi.y2 += sourceExpansion;
      roi.x1 = std::max(roi.x1, sourceRod.x1);
      roi.y1 = std::max(roi.y1, sourceRod.y1);
      roi.x2 = std::min(roi.x2, sourceRod.x2);
      roi.y2 = std::min(roi.y2, sourceRod.y2);
    } else {
      roi.x1 -= expansion;
      roi.y1 -= expansion;
      roi.x2 += expansion;
      roi.y2 += expansion;
    }
  }

  gPropHost->propSetDoubleN(outArgs, "OfxImageClipPropRoI_Source", 4, &roi.x1);
  return kOfxStatOK;
}

OfxStatus render(OfxImageEffectHandle effect, OfxPropertySetHandle inArgs, OfxPropertySetHandle) {
  InstanceData *data = getInstanceData(effect);
  if (!data) {
    return kOfxStatFailed;
  }

  OfxTime time = 0.0;
  OfxRectI renderWindow{};
  int cudaEnabled = 0;
  gPropHost->propGetDouble(inArgs, kOfxPropTime, 0, &time);
  gPropHost->propGetIntN(inArgs, kOfxImageEffectPropRenderWindow, 4, &renderWindow.x1);
#if defined(_WIN32) && defined(SPEKTRAFILM_ENABLE_CUDA)
  gPropHost->propGetInt(inArgs, kOfxImageEffectPropCudaEnabled, 0, &cudaEnabled);
#endif
  spektrafilm::RenderWindow window{renderWindow.x1, renderWindow.y1, renderWindow.x2, renderWindow.y2};
  spektrafilm::RenderParams params = readParams(data, time);

#if SPEKTRAFILM_OFX_METAL_GPU_BUFFERS
  if (metalGpuBuffersEnabled(inArgs)) {
    std::lock_guard<std::mutex> rendererLock(data->rendererMutex);
    if (!ensureRenderer(data)) {
      return kOfxStatFailed;
    }
    return probeMetalGpuBufferRender(effect, data, time, inArgs, window, params);
  }
#endif

  OfxPropertySetHandle sourceImage = nullptr;
  OfxPropertySetHandle outputImage = nullptr;
  spektrafilm::ImageView source{};
  spektrafilm::MutableImageView output{};
  OfxStatus status = kOfxStatOK;

  try {
    status = fetchImageView(data->sourceClip, time, &sourceImage, source);
    if (status != kOfxStatOK) {
      std::ostringstream log;
      log << "render source fetch failed status=" << statusName(status)
          << "(" << status << ") source=" << imageViewSummary(source);
      appendHostLog(log.str(), true);
      throw status;
    }
    status = fetchMutableImageView(data->outputClip, time, &outputImage, output);
    if (status != kOfxStatOK) {
      std::ostringstream log;
      log << "render output fetch failed status=" << statusName(status)
          << "(" << status << ") output=" << imageViewSummary(output);
      appendHostLog(log.str(), true);
      throw status;
    }

    // CUDA direct path: these OFX image pointers are already device memory, so don't stage through CPU.
#if defined(_WIN32) && defined(SPEKTRAFILM_ENABLE_CUDA)
    if (cudaEnabled) {
      source.memoryDomain = spektrafilm::ImageMemoryDomain::CudaDevice;
      output.memoryDomain = spektrafilm::ImageMemoryDomain::CudaDevice;
    }
#endif

    std::lock_guard<std::mutex> rendererLock(data->rendererMutex);
    spektrafilm::Renderer *renderer = ensureRenderer(data);
    if (!renderer) {
      throw kOfxStatFailed;
    }
    if (!renderer->render(source, output, window, params, time)) {
      std::ostringstream log;
      log << "render failed time=" << time
          << " cudaEnabled=" << cudaEnabled
          << " window=" << renderWindowSummary(window)
          << " source=" << imageViewSummary(source)
          << " output=" << imageViewSummary(output)
          << " error=\"" << renderer->lastError() << "\" "
          << rendererDiagnosticsSummary(renderer->lastDiagnostics());
      appendHostLog(log.str(), true);
      if (gMessageHost) {
        gMessageHost->message(effect, kOfxMessageError, "spektrafilmRenderer", "%s", renderer->lastError().c_str());
      }
      status = kOfxStatFailed;
    } else {
      const uint32_t logIndex = gHostRenderLogCount.fetch_add(1u);
      if (hostDebugLogVerbose() || logIndex < 200u) {
        std::ostringstream log;
        log << "render ok time=" << time
            << " cudaEnabled=" << cudaEnabled
            << " window=" << renderWindowSummary(window)
            << " process=" << static_cast<int>(params.process)
            << " outputMode=" << static_cast<int>(params.renderOutput)
            << " source=" << imageViewSummary(source)
            << " output=" << imageViewSummary(output)
            << " " << rendererDiagnosticsSummary(renderer->lastDiagnostics());
        appendHostLog(log.str(), true);
      }
    }
  } catch (OfxStatus caught) {
    status = caught;
  } catch (const std::bad_alloc &) {
    status = kOfxStatErrMemory;
  } catch (...) {
    status = kOfxStatErrUnknown;
  }

  releaseImage(sourceImage);
  releaseImage(outputImage);
  if (status != kOfxStatOK) {
    std::ostringstream log;
    log << "render returning status=" << statusName(status) << "(" << status << ")";
    appendHostLog(log.str(), true);
  }
  return gEffectHost->abort(effect) ? kOfxStatOK : status;
}

OfxStatus describeInContext(OfxImageEffectHandle effect, OfxPropertySetHandle) {
  OfxPropertySetHandle props = nullptr;
  gEffectHost->clipDefine(effect, kOfxImageEffectOutputClipName, &props);
  gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

  gEffectHost->clipDefine(effect, kOfxImageEffectSimpleSourceClipName, &props);
  gPropHost->propSetString(props, kOfxImageEffectPropSupportedComponents, 0, kOfxImageComponentRGBA);

  OfxParamSetHandle paramSet = nullptr;
  gEffectHost->getParamSet(effect, &paramSet);

  DefaultsSnapshot savedDefaults;
  bool defaultsFound = false;
  std::string defaultsError;
  if (loadDefaultsFromFile(savedDefaults, defaultsFound, defaultsError) && defaultsFound) {
    gDescribeDefaults = &savedDefaults;
  }

  defineGroup(paramSet, "colorGroup", "Color Management", true);
  defineGroup(paramSet, "filteringGroup", "Filtering", false);
  defineGroup(paramSet, "enlargerGroup", "Film Plane", false);
  defineGroup(paramSet, "filmGroup", "Film", true);
  defineGroup(paramSet, "printGroup", "Print", true);
  defineGroup(paramSet, "couplerGroup", "DIR Couplers", false);
  defineGroup(paramSet, "grainGroup", "Grain", true);
  defineGroup(paramSet, "grainSynthesisGroup", "Grain Synthesis", false);
  defineGroup(paramSet, "halationGroup", "Halation", false);
  defineGroup(paramSet, "diffusionGroup", "Diffusion", false);
  defineGroup(paramSet, "scannerGroup", "Scanner", false);
  defineGroup(paramSet, "infoGroup", "Info", false);
  defineGroup(paramSet, "manageGroup", "Manage", false);

  const char *processOptions[] = {"Print simulation", "Scan negative", "Process negative"};
  defineChoice(paramSet, "process", "Mode", processOptions, 3, 0, "colorGroup");
  defineBool(paramSet, "scanNegativeInvert", "Invert Negative Scan", false, "colorGroup");
  const char *rgbToRawOptions[] = {"Hanatos 2026", "Hanatos 2025", "Mallett 2019"};
  defineChoice(paramSet, "rgbToRawMethod", "RGB to Raw", rgbToRawOptions, 3, 0, "filmGroup");
  const char *colorSpaces[] = {
    "ARRI LogC4",
    "ARRI LogC3 EI800",
    "BMDFilm WideGamut Gen5",
    "DaVinci Intermediate WideGamut",
    "RED Log3G10 REDWideGamutRGB",
    "Sony S-Log3 S-Gamut3",
    "Sony S-Log3 S-Gamut3.Cine",
    "Canon Log2 CinemaGamut D55",
    "Canon Log3 CinemaGamut D55",
    "Panasonic V-Log V-Gamut",
    "ACES2065-1",
    "ACEScg",
    "ACEScct",
    "ACEScc",
    "Linear Rec.2020",
    "Linear Rec.709",
    "Linear P3-D65",
    "sRGB",
    "Display P3",
    "ProPhoto RGB",
    "Adobe RGB (1998)",
    "DCI-P3",
    "P3-D65 Gamma 2.2",
    "P3-D65 Gamma 2.6",
    "Rec.709 Gamma 2.2",
    "Rec.709 Gamma 2.4"
  };
  const char *rcmInputColorSpaces[] = {
    "DaVinci Intermediate WideGamut",
    "BMDFilm WideGamut Gen5",
    "ACES2065-1",
    "ACEScg",
    "ACEScct",
    "ACEScc",
    "Linear Rec.2020",
    "Linear Rec.709",
    "Linear P3-D65",
    "Rec.709 Gamma 2.4",
    "Rec.709 Gamma 2.2",
    "sRGB",
    "Display P3",
    "P3-D65 Gamma 2.2",
    "P3-D65 Gamma 2.6",
    "DCI-P3"
  };
  const char *sceneOutputColorSpaces[] = {
    "DaVinci Intermediate WideGamut"
  };
  const char *sdrOutputColorSpaces[] = {
    "sRGB",
    "Display P3",
    "ProPhoto RGB",
    "Adobe RGB (1998)",
    "DCI-P3",
    "P3-D65 Gamma 2.2",
    "P3-D65 Gamma 2.6",
    "Rec.709 Gamma 2.2",
    "Rec.709 Gamma 2.4"
  };
  defineChoice(paramSet, "inputColorSpace", "Input Color Space", colorSpaces, static_cast<int>(sizeof(colorSpaces) / sizeof(colorSpaces[0])), 0, "colorGroup");
  defineChoice(paramSet, "rcmInputColorSpace", "Input Color Space", rcmInputColorSpaces, static_cast<int>(sizeof(rcmInputColorSpaces) / sizeof(rcmInputColorSpaces[0])), 0, "colorGroup");
  const char *outputRoles[] = {"Display Out SDR", "Display Out HDR", "RCM/ACES (Beta)"};
  defineChoice(paramSet, "outputRole", "Output Role", outputRoles, outputRoleOptionCountForFlavor(), 0, "colorGroup");
  defineChoice(paramSet, "sdrOutputColorSpace", "Output Color Space", sdrOutputColorSpaces, static_cast<int>(sizeof(sdrOutputColorSpaces) / sizeof(sdrOutputColorSpaces[0])), 8, "colorGroup");
  defineChoice(paramSet, "sceneOutputColorSpace", "Output Color Space", sceneOutputColorSpaces, static_cast<int>(sizeof(sceneOutputColorSpaces) / sizeof(sceneOutputColorSpaces[0])), 0, "colorGroup");
  const char *hdrPresets[] = {"PQ 1000", "PQ 4000", "HLG 1000", "Custom"};
  defineChoice(paramSet, "hdrPreset", "HDR Preset", hdrPresets, 4, 0, "colorGroup");
  const char *hdrTransfers[] = {"Rec.2100 ST2084 (PQ)", "Rec.2100 HLG"};
  defineChoice(paramSet, "hdrTransfer", "HDR Transfer", hdrTransfers, 2, 0, "colorGroup");
  defineDouble(paramSet, "hdrReferenceWhiteNits", "Reference White Nits", 203.0, 48.0, 1000.0, "colorGroup");
  defineDouble(paramSet, "hdrPeakNits", "Peak Nits", 1000.0, 100.0, 10000.0, "colorGroup");
  defineDouble(paramSet, "hdrExposureEv", "HDR Exposure EV", 0.0, -8.0, 8.0, "colorGroup");
  const char *hdrToneMappings[] = {"Soft Rolloff", "Hard Clip"};
  defineChoice(paramSet, "hdrToneMapping", "HDR Tone Mapping", hdrToneMappings, 2, 1, "colorGroup");
  defineBool(paramSet, "colorAdaptation", "Color Adaptation", false, "colorGroup");
  defineBool(paramSet, "colorAdaptationInputCompression", "Input Compression", true, "colorGroup");
  defineBool(paramSet, "colorAdaptationCurveSmoothing", "Curve Smoothing", true, "colorGroup");
  defineBool(paramSet, "colorAdaptationOutputLightnessCompression", "Output Lightness Compression", true, "colorGroup");
  defineBool(paramSet, "colorAdaptationOutputChromaCompression", "Output Chroma Compression", true, "colorGroup");
  defineBool(paramSet, "cameraUvFilterEnabled", "Filter UV", false, "filteringGroup");
  defineDouble(paramSet, "cameraUvCutNm", "UV Cut nm", 410.0, 380.0, 450.0, "filteringGroup");
  defineBool(paramSet, "cameraIrFilterEnabled", "Filter IR", false, "filteringGroup");
  defineDouble(paramSet, "cameraIrCutNm", "IR Cut nm", 675.0, 600.0, 780.0, "filteringGroup");

  std::vector<const char *> films;
  films.reserve(spektrafilm::kSpektraFilmCount);
  for (uint32_t i = 0; i < spektrafilm::kSpektraFilmCount; ++i) {
    const spektrafilm::ProfileCurveSet *profile = spektrafilm::filmProfileCurves(static_cast<int32_t>(i));
    films.push_back(profile && profile->name ? profile->name : "Unknown Film");
  }
  defineChoice(paramSet, "film", "Stock", films.data(), static_cast<int>(films.size()), static_cast<int>(spektrafilm::kSpektraDefaultFilmIndex), "filmGroup");
  const char *filmFormats[] = {"8mm", "Super 8", "16mm", "Super 16", "35mm", "Super 35", "65mm", "70mm / IMAX"};
  defineChoice(paramSet, "filmFormat", "Film Format", filmFormats, static_cast<int>(sizeof(filmFormats) / sizeof(filmFormats[0])), 4, "filmGroup");
  const char *pushPullModes[] = {"Standard", "Experimental"};
  defineChoice(paramSet, "filmPushPullMode", "Push / Pull Mode", pushPullModes, 2, 0, "filmGroup");
  defineDouble(paramSet, "filmPushPullStops", "Film Push / Pull Stops", 0.0, -2.0, 2.0, "filmGroup");
  defineDouble(paramSet, "negativeBleachBypassAmount", "Negative Bleach Bypass", 0.0, 0.0, 1.0, "filmGroup");
  defineDouble(paramSet, "negativeLeucoCyanCoupling", "Leuco-Cyan Coupling", 1.0, 0.0, 2.0, "filmGroup");

  std::vector<const char *> papers;
  papers.reserve(spektrafilm::kSpektraPaperCount);
  for (uint32_t i = 0; i < spektrafilm::kSpektraPaperCount; ++i) {
    const spektrafilm::ProfileCurveSet *profile = spektrafilm::paperProfileCurves(static_cast<int32_t>(i));
    papers.push_back(profile && profile->name ? profile->name : "Unknown Paper");
  }
  defineChoice(paramSet, "paper", "Paper", papers.data(), static_cast<int>(papers.size()), static_cast<int>(spektrafilm::kSpektraDefaultPaperIndex), "printGroup");
  if constexpr (spektrafilm::kSpektraAcademyPrinterDensityEnabled) {
    const char *printTimingModes[] = {"Filtered Enlarger", "Printer Density"};
    defineChoice(paramSet, "printTiming", "Print Timing", printTimingModes, 2, 0, "printGroup");
  } else {
    const char *printTimingModes[] = {"Filtered Enlarger"};
    defineChoice(paramSet, "printTiming", "Print Timing", printTimingModes, 1, 0, "printGroup");
  }
  defineDouble(paramSet, "printPushPullStops", "Print Push / Pull Stops", 0.0, -2.0, 2.0, "printGroup");
  defineDouble(paramSet, "printBleachBypassAmount", "Print Bleach Bypass", 0.0, 0.0, 1.0, "printGroup");

  defineDouble(paramSet, "filmExposureEv", "Exposure EV", 0.0, -8.0, 8.0, "filmGroup");
  defineBool(paramSet, "autoExposure", "Auto Exposure", false, "filmGroup");
  const char *autoExposureMethods[] = {"Center weighted", "Median"};
  defineChoice(paramSet, "autoExposureMethod", "Auto Exposure Meter", autoExposureMethods, 2, 0, "filmGroup");
  defineDouble(paramSet, "filmGamma", "Gamma", 1.0, 0.1, 2.0, "filmGroup");
  defineDouble(paramSet, "printExposureEv", "Exposure EV", 0.0, -5.0, 5.0, "printGroup");
  defineDouble(paramSet, "printGamma", "Gamma", 1.0, 0.1, 2.0, "printGroup");
  defineDouble(paramSet, "printShadowShape", "Shadow Shape", 0.0, -1.0, 1.0, "printGroup");
  defineDouble(paramSet, "printHighlightShape", "Highlight Shape", 0.0, -1.0, 1.0, "printGroup");
  defineDouble(paramSet, "filterC", "C Filter", 0.0, 0.0, 120.0, "printGroup");
  defineDouble(paramSet, "filterMShift", "M Filter Shift", 0.0, -60.0, 60.0, "printGroup");
  defineDouble(paramSet, "filterYShift", "Y Filter Shift", 0.0, -60.0, 60.0, "printGroup");
  defineDouble(paramSet, "enlargerScale", "Scale", 1.0, 1.0, 32.0, "enlargerGroup");
  defineDouble(paramSet, "enlargerOffsetXPercent", "Offset X %", 0.0, -100.0, 100.0, "enlargerGroup");
  defineDouble(paramSet, "enlargerOffsetYPercent", "Offset Y %", 0.0, -100.0, 100.0, "enlargerGroup");
  defineDouble(paramSet, "preflashExposure", "Preflash Exposure", 0.0, 0.0, 1.0, "printGroup");
  defineDouble(paramSet, "preflashMFilterShift", "Preflash M Filter Shift", 0.0, -60.0, 60.0, "printGroup");
  defineDouble(paramSet, "preflashYFilterShift", "Preflash Y Filter Shift", 0.0, -60.0, 60.0, "printGroup");
  if constexpr (spektrafilm::kSpektraAcademyPrinterDensityEnabled) {
    defineBool(paramSet, "printerLightsGang", "Gang Printer Points", false, "printGroup");
    defineBool(paramSet, "printerLightsGroup", "Group Printer Points", false, "printGroup");
    defineDouble(paramSet, "printerLightR", "Printer Point R", 0.0, -24.0, 24.0, "printGroup");
    defineDouble(paramSet, "printerLightG", "Printer Point G", 0.0, -24.0, 24.0, "printGroup");
    defineDouble(paramSet, "printerLightB", "Printer Point B", 0.0, -24.0, 24.0, "printGroup");
    defineBool(paramSet, "printerLightCalibration", "Printer Point Calibration", true, "printGroup");
  }
  defineDouble(paramSet, "dirAmount", "Amount", 0.0, 0.0, 2.0, "couplerGroup");
  defineDouble(paramSet, "dirDiffusionUm", "Diffusion um", 20.0, 0.0, 100.0, "couplerGroup");
  defineDouble(paramSet, "dirDiffusionTailUm", "Tail um", 200.0, 0.0, 1000.0, "couplerGroup");
  defineDouble(paramSet, "dirDiffusionTailWeight", "Tail Weight", 0.06, 0.0, 1.0, "couplerGroup");
  defineDouble(paramSet, "dirInhibitionSameLayer", "Same-Layer Inhibition", 1.0, 0.0, 2.0, "couplerGroup");
  defineDouble(paramSet, "dirInhibitionInterlayer", "Interlayer Inhibition", 1.0, 0.0, 2.0, "couplerGroup");
  defineDouble3DRange(paramSet, "dirGammaSameLayerRgb", "Same-Layer Gamma RGB", 0.336, 0.319, 0.273, 0.0, 1.0, "couplerGroup");
  defineDouble2DRange(paramSet, "dirGammaRToGb", "R -> G/B Gamma", 0.353, 0.302, 0.0, 1.0, "couplerGroup");
  defineDouble2DRange(paramSet, "dirGammaGToRb", "G -> R/B Gamma", 0.154, 0.353, 0.0, 1.0, "couplerGroup");
  defineDouble2DRange(paramSet, "dirGammaBToRg", "B -> R/G Gamma", 0.168, 0.226, 0.0, 1.0, "couplerGroup");
  definePushButton(paramSet, "dirCalibrateToStock", "Calibrate to Stock", "couplerGroup");
  defineHiddenBool(paramSet, "dirUsesStockCalibration", true);
  defineBool(paramSet, "grainEnabled", "Enabled", false, "grainGroup");
  const char *grainModels[] = {"Preview", "Production", "Grain Synthesis"};
  defineChoice(paramSet, "grainModel", "Model", grainModels, grainModelOptionCountForFlavor(), 0, "grainGroup");
  defineDouble(paramSet, "grainAmount", "Amount", 1.0, 0.0, 2.0, "grainGroup");
  defineDouble(paramSet, "grainSaturation", "Saturation", 1.0, 0.0, 1.0, "grainGroup");
  defineBool(paramSet, "grainSublayersEnabled", "Sublayers", true, "grainGroup");
  defineInt(paramSet, "grainSubLayerCount", "Sub Layer Count", 1, 1, 8, "grainGroup");
  defineDouble(paramSet, "grainParticleAreaUm2", "Particle Area um2", 0.1, 0.01, 5.0, "grainGroup");
  defineDouble3D(paramSet, "grainParticleScale", "Particle Scale RGB", 1.2, 1.0, 2.5, "grainGroup");
  defineDouble3D(paramSet, "grainParticleScaleLayers", "Layer Scale", 6.0, 1.0, 0.4, "grainGroup");
  defineDouble3D(paramSet, "grainDensityMin", "Density Min", 0.04, 0.05, 0.06, "grainGroup");
  defineDouble3D(paramSet, "grainUniformity", "Uniformity RGB", 0.99, 0.97, 0.98, "grainGroup");
  defineDouble(paramSet, "grainFinalBlurUm", "Final Grain Blur", 7.17, 0.0, 25.0, "grainGroup");
  defineDouble(paramSet, "grainBlurDyeCloudsUm", "Dye Cloud Blur um", 1.0, 0.0, 10.0, "grainGroup");
  defineDouble2D(paramSet, "grainMicroStructure", "Micro Structure", 0.2, 30.0, "grainGroup");
  defineInt(paramSet, kGrainSeedParamName, "Seed", descriptorGrainSeedDefault(), kGrainSeedMin, kGrainSeedMax, "grainGroup");
  defineBool(paramSet, "grainAnimate", "Animate", true, "grainGroup");
  defineDouble(paramSet, "grainSynthesisSize", "Synthesis Size", 1.0, 0.25, 4.0, "grainGroup");
  defineDouble(paramSet, "grainSynthesisAmount", "Synthesis Amount", 1.0, 0.0, 3.0, "grainGroup");
  defineDouble(paramSet, "grainSynthesisSharpness", "Synthesis Sharpness", 1.0, 0.25, 4.0, "grainGroup");
  defineDouble(paramSet, "grainSynthesisQuality", "Synthesis Quality", 1.0, 0.25, 4.0, "grainGroup");
  defineInt(paramSet, "grainSynthesisSamples", "Samples", 128, 1, 2048, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisMeanRadiusUm", "Mean Radius um", 0.25, 0.05, 10.0, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisRadiusStdDevRatio", "Radius StdDev Ratio", 0.0, 0.0, 1.0, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisObservationSigmaUm", "Observation Aperture Sigma um", 1.0, 0.0, 20.0, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisCellSizeRatio", "Cell Size Ratio", 1.0, 0.25, 2.0, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisMaxRadiusQuantile", "Max Radius Quantile", 0.999, 0.95, 0.9999, "grainSynthesisGroup");
  defineDouble(paramSet, "grainSynthesisCoverageEpsilon", "Coverage Epsilon", 0.0001, 0.000001, 0.01, "grainSynthesisGroup");
  defineInt(paramSet, "grainSynthesisMaxGrainsPerCell", "Max Grains Per Cell", 32, 1, 128, "grainSynthesisGroup");
  defineDouble3D(paramSet, "grainSynthesisRadiusScale", "Radius Scale RGB", 1.2, 1.0, 2.5, "grainSynthesisGroup");
  defineDouble3D(paramSet, "grainSynthesisLayerScale", "Layer Scale", 6.0, 1.0, 0.4, "grainSynthesisGroup");
  defineBool(paramSet, "grainSynthesisLayered", "Layered", true, "grainSynthesisGroup");
  defineBool(paramSet, "halationEnabled", "Enabled", false, "halationGroup");
  defineDouble(paramSet, "scatterAmount", "Scatter Amount", 1.0, 0.0, 2.0, "halationGroup");
  defineDouble(paramSet, "scatterScale", "Scatter Scale", 1.0, 0.0, 4.0, "halationGroup");
  defineDouble(paramSet, "halationAmount", "Amount", 1.0, 0.0, 4.0, "halationGroup");
  defineDouble(paramSet, "halationScale", "Scale", 1.0, 0.0, 4.0, "halationGroup");
  defineDouble(paramSet, "halationBoostEv", "Boost EV", 0.0, 0.0, 20.0, "halationGroup");
  defineDouble(paramSet, "halationBoostRange", "Boost Range", 0.3, 0.0, 1.0, "halationGroup");
  defineDouble(paramSet, "halationProtectEv", "Protect EV", 4.0, 0.0, 10.0, "halationGroup");
  defineRGB(paramSet, "halationStrength", "Strength RGB", 0.05, 0.015, 0.0, "halationGroup");
  const char *diffusionFamilies[] = {"Glimmerglass", "Black Pro-Mist", "Pro-Mist", "CineBloom"};
  defineBool(paramSet, "cameraDiffusionEnabled", "Camera Enabled", false, "diffusionGroup");
  defineChoice(paramSet, "cameraDiffusionFamily", "Camera Family", diffusionFamilies, 4, 1, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionStrength", "Camera Strength", 0.5, 0.0, 2.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionSpatialScale", "Camera Spatial Scale", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionHaloWarmth", "Camera Halo Warmth", 0.0, -1.5, 1.5, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionCoreIntensity", "Camera Core Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionCoreSize", "Camera Core Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionHaloIntensity", "Camera Halo Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionHaloSize", "Camera Halo Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionBloomIntensity", "Camera Bloom Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "cameraDiffusionBloomSize", "Camera Bloom Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineBool(paramSet, "printDiffusionEnabled", "Print Enabled", false, "diffusionGroup");
  defineChoice(paramSet, "printDiffusionFamily", "Print Family", diffusionFamilies, 4, 1, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionStrength", "Print Strength", 0.5, 0.0, 2.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionSpatialScale", "Print Spatial Scale", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionHaloWarmth", "Print Halo Warmth", 0.0, -1.5, 1.5, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionCoreIntensity", "Print Core Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionCoreSize", "Print Core Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionHaloIntensity", "Print Halo Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionHaloSize", "Print Halo Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionBloomIntensity", "Print Bloom Intensity", 1.0, 0.0, 4.0, "diffusionGroup");
  defineDouble(paramSet, "printDiffusionBloomSize", "Print Bloom Size", 1.0, 0.1, 4.0, "diffusionGroup");
  defineBool(paramSet, "scannerEnabled", "Enabled", false, "scannerGroup");
  defineBool(paramSet, "scannerWhiteCorrection", "White Correction", false, "scannerGroup");
  defineBool(paramSet, "scannerBlackCorrection", "Black Correction", false, "scannerGroup");
  defineDouble(paramSet, "scannerWhiteLevel", "White Level", 0.98, 0.0, 1.0, "scannerGroup");
  defineDouble(paramSet, "scannerBlackLevel", "Black Level", 0.01, 0.0, 1.0, "scannerGroup");
  defineDouble(paramSet, "glarePercent", "Glare Percent", 0.03, 0.0, 0.2, "scannerGroup");
  defineDouble(paramSet, "glareRoughness", "Glare Roughness", 0.7, 0.0, 4.0, "scannerGroup");
  defineDouble(paramSet, "glareBlur", "Glare Blur", 0.5, 0.0, 32.0, "scannerGroup");
  defineDouble(paramSet, "scannerMtf50LpMm", "MTF50 lp/mm", 60.0, 0.0, 300.0, "scannerGroup");
  defineDouble(paramSet, "scannerUnsharpRadiusUm", "Unsharp Radius um", 5.0, 0.0, 100.0, "scannerGroup");
  defineDouble(paramSet, "scannerUnsharpAmount", "Unsharp Amount", 0.7, 0.0, 4.0, "scannerGroup");
  defineLabel(paramSet, "infoVersion", "Version:", SPEKTRAFILM_VERSION_STRING, "infoGroup");
  defineLabel(paramSet, "infoCreatedBy", "Created by:", "Aedan Diez", "infoGroup");
  defineLabel(paramSet, "infoBasedOn", "Based on work by:", "Andrea Volpato & Johannes Hanika", "infoGroup");
  const char *gpuRenderTilingOptions[] = {"Full-frame", "Experimental tiled"};
  defineChoice(paramSet, "gpuRenderTiling", "GPU Render Tiling", gpuRenderTilingOptions, 2, 0, "manageGroup");
  const char *lutSizes[] = {"33", "65"};
  const char *lutDestinations[] = {"User", "DaVinci Resolve", "Nuke", "Adobe Creative", "Final Cut Pro"};
  defineChoice(paramSet, "lutSize", "LUT Size", lutSizes, 2, 1, "manageGroup");
  defineChoice(paramSet, "lutDestination", "LUT Destination", lutDestinations, 5, 0, "manageGroup");
  defineSingleLineString(paramSet, "lutIdentifier", "LUT Identifier", "spektrafilm", "manageGroup");
  definePushButton(paramSet, "exportLut", "Export LUT", "manageGroup");
  defineSingleLineString(paramSet, "presetName", "Preset Name", "spektrafilm_preset", "manageGroup");
  const std::vector<PresetEntry> presetEntries = listPresetEntries();
  const std::vector<std::string> presetLabels = presetChoiceLabels(presetEntries);
  std::vector<const char *> presetLabelPointers;
  presetLabelPointers.reserve(presetLabels.size());
  for (const std::string &label : presetLabels) {
    presetLabelPointers.push_back(label.c_str());
  }
  defineChoice(paramSet, "presetSelection", "Preset", presetLabelPointers.data(), static_cast<int>(presetLabelPointers.size()), 0, "manageGroup");
  definePushButton(paramSet, "savePreset", "Save Preset", "manageGroup");
  definePushButton(paramSet, "loadPreset", "Load Preset", "manageGroup");
  definePushButton(paramSet, "copyParams", "Copy Params", "manageGroup");
  definePushButton(paramSet, "pasteParams", "Paste Params", "manageGroup");
  definePushButton(paramSet, "saveDefaults", "Set Defaults", "manageGroup");
  definePushButton(paramSet, "resetDefaults", "Reset Factory Defaults", "manageGroup");
  definePushButton(paramSet, "openUserManual", "Open User Manual", "manageGroup");

  gDescribeDefaults = nullptr;
  return kOfxStatOK;
}

OfxStatus describe(OfxImageEffectHandle effect) {
  OfxPropertySetHandle props = nullptr;
  gEffectHost->getPropertySet(effect, &props);
  gPropHost->propSetString(props, kOfxPropLabel, 0, kPluginLabel);
  gPropHost->propSetString(props, kOfxPropIcon, 0, svgIconFileForFlavor());
  gPropHost->propSetString(props, kOfxPropIcon, 1, pngIconFileForFlavor());
  gPropHost->propSetString(props, kOfxImageEffectPluginPropGrouping, 0, "spektrafilm OFX");
  gPropHost->propSetString(props, kOfxImageEffectPropSupportedContexts, 0, kOfxImageEffectContextFilter);
  gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 0, kOfxBitDepthHalf);
  gPropHost->propSetString(props, kOfxImageEffectPropSupportedPixelDepths, 1, kOfxBitDepthFloat);
  gPropHost->propSetString(props, kOfxImageEffectPropColourManagementStyle, 0, kOfxImageEffectColourManagementCore);
  gPropHost->propSetString(props, kOfxImageEffectPropColourManagementAvailableConfigs, 0, kOfxConfigIdentifier);
  gPropHost->propSetInt(props, kOfxImageEffectPropSupportsMultipleClipDepths, 0, 0);
  gPropHost->propSetInt(props, kOfxImageEffectPropSupportsTiles, 0, 0);
  gPropHost->propSetInt(props, kOfxImageEffectPropTemporalClipAccess, 0, 0);
#if defined(__APPLE__) && SPEKTRAFILM_OFX_METAL_GPU_BUFFERS
  gPropHost->propSetString(props, kOfxImageEffectPropMetalRenderSupported, 0, "true");
  gPropHost->propSetString(props, kOfxImageEffectPropCPURenderSupported, 0, "true");
#endif
#if defined(_WIN32) && defined(SPEKTRAFILM_ENABLE_CUDA)
  gPropHost->propSetString(props, kOfxImageEffectPropCudaRenderSupported, 0, "true");
  gPropHost->propSetString(props, kOfxImageEffectPropCPURenderSupported, 0, "true");
#endif
  return kOfxStatOK;
}

OfxStatus onLoad() {
  if (!gHost) {
    return kOfxStatErrMissingHostFeature;
  }
  gEffectHost = reinterpret_cast<OfxImageEffectSuiteV1 *>(const_cast<void *>(gHost->fetchSuite(gHost->host, kOfxImageEffectSuite, 1)));
  gPropHost = reinterpret_cast<OfxPropertySuiteV1 *>(const_cast<void *>(gHost->fetchSuite(gHost->host, kOfxPropertySuite, 1)));
  gParamHost = reinterpret_cast<OfxParameterSuiteV1 *>(const_cast<void *>(gHost->fetchSuite(gHost->host, kOfxParameterSuite, 1)));
  gMessageHost = reinterpret_cast<OfxMessageSuiteV1 *>(const_cast<void *>(gHost->fetchSuite(gHost->host, kOfxMessageSuite, 1)));
  if (!gEffectHost || !gPropHost || !gParamHost) {
    return kOfxStatErrMissingHostFeature;
  }
  return kOfxStatOK;
}

OfxStatus pluginMain(const char *action, const void *handle, OfxPropertySetHandle inArgs, OfxPropertySetHandle outArgs) {
  auto effect = reinterpret_cast<OfxImageEffectHandle>(const_cast<void *>(handle));
  if (std::strcmp(action, kOfxActionLoad) == 0) {
    return onLoad();
  }
  if (std::strcmp(action, kOfxActionDescribe) == 0) {
    return describe(effect);
  }
  if (std::strcmp(action, kOfxImageEffectActionDescribeInContext) == 0) {
    return describeInContext(effect, inArgs);
  }
  if (std::strcmp(action, kOfxActionCreateInstance) == 0) {
    return createInstance(effect);
  }
  if (std::strcmp(action, kOfxActionDestroyInstance) == 0) {
    return destroyInstance(effect);
  }
  if (std::strcmp(action, kOfxImageEffectActionEndSequenceRender) == 0) {
    if (resetRendererAfterEndSequence()) {
      return releaseInactiveInstanceResources(effect, true);
    }
    if (releaseTransientAfterEndSequence()) {
      return releaseInactiveInstanceResources(effect, false);
    }
    return kOfxStatOK;
  }
  if (std::strcmp(action, kOfxActionPurgeCaches) == 0) {
    return releaseInactiveInstanceResources(effect, resetRendererOnPurgeCaches());
  }
  if (std::strcmp(action, kOfxActionInstanceChanged) == 0) {
    return instanceChanged(effect, inArgs);
  }
  if (std::strcmp(action, kOfxImageEffectActionGetRegionOfDefinition) == 0) {
    return getRegionOfDefinition(effect, inArgs, outArgs);
  }
  if (std::strcmp(action, kOfxImageEffectActionGetRegionsOfInterest) == 0) {
    return getRegionOfInterest(effect, inArgs, outArgs);
  }
  if (std::strcmp(action, kOfxImageEffectActionRender) == 0) {
    return render(effect, inArgs, outArgs);
  }
  return kOfxStatReplyDefault;
}

void setHost(OfxHost *host) {
  gHost = host;
}

OfxPlugin gPlugin = {
  kOfxImageEffectPluginApi,
  1,
  kPluginIdentifier,
  kPluginVersionMajor,
  kPluginVersionMinor,
  setHost,
  pluginMain
};

} // namespace

extern "C" {

SPEKTRA_EXPORT OfxPlugin *OfxGetPlugin(int nth) {
  return nth == 0 ? &gPlugin : nullptr;
}

SPEKTRA_EXPORT int OfxGetNumberOfPlugins(void) {
  return 1;
}

SPEKTRA_EXPORT OfxStatus OfxSetHost(const OfxHost *host) {
  setHost(const_cast<OfxHost *>(host));
  return kOfxStatOK;
}

}
