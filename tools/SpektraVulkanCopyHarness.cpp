#include "SpektraVulkanRenderer.h"
#include "SpektraProfileCurves.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#  include <stdlib.h>
#endif

namespace {

enum class SourcePattern {
  Synthetic,
  Ramp,
};

struct Options {
  int width = 1920;
  int height = 1080;
  int iterations = 3;
  SourcePattern sourcePattern = SourcePattern::Synthetic;
  bool halfOnly = false;
  bool floatOnly = false;
  bool corePass = false;
  bool printScanPass = false;
  bool halationPass = false;
  bool cameraDiffusionPass = false;
  bool printDiffusionPass = false;
  bool dirPass = false;
  bool scannerPostPass = false;
  bool previewGrainPass = false;
  bool productionGrainPass = false;
  bool grainSynthesisPass = false;
  std::string tileMode = "tiled";
  int tileWidth = 512;
  int tileHeight = 256;
  bool compareTileParity = false;
  int sharedBackendInstances = 1;
  bool sharedBackendConcurrent = false;
  bool memoryBudgetTest = false;
  std::string resourceDir;
};

bool parseInt(const char *text, int &out) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!end || *end != '\0' || value <= 0 || value > 32768) {
    return false;
  }
  out = static_cast<int>(value);
  return true;
}

void printUsage(const char *name) {
  std::cerr
    << "Usage: " << name << " [--width N] [--height N] [--iterations N]\n"
    << "       [--half-only|--float-only] [--core-pass] [--print-scan-pass]\n"
    << "       [--halation-pass] [--camera-diffusion-pass]\n"
    << "       [--print-diffusion-pass|--diffusion-pass]\n"
    << "       [--dir-pass] [--scanner-post-pass]\n"
    << "       [--preview-grain-pass|--production-grain-pass|--grain-synthesis-pass]\n"
    << "       [--tile-mode tiled|legacy] [--tile-size WIDTHxHEIGHT]\n"
    << "       [--compare-tile-parity]\n"
    << "       [--shared-backend-instances N] [--shared-backend-concurrent]\n"
    << "       [--memory-budget-test]\n"
    << "       [--input-pattern synthetic|ramp] [--input-ramp]\n"
    << "       [--resource-dir PATH]\n";
}

bool parseArgs(int argc, char **argv, Options &options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto requireValue = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << flag << " requires a value.\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      std::exit(0);
    } else if (arg == "--width") {
      const char *value = requireValue("--width");
      if (!value || !parseInt(value, options.width)) {
        return false;
      }
    } else if (arg == "--height") {
      const char *value = requireValue("--height");
      if (!value || !parseInt(value, options.height)) {
        return false;
      }
    } else if (arg == "--iterations") {
      const char *value = requireValue("--iterations");
      if (!value || !parseInt(value, options.iterations)) {
        return false;
      }
    } else if (arg == "--input-ramp") {
      options.sourcePattern = SourcePattern::Ramp;
    } else if (arg == "--input-pattern") {
      const char *value = requireValue("--input-pattern");
      if (!value) {
        return false;
      }
      const std::string pattern(value);
      if (pattern == "synthetic") {
        options.sourcePattern = SourcePattern::Synthetic;
      } else if (pattern == "ramp") {
        options.sourcePattern = SourcePattern::Ramp;
      } else {
        std::cerr << "--input-pattern expects synthetic or ramp.\n";
        return false;
      }
    } else if (arg == "--half-only") {
      options.halfOnly = true;
    } else if (arg == "--float-only") {
      options.floatOnly = true;
    } else if (arg == "--core-pass") {
      options.corePass = true;
    } else if (arg == "--print-scan-pass") {
      options.corePass = true;
      options.printScanPass = true;
    } else if (arg == "--halation-pass") {
      options.corePass = true;
      options.halationPass = true;
    } else if (arg == "--camera-diffusion-pass") {
      options.corePass = true;
      options.cameraDiffusionPass = true;
    } else if (arg == "--print-diffusion-pass") {
      options.corePass = true;
      options.printScanPass = true;
      options.printDiffusionPass = true;
    } else if (arg == "--diffusion-pass") {
      options.corePass = true;
      options.printScanPass = true;
      options.cameraDiffusionPass = true;
      options.printDiffusionPass = true;
    } else if (arg == "--dir-pass") {
      options.corePass = true;
      options.dirPass = true;
    } else if (arg == "--scanner-post-pass") {
      options.corePass = true;
      options.printScanPass = true;
      options.scannerPostPass = true;
    } else if (arg == "--preview-grain-pass") {
      options.corePass = true;
      options.previewGrainPass = true;
    } else if (arg == "--production-grain-pass") {
      options.corePass = true;
      options.productionGrainPass = true;
    } else if (arg == "--grain-synthesis-pass") {
      options.corePass = true;
      options.grainSynthesisPass = true;
    } else if (arg == "--tile-mode") {
      const char *value = requireValue("--tile-mode");
      if (!value) {
        return false;
      }
      options.tileMode = value;
      if (options.tileMode != "tiled" && options.tileMode != "legacy") {
        std::cerr << "--tile-mode expects tiled or legacy.\n";
        return false;
      }
    } else if (arg == "--tile-size") {
      const char *value = requireValue("--tile-size");
      if (!value) {
        return false;
      }
      const char *x = std::strchr(value, 'x');
      if (!x) {
        x = std::strchr(value, 'X');
      }
      if (!x) {
        std::cerr << "--tile-size expects WIDTHxHEIGHT.\n";
        return false;
      }
      std::string widthText(value, static_cast<size_t>(x - value));
      std::string heightText(x + 1);
      if (!parseInt(widthText.c_str(), options.tileWidth) || !parseInt(heightText.c_str(), options.tileHeight)) {
        return false;
      }
    } else if (arg == "--compare-tile-parity") {
      options.compareTileParity = true;
      options.corePass = true;
    } else if (arg == "--shared-backend-instances") {
      const char *value = requireValue("--shared-backend-instances");
      if (!value || !parseInt(value, options.sharedBackendInstances)) {
        return false;
      }
    } else if (arg == "--shared-backend-concurrent") {
      options.sharedBackendConcurrent = true;
      options.sharedBackendInstances = std::max(options.sharedBackendInstances, 4);
    } else if (arg == "--memory-budget-test") {
      options.memoryBudgetTest = true;
      options.corePass = true;
      options.sharedBackendInstances = std::max(options.sharedBackendInstances, 4);
    } else if (arg == "--resource-dir") {
      const char *value = requireValue("--resource-dir");
      if (!value) {
        return false;
      }
      options.resourceDir = value;
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return false;
    }
  }

  if (options.halfOnly && options.floatOnly) {
    std::cerr << "--half-only and --float-only are mutually exclusive.\n";
    return false;
  }
  const int grainModeCount =
    (options.previewGrainPass ? 1 : 0) +
    (options.productionGrainPass ? 1 : 0) +
    (options.grainSynthesisPass ? 1 : 0);
  if (grainModeCount > 1) {
    std::cerr << "--preview-grain-pass, --production-grain-pass, and --grain-synthesis-pass are mutually exclusive.\n";
    return false;
  }
  return true;
}

void setEnv(const char *name, const char *value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

std::vector<float> readFloatResourceFile(
  const std::filesystem::path &path,
  uint32_t expectedElementCount,
  std::string &error
) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    error = "Unable to open " + path.string() + ".";
    return {};
  }
  const std::streamoff byteSize = static_cast<std::streamoff>(input.tellg());
  const uint64_t expectedBytes = static_cast<uint64_t>(expectedElementCount) * sizeof(float);
  if (byteSize <= 0 || static_cast<uint64_t>(byteSize) != expectedBytes) {
    error = "Unexpected byte size for " + path.string() + ".";
    return {};
  }
  input.seekg(0, std::ios::beg);
  std::vector<float> data(expectedElementCount);
  if (!input.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(byteSize))) {
    error = "Unable to read " + path.string() + ".";
    return {};
  }
  return data;
}

std::vector<float> loadHanatosSpectra(const std::string &resourceDir) {
  const spektrafilm::HanatosSpectraLutInfo &hanatos = spektrafilm::hanatosSpectraLutInfo();
  if (hanatos.elementCount == 0u) {
    std::cerr << "Generated Hanatos spectra LUT metadata is unavailable.\n";
    return {};
  }

  std::vector<std::filesystem::path> candidates;
  if (!resourceDir.empty()) {
    candidates.push_back(std::filesystem::path(resourceDir) / "SpektraHanatos2025Spectra.f32");
  }
  candidates.push_back(std::filesystem::current_path() / "SpektraHanatos2025Spectra.f32");

  std::string lastError;
  for (const std::filesystem::path &candidate : candidates) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(candidate, ec)) {
      continue;
    }
    std::vector<float> data = readFloatResourceFile(candidate, hanatos.elementCount, lastError);
    if (!data.empty()) {
      return data;
    }
  }

  std::cerr << (lastError.empty()
    ? "Unable to locate SpektraHanatos2025Spectra.f32 for core harness validation."
    : lastError) << "\n";
  return {};
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

uint16_t floatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x007fffffu;

  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    mantissa = (mantissa | 0x00800000u) >> static_cast<uint32_t>(1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x00001000u) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  uint32_t roundedMantissa = (mantissa + 0x00001000u) >> 13;
  if (roundedMantissa == 0x0400u) {
    roundedMantissa = 0;
    ++exponent;
    if (exponent >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | roundedMantissa);
}

float sourceValue(int x, int y, int channel, int width, int height, SourcePattern pattern) {
  const float fx = static_cast<float>(x + 1) / static_cast<float>(std::max(width, 1));
  const float fy = static_cast<float>(y + 1) / static_cast<float>(std::max(height, 1));
  if (pattern == SourcePattern::Ramp) {
    if (channel == 3) {
      return 1.0f;
    }
    return width > 1
      ? static_cast<float>(x) / static_cast<float>(width - 1)
      : 0.0f;
  }
  if (channel == 0) {
    return 0.22f + 0.58f * fx;
  }
  if (channel == 1) {
    return 0.24f + 0.52f * fy;
  }
  if (channel == 2) {
    return 0.18f + 0.54f * (fx * 0.65f + fy * 0.35f);
  }
  return 1.0f;
}

float interpDensityCurve(
  float logRaw,
  uint32_t channel,
  float gammaFactor,
  const spektrafilm::ProfileCurveSet &curves
) {
  const uint32_t count = curves.exposureCount;
  if (count == 0u || !curves.logExposure || !curves.densityCurves) {
    return 0.0f;
  }

  const float gamma = std::max(gammaFactor, 1.0e-6f);
  const float firstX = curves.logExposure[0] / gamma;
  const float lastX = curves.logExposure[count - 1u] / gamma;
  if (logRaw <= firstX) {
    return curves.densityCurves[channel];
  }
  if (logRaw >= lastX) {
    return curves.densityCurves[(count - 1u) * 3u + channel];
  }

  uint32_t lo = 0u;
  uint32_t hi = count - 1u;
  while (hi - lo > 1u) {
    const uint32_t mid = (lo + hi) >> 1u;
    const float x = curves.logExposure[mid] / gamma;
    if (x <= logRaw) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  const float x0 = curves.logExposure[lo] / gamma;
  const float x1 = curves.logExposure[hi] / gamma;
  const float y0 = curves.densityCurves[lo * 3u + channel];
  const float y1 = curves.densityCurves[hi * 3u + channel];
  const float t = std::clamp((logRaw - x0) / std::max(x1 - x0, 1.0e-9f), 0.0f, 1.0f);
  return y0 + (y1 - y0) * t;
}

float sampleDecodeLut(float value, uint32_t colorSpace) {
  const float *luts = spektrafilm::colorDecodeLuts();
  if (!luts || spektrafilm::kSpektraColorTransferLutSize <= 1u) {
    return value;
  }
  const float decodeMin = spektrafilm::colorDecodeLutMin();
  const float decodeMax = spektrafilm::colorDecodeLutMax();
  const float range = std::max(decodeMax - decodeMin, 1.0e-6f);
  const float step = range / static_cast<float>(spektrafilm::kSpektraColorTransferLutSize - 1u);
  const uint32_t offset = std::min(colorSpace, spektrafilm::kSpektraColorSpaceCount - 1u) *
    spektrafilm::kSpektraColorTransferLutSize;
  if (value <= decodeMin) {
    const float y0 = luts[offset];
    const float y1 = luts[offset + 1u];
    return y0 + (value - decodeMin) * ((y1 - y0) / std::max(step, 1.0e-12f));
  }
  if (value >= decodeMax) {
    const float y0 = luts[offset + spektrafilm::kSpektraColorTransferLutSize - 2u];
    const float y1 = luts[offset + spektrafilm::kSpektraColorTransferLutSize - 1u];
    return y1 + (value - decodeMax) * ((y1 - y0) / std::max(step, 1.0e-12f));
  }

  const float position = ((value - decodeMin) / range) *
    static_cast<float>(spektrafilm::kSpektraColorTransferLutSize - 1u);
  const uint32_t lo = static_cast<uint32_t>(std::floor(position));
  const uint32_t hi = std::min(lo + 1u, spektrafilm::kSpektraColorTransferLutSize - 1u);
  const float t = position - static_cast<float>(lo);
  return luts[offset + lo] + (luts[offset + hi] - luts[offset + lo]) * t;
}

std::array<float, 3> inputToLinearSrgb(
  const std::array<float, 4> &rgba,
  const spektrafilm::RenderParams &params,
  const spektrafilm::ProfileCurveSet &curves
) {
  const uint32_t colorSpace = static_cast<uint32_t>(std::clamp<int32_t>(
    static_cast<int32_t>(params.inputColorSpace),
    0,
    static_cast<int32_t>(spektrafilm::kSpektraColorSpaceCount - 1u)
  ));

  std::array<float, 3> decoded{rgba[0], rgba[1], rgba[2]};
  const uint32_t *transferKinds = spektrafilm::colorTransferKinds();
  if (transferKinds && transferKinds[colorSpace] != 0u) {
    decoded = {
      sampleDecodeLut(rgba[0], colorSpace),
      sampleDecodeLut(rgba[1], colorSpace),
      sampleDecodeLut(rgba[2], colorSpace)
    };
  }

  const float *matrix = curves.inputToSrgb + static_cast<size_t>(colorSpace) * 9u;
  return {
    matrix[0] * decoded[0] + matrix[1] * decoded[1] + matrix[2] * decoded[2],
    matrix[3] * decoded[0] + matrix[4] * decoded[1] + matrix[5] * decoded[2],
    matrix[6] * decoded[0] + matrix[7] * decoded[1] + matrix[8] * decoded[2]
  };
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
  const spektrafilm::ProfileCurveSet &curves,
  const std::vector<float> &linearSensitivity,
  const spektrafilm::RenderParams &params
) {
  std::vector<float> filtered = linearSensitivity;
  if (!curves.wavelengths ||
      filtered.size() < static_cast<size_t>(curves.wavelengthCount) * 3u ||
      (!params.cameraUvFilterEnabled && !params.cameraIrFilterEnabled)) {
    return filtered;
  }

  constexpr float kPythonUvTransitionNm = 8.0f;
  constexpr float kPythonIrTransitionNm = 15.0f;
  std::array<float, 3> numerator = {0.0f, 0.0f, 0.0f};
  std::array<float, 3> denominator = {0.0f, 0.0f, 0.0f};
  std::vector<float> transmissionByWavelength(curves.wavelengthCount, 1.0f);
  for (uint32_t wavelength = 0; wavelength < curves.wavelengthCount; ++wavelength) {
    const float wl = curves.wavelengths[wavelength];
    const float uvTransmission = params.cameraUvFilterEnabled
      ? smoothErfEdge(wl, params.cameraUvCutNm, kPythonUvTransitionNm)
      : 1.0f;
    const float irTransmission = params.cameraIrFilterEnabled
      ? smoothErfEdge(wl, params.cameraIrCutNm, -kPythonIrTransitionNm)
      : 1.0f;
    const float transmission = uvTransmission * irTransmission;
    transmissionByWavelength[wavelength] = transmission;
    const uint32_t offset = wavelength * 3u;
    const float illuminant = curves.referenceIlluminantSpectrum ? curves.referenceIlluminantSpectrum[wavelength] : 1.0f;
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
  for (uint32_t wavelength = 0; wavelength < curves.wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    for (uint32_t channel = 0; channel < 3u; ++channel) {
      filtered[offset + channel] *= transmissionByWavelength[wavelength] / normalization[channel];
    }
  }
  return filtered;
}

std::array<float, 9> makeMallettRawMatrix(
  const spektrafilm::ProfileCurveSet &curves,
  const spektrafilm::RenderParams &params
) {
  const std::vector<float> linearSensitivity = applyCameraBandPass(
    curves,
    makeLinearSensitivity(curves.logSensitivity, curves.wavelengthCount),
    params
  );
  std::array<float, 9> matrix{};
  if (!curves.mallettBasisIlluminant ||
      linearSensitivity.size() < static_cast<size_t>(curves.wavelengthCount) * 3u) {
    return matrix;
  }
  for (uint32_t wavelength = 0; wavelength < curves.wavelengthCount; ++wavelength) {
    const uint32_t offset = wavelength * 3u;
    for (uint32_t outChannel = 0; outChannel < 3u; ++outChannel) {
      for (uint32_t inChannel = 0; inChannel < 3u; ++inChannel) {
        matrix[outChannel * 3u + inChannel] +=
          linearSensitivity[offset + outChannel] * curves.mallettBasisIlluminant[offset + inChannel];
      }
    }
  }

  const float normalization = std::max(curves.mallettRawMidgrayGreen, 1.0e-10f);
  for (float &value : matrix) {
    value /= normalization;
  }
  return matrix;
}

std::array<float, 3> applyMallettRawMatrix(
  const std::array<float, 3> &linearSrgb,
  const std::array<float, 9> &matrix
) {
  const std::array<float, 3> srgb{
    std::max(linearSrgb[0], 0.0f),
    std::max(linearSrgb[1], 0.0f),
    std::max(linearSrgb[2], 0.0f)
  };
  return {
    matrix[0] * srgb[0] + matrix[1] * srgb[1] + matrix[2] * srgb[2],
    matrix[3] * srgb[0] + matrix[4] * srgb[1] + matrix[5] * srgb[2],
    matrix[6] * srgb[0] + matrix[7] * srgb[1] + matrix[8] * srgb[2]
  };
}

std::vector<float> makeHanatosRawResponse(
  const spektrafilm::ProfileCurveSet &curves,
  const std::vector<float> &linearSensitivity,
  const std::vector<float> &hanatosSpectra,
  spektrafilm::RgbToRawMethod method
) {
  const spektrafilm::HanatosSpectraLutInfo &hanatos = spektrafilm::hanatosSpectraLutInfo();
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
    method == spektrafilm::RgbToRawMethod::Hanatos2026 &&
    curves.hanatos2026WindowParams &&
    curves.referenceIlluminantSpectrum;
  if (useHanatos2026) {
    constexpr float kSqrt2 = 1.4142135623730951f;
    const float cUv = curves.hanatos2026WindowParams[0];
    const float sigmaUv = curves.hanatos2026WindowParams[1];
    const float cIr = curves.hanatos2026WindowParams[2];
    const float sigmaIr = curves.hanatos2026WindowParams[3];
    if (sigmaUv > 0.0f && sigmaIr > 0.0f) {
      std::array<float, 3> numerator = {0.0f, 0.0f, 0.0f};
      std::array<float, 3> denominator = {0.0f, 0.0f, 0.0f};
      for (uint32_t wavelength = 0; wavelength < hanatos.wavelengthCount; ++wavelength) {
        const float wl = curves.wavelengths ? curves.wavelengths[wavelength] : 0.0f;
        const float edgeUv = smoothErfEdge(wl, cUv, sigmaUv * kSqrt2);
        const float edgeIr = smoothErfEdge(wl, cIr, -sigmaIr * kSqrt2);
        const float window = edgeUv * edgeIr;
        hanatos2026Window[wavelength] = window;
        const float illuminant = curves.referenceIlluminantSpectrum[wavelength];
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
  if (!useHanatos2026 && !curves.bandpassHanatos2025) {
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
          raw[0] += spectrum * linearSensitivity[sensitivityOffset] * curves.bandpassHanatos2025[sensitivityOffset];
          raw[1] += spectrum * linearSensitivity[sensitivityOffset + 1u] * curves.bandpassHanatos2025[sensitivityOffset + 1u];
          raw[2] += spectrum * linearSensitivity[sensitivityOffset + 2u] * curves.bandpassHanatos2025[sensitivityOffset + 2u];
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

float mitchellWeight(float value) {
  constexpr float b = 1.0f / 3.0f;
  constexpr float c = 1.0f / 3.0f;
  const float x = std::abs(value);
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

uint32_t safeIndex(int32_t index, uint32_t size) {
  if (size <= 1u) {
    return 0u;
  }
  const int32_t period = static_cast<int32_t>(size) * 2 - 2;
  int32_t mirrored = index % period;
  if (mirrored < 0) {
    mirrored += period;
  }
  if (mirrored >= static_cast<int32_t>(size)) {
    mirrored = period - mirrored;
  }
  return static_cast<uint32_t>(mirrored);
}

std::array<float, 3> applyHanatosRawResponse(
  const std::array<float, 3> &xyz,
  const std::vector<float> &rawResponse
) {
  const spektrafilm::HanatosSpectraLutInfo &hanatos = spektrafilm::hanatosSpectraLutInfo();
  if (hanatos.width < 2u ||
      hanatos.height < 2u ||
      rawResponse.size() < static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) * 3u) {
    return {0.0f, 0.0f, 0.0f};
  }

  const float b = xyz[0] + xyz[1] + xyz[2];
  const float invB = 1.0f / std::max(b, 1.0e-10f);
  const float xChromaticity = std::clamp(xyz[0] * invB, 0.0f, 1.0f);
  const float yChromaticity = std::clamp(xyz[1] * invB, 0.0f, 1.0f);
  const float tx = std::clamp((1.0f - xChromaticity) * (1.0f - xChromaticity), 0.0f, 1.0f);
  const float ty = std::clamp(yChromaticity / std::max(1.0f - xChromaticity, 1.0e-10f), 0.0f, 1.0f);
  const float xCoord = tx * static_cast<float>(hanatos.width - 1u);
  const float yCoord = ty * static_cast<float>(hanatos.height - 1u);
  const int32_t xBase = xCoord >= static_cast<float>(hanatos.width - 1u)
    ? static_cast<int32_t>(hanatos.width - 2u)
    : static_cast<int32_t>(std::floor(xCoord));
  const int32_t yBase = yCoord >= static_cast<float>(hanatos.height - 1u)
    ? static_cast<int32_t>(hanatos.height - 2u)
    : static_cast<int32_t>(std::floor(yCoord));
  const float xFrac = xCoord >= static_cast<float>(hanatos.width - 1u) ? 1.0f : xCoord - static_cast<float>(xBase);
  const float yFrac = yCoord >= static_cast<float>(hanatos.height - 1u) ? 1.0f : yCoord - static_cast<float>(yBase);
  const std::array<float, 4> wx = {
    mitchellWeight(xFrac + 1.0f),
    mitchellWeight(xFrac),
    mitchellWeight(xFrac - 1.0f),
    mitchellWeight(xFrac - 2.0f)
  };
  const std::array<float, 4> wy = {
    mitchellWeight(yFrac + 1.0f),
    mitchellWeight(yFrac),
    mitchellWeight(yFrac - 1.0f),
    mitchellWeight(yFrac - 2.0f)
  };

  std::array<float, 3> raw = {0.0f, 0.0f, 0.0f};
  float weightSum = 0.0f;
  for (uint32_t i = 0; i < 4u; ++i) {
    const uint32_t xi = safeIndex(xBase - 1 + static_cast<int32_t>(i), hanatos.width);
    for (uint32_t j = 0; j < 4u; ++j) {
      const uint32_t yj = safeIndex(yBase - 1 + static_cast<int32_t>(j), hanatos.height);
      const float weight = wx[i] * wy[j];
      weightSum += weight;
      const size_t lutOffset = (static_cast<size_t>(xi) * static_cast<size_t>(hanatos.height) + yj) * 3u;
      raw[0] += weight * rawResponse[lutOffset];
      raw[1] += weight * rawResponse[lutOffset + 1u];
      raw[2] += weight * rawResponse[lutOffset + 2u];
    }
  }
  if (weightSum != 0.0f) {
    raw[0] /= weightSum;
    raw[1] /= weightSum;
    raw[2] /= weightSum;
  }
  raw[0] *= std::max(b, 0.0f);
  raw[1] *= std::max(b, 0.0f);
  raw[2] *= std::max(b, 0.0f);
  return raw;
}

std::array<float, 3> coreExpectedRgb(
  const std::array<float, 4> &rgba,
  const spektrafilm::RenderParams &params,
  const spektrafilm::ProfileCurveSet &curves,
  const std::array<float, 9> &mallettRawMatrix,
  const std::vector<float> &hanatosRawResponse
) {
  const uint32_t colorSpace = static_cast<uint32_t>(std::clamp<int32_t>(
    static_cast<int32_t>(params.inputColorSpace),
    0,
    static_cast<int32_t>(spektrafilm::kSpektraColorSpaceCount - 1u)
  ));
  std::array<float, 3> decoded{rgba[0], rgba[1], rgba[2]};
  const uint32_t *transferKinds = spektrafilm::colorTransferKinds();
  if (transferKinds && transferKinds[colorSpace] != 0u) {
    decoded = {
      sampleDecodeLut(rgba[0], colorSpace),
      sampleDecodeLut(rgba[1], colorSpace),
      sampleDecodeLut(rgba[2], colorSpace)
    };
  }

  const float *srgbMatrix = curves.inputToSrgb + static_cast<size_t>(colorSpace) * 9u;
  const std::array<float, 3> linearSrgb = {
    srgbMatrix[0] * decoded[0] + srgbMatrix[1] * decoded[1] + srgbMatrix[2] * decoded[2],
    srgbMatrix[3] * decoded[0] + srgbMatrix[4] * decoded[1] + srgbMatrix[5] * decoded[2],
    srgbMatrix[6] * decoded[0] + srgbMatrix[7] * decoded[1] + srgbMatrix[8] * decoded[2]
  };

  std::array<float, 3> raw{};
  if (params.rgbToRawMethod == spektrafilm::RgbToRawMethod::Mallett2019) {
    raw = applyMallettRawMatrix(linearSrgb, mallettRawMatrix);
  } else {
    const float *xyzMatrix = curves.inputToReferenceXyz + static_cast<size_t>(colorSpace) * 9u;
    const std::array<float, 3> referenceXyz = {
      xyzMatrix[0] * decoded[0] + xyzMatrix[1] * decoded[1] + xyzMatrix[2] * decoded[2],
      xyzMatrix[3] * decoded[0] + xyzMatrix[4] * decoded[1] + xyzMatrix[5] * decoded[2],
      xyzMatrix[6] * decoded[0] + xyzMatrix[7] * decoded[1] + xyzMatrix[8] * decoded[2]
    };
    raw = applyHanatosRawResponse(referenceXyz, hanatosRawResponse);
  }
  std::array<float, 3> out{};
  for (uint32_t channel = 0; channel < 3u; ++channel) {
    const float logRaw = std::log10(
      std::max(raw[channel] * std::exp2(params.filmExposureEv), 0.0f) + 1.0e-10f
    );
    out[channel] = std::max(interpDensityCurve(logRaw, channel, params.filmGamma, curves), 0.0f);
  }
  return out;
}

bool runCopyCase(spektrafilm::Renderer &renderer, int width, int height, int bytesPerComponent, int iterations) {
  constexpr int components = 4;
  const int pixelBytes = components * bytesPerComponent;
  const int sourceRowBytes = width * pixelBytes + 16;
  const int destinationRowBytes = width * pixelBytes + 32;
  std::vector<uint8_t> source(static_cast<size_t>(height) * static_cast<size_t>(sourceRowBytes), 0);
  std::vector<uint8_t> destination(static_cast<size_t>(height) * static_cast<size_t>(destinationRowBytes), 0xcd);

  for (int y = 0; y < height; ++y) {
    uint8_t *row = source.data() + static_cast<size_t>(y) * static_cast<size_t>(sourceRowBytes);
    for (int x = 0; x < width * pixelBytes; ++x) {
      row[x] = static_cast<uint8_t>((x * 13 + y * 17 + bytesPerComponent * 19) & 0xff);
    }
  }

  spektrafilm::ImageView sourceView{};
  sourceView.data = source.data();
  sourceView.width = width;
  sourceView.height = height;
  sourceView.rowBytes = sourceRowBytes;
  sourceView.components = components;
  sourceView.bytesPerComponent = bytesPerComponent;

  spektrafilm::MutableImageView destinationView{};
  destinationView.data = destination.data();
  destinationView.width = width;
  destinationView.height = height;
  destinationView.rowBytes = destinationRowBytes;
  destinationView.components = components;
  destinationView.bytesPerComponent = bytesPerComponent;

  spektrafilm::RenderWindow window{0, 0, width, height};
  spektrafilm::RenderParams params{};
  for (int i = 0; i < iterations; ++i) {
    if (!renderer.render(sourceView, destinationView, window, params, static_cast<double>(i))) {
      std::cerr << "Render failed: " << renderer.lastError() << "\n";
      return false;
    }
  }

  for (int y = 0; y < height; ++y) {
    const uint8_t *sourceRow = source.data() + static_cast<size_t>(y) * static_cast<size_t>(sourceRowBytes);
    const uint8_t *destinationRow = destination.data() + static_cast<size_t>(y) * static_cast<size_t>(destinationRowBytes);
    if (std::memcmp(sourceRow, destinationRow, static_cast<size_t>(width) * static_cast<size_t>(pixelBytes)) != 0) {
      std::cerr << "Copy mismatch at row " << y << " for " << bytesPerComponent * 8 << "-bit components.\n";
      return false;
    }
    const uint8_t *padding = destinationRow + static_cast<size_t>(width) * static_cast<size_t>(pixelBytes);
    if (!std::all_of(padding, padding + (destinationRowBytes - width * pixelBytes), [](uint8_t value) { return value == 0xcd; })) {
      std::cerr << "Destination row padding was modified at row " << y << ".\n";
      return false;
    }
  }

  const spektrafilm::RendererDiagnostics &diagnostics = renderer.lastDiagnostics();
  if (iterations > 1 && diagnostics.scratchAllocationCount != 0) {
    std::cerr
      << "Scratch buffers were reallocated on the final iteration for "
      << bytesPerComponent * 8 << "-bit components.\n";
    return false;
  }
  std::cout
    << bytesPerComponent * 8 << "-bit RGBA copy ok: "
    << width << "x" << height
    << ", passes=" << diagnostics.passCount
    << ", source_copy_ms=" << diagnostics.sourceCopyMs
    << ", cpu_setup_ms=" << diagnostics.cpuSetupMs
    << ", command_ms=" << diagnostics.commandBufferMs
    << ", output_copy_ms=" << diagnostics.outputCopyMs
    << ", total_ms=" << (diagnostics.sourceCopyMs + diagnostics.cpuSetupMs + diagnostics.commandBufferMs + diagnostics.outputCopyMs)
    << ", scratch_allocs=" << diagnostics.scratchAllocationCount
    << ", scratch_bytes=" << diagnostics.scratchAllocationBytes
    << "\n";
  return true;
}

const char *methodLabel(spektrafilm::RgbToRawMethod method) {
  switch (method) {
    case spektrafilm::RgbToRawMethod::Hanatos2025:
      return "Hanatos 2025";
    case spektrafilm::RgbToRawMethod::Mallett2019:
      return "Mallett 2019";
    case spektrafilm::RgbToRawMethod::Hanatos2026:
      return "Hanatos 2026";
    default:
      return "unknown";
  }
}

const char *sourcePatternLabel(SourcePattern pattern) {
  switch (pattern) {
    case SourcePattern::Ramp:
      return "ramp";
    case SourcePattern::Synthetic:
    default:
      return "synthetic";
  }
}

spektrafilm::RenderParams makeCoreHarnessParams(
  spektrafilm::RgbToRawMethod method,
  bool printScanPass,
  bool halationPass,
  bool cameraDiffusionPass,
  bool printDiffusionPass,
  bool dirPass,
  bool scannerPostPass,
  bool previewGrainPass,
  bool productionGrainPass,
  bool grainSynthesisPass
) {
  spektrafilm::RenderParams params{};
  params.inputColorSpace = spektrafilm::ColorSpace::DisplayP3;
  params.rgbToRawMethod = method;
  params.cameraUvFilterEnabled = true;
  params.cameraUvCutNm = 405.0f;
  params.cameraIrFilterEnabled = true;
  params.cameraIrCutNm = 680.0f;
  params.filmExposureEv = 0.5f;
  params.filmGamma = 1.15f;
  params.halationEnabled = halationPass;
  params.scatterAmount = halationPass ? 0.55f : params.scatterAmount;
  params.scatterScale = halationPass ? 0.65f : params.scatterScale;
  params.halationAmount = halationPass ? 0.45f : params.halationAmount;
  params.halationScale = halationPass ? 0.25f : params.halationScale;
  params.halationStrengthR = halationPass ? 0.035f : params.halationStrengthR;
  params.halationStrengthG = halationPass ? 0.012f : params.halationStrengthG;
  params.halationStrengthB = halationPass ? 0.002f : params.halationStrengthB;
  params.cameraDiffusionEnabled = cameraDiffusionPass;
  params.cameraDiffusionStrength = cameraDiffusionPass ? 0.5f : params.cameraDiffusionStrength;
  params.cameraDiffusionSpatialScale = cameraDiffusionPass ? 0.55f : params.cameraDiffusionSpatialScale;
  params.cameraDiffusionHaloWarmth = cameraDiffusionPass ? 0.2f : params.cameraDiffusionHaloWarmth;
  params.printDiffusionEnabled = printDiffusionPass;
  params.printDiffusionStrength = printDiffusionPass ? 0.45f : params.printDiffusionStrength;
  params.printDiffusionSpatialScale = printDiffusionPass ? 0.6f : params.printDiffusionSpatialScale;
  params.printDiffusionHaloWarmth = printDiffusionPass ? 0.1f : params.printDiffusionHaloWarmth;
  params.dirCouplersAmount = dirPass ? 0.85f : params.dirCouplersAmount;
  params.dirCouplersDiffusionUm = dirPass ? 18.0f : params.dirCouplersDiffusionUm;
  params.dirCouplersDiffusionTailUm = dirPass ? 160.0f : params.dirCouplersDiffusionTailUm;
  params.dirCouplersDiffusionTailWeight = dirPass ? 0.05f : params.dirCouplersDiffusionTailWeight;
  params.scannerEnabled = scannerPostPass;
  params.glarePercent = scannerPostPass ? 0.025f : params.glarePercent;
  params.glareRoughness = scannerPostPass ? 0.5f : params.glareRoughness;
  params.glareBlur = scannerPostPass ? 1.2f : params.glareBlur;
  params.scannerMtf50LpMm = scannerPostPass ? 80.0f : params.scannerMtf50LpMm;
  params.scannerUnsharpRadiusUm = scannerPostPass ? 8.0f : params.scannerUnsharpRadiusUm;
  params.scannerUnsharpAmount = scannerPostPass ? 0.35f : params.scannerUnsharpAmount;
  const bool grainPass = previewGrainPass || productionGrainPass || grainSynthesisPass;
  params.grainEnabled = grainPass;
  params.grainModel = grainSynthesisPass
    ? spektrafilm::GrainModel::GrainSynthesis
    : (productionGrainPass ? spektrafilm::GrainModel::Production : spektrafilm::GrainModel::Preview);
  params.grainAmount = grainPass ? 0.85f : params.grainAmount;
  params.grainSaturation = grainPass ? 0.75f : params.grainSaturation;
  params.grainSublayersEnabled = productionGrainPass || grainSynthesisPass;
  params.grainSubLayerCount = previewGrainPass ? 4 : params.grainSubLayerCount;
  params.grainParticleAreaUm2 = grainPass ? 0.12f : params.grainParticleAreaUm2;
  params.grainFinalBlurUm = grainPass ? 6.5f : params.grainFinalBlurUm;
  params.grainBlurDyeCloudsUm = (productionGrainPass || grainSynthesisPass) ? 1.0f : params.grainBlurDyeCloudsUm;
  params.grainSynthesisQuality = grainSynthesisPass ? 0.75f : params.grainSynthesisQuality;
  params.grainSynthesisSamples = grainSynthesisPass ? 64 : params.grainSynthesisSamples;
  params.grainSynthesisMeanRadiusUm = grainSynthesisPass ? 0.22f : params.grainSynthesisMeanRadiusUm;
  params.grainSynthesisObservationSigmaUm = grainSynthesisPass ? 0.65f : params.grainSynthesisObservationSigmaUm;
  params.grainSynthesisMaxGrainsPerCell = grainSynthesisPass ? 24 : params.grainSynthesisMaxGrainsPerCell;
  params.grainSeed = grainPass ? 37u : params.grainSeed;
  params.grainAnimate = false;
  (void)printScanPass;
  return params;
}

bool runCoreCase(
  spektrafilm::Renderer &renderer,
  int width,
  int height,
  int bytesPerComponent,
  int iterations,
  SourcePattern sourcePattern,
  spektrafilm::RgbToRawMethod method,
  const std::string &resourceDir,
  bool printScanPass,
  bool halationPass,
  bool cameraDiffusionPass,
  bool printDiffusionPass,
  bool dirPass,
  bool scannerPostPass,
  bool previewGrainPass,
  bool productionGrainPass,
  bool grainSynthesisPass
) {
  constexpr int components = 4;
  const int pixelBytes = components * bytesPerComponent;
  const int sourceRowBytes = width * pixelBytes + 16;
  const int destinationRowBytes = width * pixelBytes + 32;
  std::vector<uint8_t> source(static_cast<size_t>(height) * static_cast<size_t>(sourceRowBytes), 0);
  std::vector<uint8_t> destination(static_cast<size_t>(height) * static_cast<size_t>(destinationRowBytes), 0xcd);

  for (int y = 0; y < height; ++y) {
    uint8_t *row = source.data() + static_cast<size_t>(y) * static_cast<size_t>(sourceRowBytes);
    for (int x = 0; x < width; ++x) {
      if (bytesPerComponent == 4) {
        auto *pixel = reinterpret_cast<float *>(row + static_cast<size_t>(x) * 4u * sizeof(float));
        for (int c = 0; c < 4; ++c) {
          pixel[c] = sourceValue(x, y, c, width, height, sourcePattern);
        }
      } else {
        auto *pixel = reinterpret_cast<uint16_t *>(row + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
        for (int c = 0; c < 4; ++c) {
          pixel[c] = floatToHalf(sourceValue(x, y, c, width, height, sourcePattern));
        }
      }
    }
  }

  spektrafilm::ImageView sourceView{};
  sourceView.data = source.data();
  sourceView.width = width;
  sourceView.height = height;
  sourceView.rowBytes = sourceRowBytes;
  sourceView.components = components;
  sourceView.bytesPerComponent = bytesPerComponent;

  spektrafilm::MutableImageView destinationView{};
  destinationView.data = destination.data();
  destinationView.width = width;
  destinationView.height = height;
  destinationView.rowBytes = destinationRowBytes;
  destinationView.components = components;
  destinationView.bytesPerComponent = bytesPerComponent;

  spektrafilm::RenderWindow window{0, 0, width, height};
  spektrafilm::RenderParams params = makeCoreHarnessParams(
    method,
    printScanPass,
    halationPass,
    cameraDiffusionPass,
    printDiffusionPass,
    dirPass,
    scannerPostPass,
    previewGrainPass,
    productionGrainPass,
    grainSynthesisPass
  );
  const bool grainPass = previewGrainPass || productionGrainPass || grainSynthesisPass;
  const spektrafilm::ProfileCurveSet *curves = spektrafilm::filmProfileCurves(params.film);
  if (!curves) {
    curves = spektrafilm::filmProfileCurves(static_cast<int32_t>(spektrafilm::kSpektraDefaultFilmIndex));
  }
  if (!curves || curves->exposureCount == 0u || !curves->logExposure || !curves->densityCurves) {
    std::cerr << "Generated film density curves are unavailable.\n";
    return false;
  }
  if ((productionGrainPass || grainSynthesisPass) && (!curves->densityCurveLayers || !curves->densityCurveLayerMaxima)) {
    std::cerr << "Generated film grain layer data is unavailable.\n";
    return false;
  }
  if (!curves->inputToReferenceXyz || !curves->inputToSrgb || !spektrafilm::colorDecodeLuts() || !spektrafilm::colorTransferKinds()) {
    std::cerr << "Generated input color transform data is unavailable.\n";
    return false;
  }
  if (curves->wavelengthCount == 0u || !curves->logSensitivity ||
      !curves->mallettBasisIlluminant || curves->mallettRawMidgrayGreen <= 0.0f) {
    std::cerr << "Generated Mallett raw matrix data is unavailable.\n";
    return false;
  }
  if (params.rgbToRawMethod == spektrafilm::RgbToRawMethod::Hanatos2025 && !curves->bandpassHanatos2025) {
    std::cerr << "Generated Hanatos 2025 bandpass data is unavailable.\n";
    return false;
  }
  if (params.rgbToRawMethod == spektrafilm::RgbToRawMethod::Hanatos2026 &&
      (!curves->hanatos2026WindowParams || !curves->referenceIlluminantSpectrum || !curves->wavelengths)) {
    std::cerr << "Generated Hanatos 2026 adaptation data is unavailable.\n";
    return false;
  }
  const std::vector<float> filmSensitivityLinear = applyCameraBandPass(
    *curves,
    makeLinearSensitivity(curves->logSensitivity, curves->wavelengthCount),
    params
  );
  std::array<float, 9> mallettRawMatrix{};
  std::vector<float> hanatosRawResponse;
  if (params.rgbToRawMethod == spektrafilm::RgbToRawMethod::Mallett2019) {
    mallettRawMatrix = makeMallettRawMatrix(*curves, params);
  } else {
    const std::vector<float> hanatosSpectra = loadHanatosSpectra(resourceDir);
    if (hanatosSpectra.empty()) {
      return false;
    }
    hanatosRawResponse = makeHanatosRawResponse(*curves, filmSensitivityLinear, hanatosSpectra, params.rgbToRawMethod);
  }
  for (int i = 0; i < iterations; ++i) {
    if (!renderer.render(sourceView, destinationView, window, params, static_cast<double>(i))) {
      std::cerr << "Render failed: " << renderer.lastError() << "\n";
      return false;
    }
  }

  for (int y = 0; y < height; ++y) {
    const uint8_t *sourceRow = source.data() + static_cast<size_t>(y) * static_cast<size_t>(sourceRowBytes);
    const uint8_t *destinationRow = destination.data() + static_cast<size_t>(y) * static_cast<size_t>(destinationRowBytes);
    for (int x = 0; x < width; ++x) {
      std::array<float, 4> sourceComponents{};
      for (int c = 0; c < 4; ++c) {
        if (bytesPerComponent == 4) {
          const auto *pixel = reinterpret_cast<const float *>(sourceRow + static_cast<size_t>(x) * 4u * sizeof(float));
          sourceComponents[static_cast<size_t>(c)] = pixel[c];
        } else {
          const auto *pixel = reinterpret_cast<const uint16_t *>(sourceRow + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
          sourceComponents[static_cast<size_t>(c)] = halfToFloat(pixel[c]);
        }
      }
      const std::array<float, 3> expectedRgb = coreExpectedRgb(
        sourceComponents,
        params,
        *curves,
        mallettRawMatrix,
        hanatosRawResponse
      );
      for (int c = 0; c < 4; ++c) {
        const float expected = c == 3 ? sourceComponents[3] : expectedRgb[static_cast<size_t>(c)];
        if (bytesPerComponent == 4) {
          const auto *pixel = reinterpret_cast<const float *>(destinationRow + static_cast<size_t>(x) * 4u * sizeof(float));
          if (printScanPass || halationPass || cameraDiffusionPass || printDiffusionPass || dirPass || scannerPostPass || grainPass) {
            if (!std::isfinite(pixel[c])) {
              std::cerr << "Core smoke output was not finite at " << x << "," << y << " channel " << c << ".\n";
              return false;
            }
            if (c == 3 && std::abs(pixel[c] - expected) > 2.0e-4f) {
              std::cerr << "Core smoke alpha mismatch at " << x << "," << y << ".\n";
              return false;
            }
          } else {
            if (std::abs(pixel[c] - expected) > 2.0e-4f) {
              std::cerr << "Core mismatch at " << x << "," << y << " channel " << c << ".\n";
              return false;
            }
          }
        } else {
          const auto *pixel = reinterpret_cast<const uint16_t *>(destinationRow + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
          const uint16_t expectedHalf = floatToHalf(expected);
          const uint16_t actualHalf = pixel[c];
          const uint16_t halfDiff = actualHalf > expectedHalf
            ? static_cast<uint16_t>(actualHalf - expectedHalf)
            : static_cast<uint16_t>(expectedHalf - actualHalf);
          if (printScanPass || halationPass || cameraDiffusionPass || printDiffusionPass || dirPass || scannerPostPass || grainPass) {
            if (!std::isfinite(halfToFloat(actualHalf))) {
              std::cerr << "Core smoke half output was not finite at " << x << "," << y << " channel " << c << ".\n";
              return false;
            }
            if (c == 3 && halfDiff > 1u) {
              std::cerr << "Core smoke half alpha mismatch at " << x << "," << y << ".\n";
              return false;
            }
          } else if (halfDiff > 1u) {
            std::cerr
              << "Core half mismatch at " << x << "," << y << " channel " << c
              << ": actual=" << halfToFloat(actualHalf)
              << " expected=" << halfToFloat(expectedHalf)
              << " half_ulp_diff=" << halfDiff << ".\n";
            return false;
          }
        }
      }
    }
    const uint8_t *padding = destinationRow + static_cast<size_t>(width) * static_cast<size_t>(pixelBytes);
    if (!std::all_of(padding, padding + (destinationRowBytes - width * pixelBytes), [](uint8_t value) { return value == 0xcd; })) {
      std::cerr << "Destination row padding was modified at row " << y << ".\n";
      return false;
    }
  }

  const spektrafilm::RendererDiagnostics &diagnostics = renderer.lastDiagnostics();
  constexpr uint32_t kHarnessDiffusionComponentCount = 27u;
  const uint32_t cameraDiffusionExtraPassCount = cameraDiffusionPass
    ? 2u * kHarnessDiffusionComponentCount + 2u + (halationPass ? 0u : 1u)
    : 0u;
  const uint32_t printDiffusionExtraPassCount = printDiffusionPass
    ? 2u * kHarnessDiffusionComponentCount + 4u
    : 0u;
  const uint32_t dirExtraPassCount = dirPass ? 11u : 0u;
  const uint32_t scannerPostExtraPassCount = scannerPostPass ? 9u : 0u;
  const uint32_t grainExtraPassCount = previewGrainPass ? 1u : ((productionGrainPass || grainSynthesisPass) ? 10u : 0u);
  const uint32_t expectedPassCount =
    2u +
    cameraDiffusionExtraPassCount +
    (halationPass ? 18u : 0u) +
    dirExtraPassCount +
    grainExtraPassCount +
    (printScanPass ? (printDiffusionPass ? printDiffusionExtraPassCount : 1u) : 0u) +
    scannerPostExtraPassCount;
  const uint32_t expectedTotalPassCount =
    expectedPassCount * (diagnostics.tiledRendering ? std::max(diagnostics.tileCount, 1u) : 1u);
  if (diagnostics.passCount != expectedTotalPassCount) {
    std::cerr << "Expected " << expectedTotalPassCount << " core bootstrap passes, saw " << diagnostics.passCount << ".\n";
    return false;
  }
  if (iterations > 1 && diagnostics.scratchAllocationCount != 0) {
    std::cerr
      << "Scratch buffers were reallocated on the final core iteration for "
      << bytesPerComponent * 8 << "-bit components.\n";
    return false;
  }
  std::cout
    << bytesPerComponent * 8 << "-bit RGBA core bootstrap ok: "
    << width << "x" << height
    << ", method=" << methodLabel(params.rgbToRawMethod)
    << ", input=" << sourcePatternLabel(sourcePattern)
    << ", print_scan=" << (printScanPass ? "on" : "off")
    << ", halation=" << (halationPass ? "on" : "off")
    << ", camera_diffusion=" << (cameraDiffusionPass ? "on" : "off")
    << ", print_diffusion=" << (printDiffusionPass ? "on" : "off")
    << ", dir=" << (dirPass ? "on" : "off")
    << ", scanner_post=" << (scannerPostPass ? "on" : "off")
    << ", grain=" << (grainSynthesisPass ? "synthesis" : (productionGrainPass ? "production" : (previewGrainPass ? "preview" : "off")))
    << ", tile_mode=" << (diagnostics.tiledRendering ? "tiled" : "legacy")
    << ", tile_count=" << diagnostics.tileCount
    << ", tile_size=" << diagnostics.tileWidth << "x" << diagnostics.tileHeight
    << ", tile_overlap=" << diagnostics.tileOverlap
    << ", tile_records=" << diagnostics.tiles.size()
    << ", passes=" << diagnostics.passCount
    << ", source_copy_ms=" << diagnostics.sourceCopyMs
    << ", cpu_setup_ms=" << diagnostics.cpuSetupMs
    << ", command_ms=" << diagnostics.commandBufferMs
    << ", output_copy_ms=" << diagnostics.outputCopyMs
    << ", total_ms=" << (diagnostics.sourceCopyMs + diagnostics.cpuSetupMs + diagnostics.commandBufferMs + diagnostics.outputCopyMs)
    << ", private_scratch=" << (diagnostics.privateScratchEnabled ? "on" : "off")
    << ", scratch_allocs=" << diagnostics.scratchAllocationCount
    << ", scratch_bytes=" << diagnostics.scratchAllocationBytes
    << "\n";
  return true;
}

float readRenderedComponent(
  const std::vector<uint8_t> &image,
  int rowBytes,
  int bytesPerComponent,
  int x,
  int y,
  int channel
) {
  const uint8_t *ptr = image.data() +
    static_cast<size_t>(y) * static_cast<size_t>(rowBytes) +
    static_cast<size_t>(x * 4 + channel) * static_cast<size_t>(bytesPerComponent);
  if (bytesPerComponent == 4) {
    float value = 0.0f;
    std::memcpy(&value, ptr, sizeof(value));
    return value;
  }
  uint16_t half = 0u;
  std::memcpy(&half, ptr, sizeof(half));
  return halfToFloat(half);
}

uint16_t readRenderedHalfComponent(
  const std::vector<uint8_t> &image,
  int rowBytes,
  int x,
  int y,
  int channel
) {
  uint16_t half = 0u;
  const uint8_t *ptr = image.data() +
    static_cast<size_t>(y) * static_cast<size_t>(rowBytes) +
    static_cast<size_t>(x * 4 + channel) * sizeof(uint16_t);
  std::memcpy(&half, ptr, sizeof(half));
  return half;
}

bool runTileParityCase(
  spektrafilm::Renderer &renderer,
  int width,
  int height,
  int bytesPerComponent,
  SourcePattern sourcePattern,
  spektrafilm::RgbToRawMethod method,
  const Options &options
) {
  constexpr int components = 4;
  const int pixelBytes = components * bytesPerComponent;
  const int sourceRowBytes = width * pixelBytes + 16;
  const int destinationRowBytes = width * pixelBytes + 32;
  std::vector<uint8_t> source(static_cast<size_t>(height) * static_cast<size_t>(sourceRowBytes), 0);
  std::vector<uint8_t> legacyDestination(static_cast<size_t>(height) * static_cast<size_t>(destinationRowBytes), 0xcd);
  std::vector<uint8_t> tiledDestination(static_cast<size_t>(height) * static_cast<size_t>(destinationRowBytes), 0xcd);

  for (int y = 0; y < height; ++y) {
    uint8_t *row = source.data() + static_cast<size_t>(y) * static_cast<size_t>(sourceRowBytes);
    for (int x = 0; x < width; ++x) {
      if (bytesPerComponent == 4) {
        auto *pixel = reinterpret_cast<float *>(row + static_cast<size_t>(x) * 4u * sizeof(float));
        for (int c = 0; c < 4; ++c) {
          pixel[c] = sourceValue(x, y, c, width, height, sourcePattern);
        }
      } else {
        auto *pixel = reinterpret_cast<uint16_t *>(row + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
        for (int c = 0; c < 4; ++c) {
          pixel[c] = floatToHalf(sourceValue(x, y, c, width, height, sourcePattern));
        }
      }
    }
  }

  spektrafilm::ImageView sourceView{};
  sourceView.data = source.data();
  sourceView.width = width;
  sourceView.height = height;
  sourceView.rowBytes = sourceRowBytes;
  sourceView.components = components;
  sourceView.bytesPerComponent = bytesPerComponent;

  auto makeDestinationView = [&](std::vector<uint8_t> &buffer) {
    spektrafilm::MutableImageView view{};
    view.data = buffer.data();
    view.width = width;
    view.height = height;
    view.rowBytes = destinationRowBytes;
    view.components = components;
    view.bytesPerComponent = bytesPerComponent;
    return view;
  };

  spektrafilm::RenderWindow window{0, 0, width, height};
  spektrafilm::RenderParams params = makeCoreHarnessParams(
    method,
    options.printScanPass,
    options.halationPass,
    options.cameraDiffusionPass,
    options.printDiffusionPass,
    options.dirPass,
    options.scannerPostPass,
    options.previewGrainPass,
    options.productionGrainPass,
    options.grainSynthesisPass
  );

  setEnv("SPEKTRAFILM_VULKAN_TILE_MODE", "legacy");
  params.gpuRenderTiling = spektrafilm::GpuRenderTilingMode::LegacyFullFrame;
  spektrafilm::MutableImageView legacyView = makeDestinationView(legacyDestination);
  if (!renderer.render(sourceView, legacyView, window, params, 0.0)) {
    std::cerr << "Legacy parity render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const spektrafilm::RendererDiagnostics legacyDiagnostics = renderer.lastDiagnostics();

  setEnv("SPEKTRAFILM_VULKAN_TILE_MODE", "tiled");
  const std::string tileWidthText = std::to_string(options.tileWidth);
  const std::string tileHeightText = std::to_string(options.tileHeight);
  setEnv("SPEKTRAFILM_VULKAN_TILE_WIDTH", tileWidthText.c_str());
  setEnv("SPEKTRAFILM_VULKAN_TILE_HEIGHT", tileHeightText.c_str());
  params.gpuRenderTiling = spektrafilm::GpuRenderTilingMode::Tiled;
  spektrafilm::MutableImageView tiledView = makeDestinationView(tiledDestination);
  if (!renderer.render(sourceView, tiledView, window, params, 0.0)) {
    std::cerr << "Tiled parity render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const spektrafilm::RendererDiagnostics tiledDiagnostics = renderer.lastDiagnostics();

  double sumSquared = 0.0;
  double maxAbs = 0.0;
  double seamMaxAbs = 0.0;
  double nonSeamMaxAbs = 0.0;
  uint32_t maxHalfUlp = 0u;
  uint64_t sampleCount = 0u;
  const int tileWidth = std::max(options.tileWidth, 1);
  const int tileHeight = std::max(options.tileHeight, 1);
  for (int y = 0; y < height; ++y) {
    const bool ySeam = y > 0 && (y % tileHeight == 0 || ((y + 1) % tileHeight == 0));
    for (int x = 0; x < width; ++x) {
      const bool seam = ySeam || (x > 0 && (x % tileWidth == 0 || ((x + 1) % tileWidth == 0)));
      for (int c = 0; c < 4; ++c) {
        const float legacy = readRenderedComponent(legacyDestination, destinationRowBytes, bytesPerComponent, x, y, c);
        const float tiled = readRenderedComponent(tiledDestination, destinationRowBytes, bytesPerComponent, x, y, c);
        const double delta = static_cast<double>(std::abs(tiled - legacy));
        maxAbs = std::max(maxAbs, delta);
        sumSquared += delta * delta;
        ++sampleCount;
        if (seam) {
          seamMaxAbs = std::max(seamMaxAbs, delta);
        } else {
          nonSeamMaxAbs = std::max(nonSeamMaxAbs, delta);
        }
        if (bytesPerComponent == 2) {
          const uint16_t legacyHalf = readRenderedHalfComponent(legacyDestination, destinationRowBytes, x, y, c);
          const uint16_t tiledHalf = readRenderedHalfComponent(tiledDestination, destinationRowBytes, x, y, c);
          const uint32_t halfDiff = legacyHalf > tiledHalf
            ? static_cast<uint32_t>(legacyHalf - tiledHalf)
            : static_cast<uint32_t>(tiledHalf - legacyHalf);
          maxHalfUlp = std::max(maxHalfUlp, halfDiff);
        }
      }
    }
    const uint8_t *legacyPadding =
      legacyDestination.data() + static_cast<size_t>(y) * static_cast<size_t>(destinationRowBytes) +
      static_cast<size_t>(width) * static_cast<size_t>(pixelBytes);
    const uint8_t *tiledPadding =
      tiledDestination.data() + static_cast<size_t>(y) * static_cast<size_t>(destinationRowBytes) +
      static_cast<size_t>(width) * static_cast<size_t>(pixelBytes);
    if (!std::all_of(legacyPadding, legacyPadding + (destinationRowBytes - width * pixelBytes), [](uint8_t value) { return value == 0xcd; }) ||
        !std::all_of(tiledPadding, tiledPadding + (destinationRowBytes - width * pixelBytes), [](uint8_t value) { return value == 0xcd; })) {
      std::cerr << "Destination row padding was modified during tile parity at row " << y << ".\n";
      return false;
    }
  }

  const double rms = sampleCount > 0u ? std::sqrt(sumSquared / static_cast<double>(sampleCount)) : 0.0;
  const bool seamOk = seamMaxAbs <= std::max(nonSeamMaxAbs, bytesPerComponent == 4 ? 2.0e-5 : 1.0e-3);
  const bool valueOk = bytesPerComponent == 4
    ? (maxAbs <= 2.0e-5 && rms <= 2.0e-6)
    : (maxHalfUlp <= 1u);

  std::cout
    << bytesPerComponent * 8 << "-bit RGBA tile parity: "
    << width << "x" << height
    << ", method=" << methodLabel(method)
    << ", input=" << sourcePatternLabel(sourcePattern)
    << ", tile_size=" << options.tileWidth << "x" << options.tileHeight
    << ", tile_count=" << tiledDiagnostics.tileCount
    << ", tile_overlap=" << tiledDiagnostics.tileOverlap
    << ", tile_records=" << tiledDiagnostics.tiles.size()
    << ", max_abs=" << maxAbs
    << ", rms=" << rms
    << ", seam_max_abs=" << seamMaxAbs
    << ", non_seam_max_abs=" << nonSeamMaxAbs
    << ", half_ulp_max=" << maxHalfUlp
    << ", legacy_scratch_bytes=" << legacyDiagnostics.scratchAllocationBytes
    << ", tiled_scratch_bytes=" << tiledDiagnostics.scratchAllocationBytes
    << "\n";

  if (!valueOk || !seamOk) {
    std::cerr << "Tile parity thresholds failed.\n";
    return false;
  }
  return true;
}

bool runSharedBackendCase(const Options &options) {
  const int instanceCount = std::max(options.sharedBackendInstances, options.memoryBudgetTest ? 4 : 1);
  const int bytesPerComponent = options.floatOnly ? 4 : 2;
  const int iterations = options.memoryBudgetTest ? 1 : options.iterations;
  const bool runCore = options.corePass || options.memoryBudgetTest;

  if (options.memoryBudgetTest) {
    setEnv("SPEKTRAFILM_VULKAN_TRANSIENT_BUDGET_MB", "64");
  }

  std::vector<std::unique_ptr<spektrafilm::Renderer>> renderers;
  renderers.reserve(static_cast<size_t>(instanceCount));
  for (int i = 0; i < instanceCount; ++i) {
    std::unique_ptr<spektrafilm::Renderer> renderer = spektrafilm::createVulkanRenderer();
    if (!renderer || !renderer->isAvailable()) {
      std::cerr << "Shared backend renderer " << i << " is not available";
      if (renderer && !renderer->lastError().empty()) {
        std::cerr << ": " << renderer->lastError();
      }
      std::cerr << "\n";
      return false;
    }
    renderers.push_back(std::move(renderer));
  }

  auto runOne = [&](int index) -> bool {
    if (runCore) {
      return runCoreCase(
        *renderers[static_cast<size_t>(index)],
        options.width,
        options.height,
        bytesPerComponent,
        iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Mallett2019,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      );
    }
    return runCopyCase(
      *renderers[static_cast<size_t>(index)],
      options.width,
      options.height,
      bytesPerComponent,
      iterations
    );
  };

  bool ok = true;
  if (options.sharedBackendConcurrent) {
    std::vector<int> results(static_cast<size_t>(instanceCount), 0);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(instanceCount));
    for (int i = 0; i < instanceCount; ++i) {
      threads.emplace_back([&, i]() {
        results[static_cast<size_t>(i)] = runOne(i) ? 1 : 0;
      });
    }
    for (std::thread &thread : threads) {
      thread.join();
    }
    for (int result : results) {
      ok = (result != 0) && ok;
    }
  } else {
    for (int i = 0; i < instanceCount; ++i) {
      ok = runOne(i) && ok;
    }
  }
  if (!ok) {
    return false;
  }

  if (options.memoryBudgetTest) {
    if (!runOne(0)) {
      std::cerr << "Renderer failed after transient budget release.\n";
      return false;
    }
  }

  uint32_t generation = 0;
  uint32_t queueCount = 0;
  uint64_t transientTotal = 0;
  uint64_t transientBudget = 0;
  bool sawReleasedInactiveRenderer = false;
  for (size_t i = 0; i < renderers.size(); ++i) {
    const spektrafilm::RendererDiagnostics &diagnostics = renderers[i]->lastDiagnostics();
    if (!diagnostics.sharedBackend) {
      std::cerr << "Renderer " << i << " did not report a shared Vulkan backend.\n";
      return false;
    }
    if (diagnostics.sharedBackendGeneration == 0u || diagnostics.sharedQueueCount == 0u) {
      std::cerr << "Renderer " << i << " reported invalid shared backend diagnostics.\n";
      return false;
    }
    if (generation == 0u) {
      generation = diagnostics.sharedBackendGeneration;
      queueCount = diagnostics.sharedQueueCount;
      transientBudget = diagnostics.transientBudgetBytes;
    } else if (diagnostics.sharedBackendGeneration != generation || diagnostics.sharedQueueCount != queueCount) {
      std::cerr << "Renderer " << i << " did not share the same backend generation or queue count.\n";
      return false;
    }
    transientTotal += diagnostics.transientCachedBytes;
    sawReleasedInactiveRenderer = sawReleasedInactiveRenderer || diagnostics.transientCachedBytes == 0u;
  }

  if (options.memoryBudgetTest && !sawReleasedInactiveRenderer) {
    std::cerr << "Transient budget test did not release any inactive renderer resources.\n";
    return false;
  }

  std::cout
    << "Shared Vulkan backend ok: instances=" << instanceCount
    << ", generation=" << generation
    << ", queues=" << queueCount
    << ", transient_cached_bytes=" << transientTotal
    << ", transient_budget_bytes=" << transientBudget
    << ", concurrent=" << (options.sharedBackendConcurrent ? "on" : "off")
    << ", memory_budget_test=" << (options.memoryBudgetTest ? "on" : "off")
    << "\n";
  return true;
}

} // namespace

int main(int argc, char **argv) {
  Options options;
  if (!parseArgs(argc, argv, options)) {
    printUsage(argv[0]);
    return 2;
  }

  if (options.resourceDir.empty()) {
    std::error_code ec;
    const std::filesystem::path executablePath = std::filesystem::absolute(argv[0], ec);
    if (!ec && !executablePath.empty()) {
      options.resourceDir = executablePath.parent_path().string();
    }
  }

  setEnv("SPEKTRAFILM_VULKAN_COPY_PASS", options.corePass ? "0" : "1");
  setEnv("SPEKTRAFILM_VULKAN_PRINT_SCAN_PASS", options.printScanPass ? "1" : "0");
  setEnv("SPEKTRAFILM_VULKAN_HALATION_PASS", options.halationPass ? "1" : "0");
  setEnv("SPEKTRAFILM_VULKAN_DIFFUSION_PASS", (options.cameraDiffusionPass || options.printDiffusionPass) ? "1" : "0");
  setEnv("SPEKTRAFILM_VULKAN_DIR_PASS", options.dirPass ? "1" : "0");
  setEnv("SPEKTRAFILM_VULKAN_SCANNER_POST_PASS", options.scannerPostPass ? "1" : "0");
  setEnv(
    "SPEKTRAFILM_VULKAN_GRAIN_PASS",
    (options.previewGrainPass || options.productionGrainPass || options.grainSynthesisPass) ? "1" : "0"
  );
  setEnv("SPEKTRAFILM_VULKAN_TILE_MODE", options.tileMode.c_str());
  const std::string tileWidthText = std::to_string(options.tileWidth);
  const std::string tileHeightText = std::to_string(options.tileHeight);
  setEnv("SPEKTRAFILM_VULKAN_TILE_WIDTH", tileWidthText.c_str());
  setEnv("SPEKTRAFILM_VULKAN_TILE_HEIGHT", tileHeightText.c_str());
  if (!options.resourceDir.empty()) {
    setEnv("SPEKTRAFILM_RESOURCE_DIR", options.resourceDir.c_str());
  }

  if (options.sharedBackendInstances > 1 || options.sharedBackendConcurrent || options.memoryBudgetTest) {
    return runSharedBackendCase(options) ? 0 : 1;
  }

  std::unique_ptr<spektrafilm::Renderer> renderer = spektrafilm::createVulkanRenderer();
  if (!renderer || !renderer->isAvailable()) {
    std::cerr << "Vulkan renderer is not available";
    if (renderer && !renderer->lastError().empty()) {
      std::cerr << ": " << renderer->lastError();
    }
    std::cerr << "\n";
    return 1;
  }

  bool ok = true;
  if (options.compareTileParity) {
    if (!options.floatOnly) {
      ok = runTileParityCase(
        *renderer,
        options.width,
        options.height,
        2,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2026,
        options
      ) && ok;
    }
    if (!options.halfOnly) {
      ok = runTileParityCase(
        *renderer,
        options.width,
        options.height,
        4,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2026,
        options
      ) && ok;
    }
    return ok ? 0 : 1;
  }

  if (!options.floatOnly) {
    if (options.corePass) {
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        2,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Mallett2019,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        2,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2025,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        2,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2026,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
    } else {
      ok = runCopyCase(*renderer, options.width, options.height, 2, options.iterations) && ok;
    }
  }
  if (!options.halfOnly) {
    if (options.corePass) {
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        4,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Mallett2019,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        4,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2025,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
      ok = runCoreCase(
        *renderer,
        options.width,
        options.height,
        4,
        options.iterations,
        options.sourcePattern,
        spektrafilm::RgbToRawMethod::Hanatos2026,
        options.resourceDir,
        options.printScanPass,
        options.halationPass,
        options.cameraDiffusionPass,
        options.printDiffusionPass,
        options.dirPass,
        options.scannerPostPass,
        options.previewGrainPass,
        options.productionGrainPass,
        options.grainSynthesisPass
      ) && ok;
    } else {
      ok = runCopyCase(*renderer, options.width, options.height, 4, options.iterations) && ok;
    }
  }
  return ok ? 0 : 1;
}
