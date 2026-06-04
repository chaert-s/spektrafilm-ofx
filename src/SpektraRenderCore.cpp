#include "SpektraRenderCore.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <utility>

namespace spektrafilm {
namespace {

uint32_t halfToFloatBits(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exponent = static_cast<uint32_t>((h >> 10) & 0x1fu);
  uint32_t mantissa = static_cast<uint32_t>(h & 0x03ffu);
  if (exponent == 0u) {
    if (mantissa == 0u) {
      return sign;
    }
    while ((mantissa & 0x0400u) == 0u) {
      mantissa <<= 1u;
      --exponent;
    }
    ++exponent;
    mantissa &= 0x03ffu;
  } else if (exponent == 31u) {
    return sign | 0x7f800000u | (mantissa << 13u);
  }
  exponent = exponent + (127u - 15u);
  return sign | (exponent << 23u) | (mantissa << 13u);
}

} // namespace

float filmFormatMm(FilmFormat format) {
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

float enlargerScale(const RenderParams &params) {
  return std::clamp(params.enlargerScale, 1.0f, 32.0f);
}

bool enlargerTransformActive(const RenderParams &params) {
  return std::abs(enlargerScale(params) - 1.0f) > 1.0e-6f;
}

namespace {

float filmPushPullGamma(float stops) {
  const float clampedStops = std::clamp(stops, -2.0f, 2.0f);
  constexpr float kPush1Seconds = 46.0f;
  constexpr float kPush2Seconds = 71.0f;
  const float absStops = std::abs(clampedStops);
  const float segment = std::min(absStops, 1.0f);
  const float secondSegment = std::max(absStops - 1.0f, 0.0f);
  const float logFactor =
    std::log(kPush1Seconds / 30.0f) * segment +
    std::log(kPush2Seconds / kPush1Seconds) * secondSegment;
  const float factor = std::exp(logFactor);
  return clampedStops >= 0.0f ? factor : 1.0f / factor;
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

const ProfileCurveSet *selectedFilmCurves(const RenderParams &params) {
  const ProfileCurveSet *curves = filmProfileCurves(params.film);
  return curves ? curves : filmProfileCurves(static_cast<int32_t>(kSpektraDefaultFilmIndex));
}

const ProfileCurveSet *selectedPaperCurves(const RenderParams &params) {
  const ProfileCurveSet *curves = paperProfileCurves(params.paper);
  return curves ? curves : paperProfileCurves(static_cast<int32_t>(kSpektraDefaultPaperIndex));
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
    const float illuminant = filmCurves.referenceIlluminantSpectrum
      ? filmCurves.referenceIlluminantSpectrum[wavelength]
      : 1.0f;
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

std::array<float, 9> makeMallettRawMatrixUnnormalized(
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
  return matrix;
}

std::array<float, 9> makeMallettRawMatrix(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity
) {
  std::array<float, 9> matrix = makeMallettRawMatrixUnnormalized(filmCurves, linearSensitivity);
  const float normalization = std::max(filmCurves.mallettRawMidgrayGreen, 1.0e-10f);
  for (float &value : matrix) {
    value /= normalization;
  }
  return matrix;
}

std::vector<float> makeHanatosRawResponse(
  const ProfileCurveSet &filmCurves,
  const std::vector<float> &linearSensitivity,
  const std::vector<float> &hanatosSpectra,
  const HanatosSpectraLutInfo &hanatos,
  RgbToRawMethod method
) {
  const size_t responseCount =
    static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) * 3u;
  std::vector<float> response(responseCount, 0.0f);
  const size_t expectedSpectra =
    static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) *
    static_cast<size_t>(hanatos.wavelengthCount);
  if (hanatos.width == 0 ||
      hanatos.height == 0 ||
      hanatos.wavelengthCount == 0 ||
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
      float raw[3] = {0.0f, 0.0f, 0.0f};
      const size_t spectraOffset =
        (static_cast<size_t>(x) * hanatos.height + y) * hanatos.wavelengthCount;
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
      const size_t responseOffset = (static_cast<size_t>(x) * hanatos.height + y) * 3u;
      response[responseOffset] = raw[0];
      response[responseOffset + 1u] = raw[1];
      response[responseOffset + 2u] = raw[2];
    }
  }
  return response;
}

std::array<float, 3> densityCurveMaximums(const ProfileCurveSet &curves) {
  std::array<float, 3> maxima = {0.0f, 0.0f, 0.0f};
  for (uint32_t channel = 0; channel < 3u; ++channel) {
    for (uint32_t i = 0; i < curves.exposureCount; ++i) {
      maxima[channel] = std::max(maxima[channel], curves.densityCurves[i * 3u + channel]);
    }
  }
  return maxima;
}

float interpLinear(const std::vector<float> &x, const float *y, uint32_t channel, float target) {
  if (x.empty() || !y) {
    return 0.0f;
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

struct DiffusionGroup {
  float lambdaUm;
  float spread;
  uint32_t count;
  float alpha;
};

struct DiffusionFamilyShape {
  DiffusionGroup core;
  DiffusionGroup halo;
  DiffusionGroup bloom;
  float weightCore;
  float weightHalo;
  float weightBloom;
  float warmthBase;
  float totalGain;
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
  std::vector<KernelDiffusionComponent> &components,
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

void clusterDiffusionComponents(std::vector<KernelDiffusionComponent> &components, float sigmaRatio) {
  if (components.size() < 2u || sigmaRatio <= 0.0f) {
    return;
  }
  std::sort(components.begin(), components.end(), [](const KernelDiffusionComponent &a, const KernelDiffusionComponent &b) {
    return a.sigmaPx < b.sigmaPx;
  });
  std::vector<KernelDiffusionComponent> clustered;
  clustered.reserve(components.size());
  for (const KernelDiffusionComponent &component : components) {
    if (clustered.empty()) {
      clustered.push_back(component);
      continue;
    }
    KernelDiffusionComponent &last = clustered.back();
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

std::vector<KernelDiffusionComponent> makeDiffusionComponents(
  const DiffusionSettings &settings,
  float pixelSizeUm,
  KernelDiffusionInfo &info,
  float clusterSigmaRatio
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
    info.scatterFraction = 0.0f;
    info.componentCount = 0u;
    return {};
  }
  shape.core.lambdaUm *= std::max(settings.coreSize, 1.0e-6f);
  shape.halo.lambdaUm *= std::max(settings.haloSize, 1.0e-6f);
  shape.bloom.lambdaUm *= std::max(settings.bloomSize, 1.0e-6f);

  info.scatterFraction = diffusionScatterFraction(settings.strength, shape.totalGain);
  std::vector<KernelDiffusionComponent> components;
  if (info.scatterFraction <= 0.0f || settings.spatialScale <= 0.0f) {
    info.componentCount = 0u;
    return components;
  }

  appendDiffusionGroupComponents(components, shape.core, diffusionWeights(shape.core, false), {1.0f, 1.0f, 1.0f}, wc, settings.spatialScale, pixelSizeUm);

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

  appendDiffusionGroupComponents(components, shape.bloom, diffusionWeights(shape.bloom, true), {1.0f, 1.0f, 1.0f}, wb, settings.spatialScale, pixelSizeUm);

  clusterDiffusionComponents(components, clusterSigmaRatio);
  info.componentCount = static_cast<uint32_t>(components.size());
  return components;
}

void assignFloats(std::vector<float> &out, const float *data, size_t count) {
  out.assign(data, data + count);
}

void assignUInts(std::vector<uint32_t> &out, const uint32_t *data, size_t count) {
  out.assign(data, data + count);
}

} // namespace

KernelParams toKernelParams(const RenderParams &params, double time, int32_t width, int32_t height) {
  KernelParams out{};
  out.process = static_cast<int32_t>(params.process);
  out.rgbToRawMethod = static_cast<int32_t>(params.rgbToRawMethod);
  out.inputColorSpace = static_cast<int32_t>(params.inputColorSpace);
  out.outputColorSpace = static_cast<int32_t>(params.outputColorSpace);
  out.outputRole = static_cast<int32_t>(params.outputRole);
  out.hdrPreset = static_cast<int32_t>(params.hdrPreset);
  out.hdrTransfer = static_cast<int32_t>(params.hdrTransfer);
  out.hdrReferenceWhiteNits = params.hdrReferenceWhiteNits;
  out.hdrPeakNits = params.hdrPeakNits;
  out.hdrExposureEv = params.hdrExposureEv;
  out.hdrToneMapping = static_cast<int32_t>(params.hdrToneMapping);
  out.film = params.film;
  out.paper = params.paper;
  out.printTiming = static_cast<int32_t>(params.printTiming);
  out.filmExposureEv = params.filmExposureEv;
  out.autoExposureEnabled = params.autoExposure ? 1u : 0u;
  out.autoExposureMethod = static_cast<int32_t>(params.autoExposureMethod);
  out.autoExposureEv = 0.0f;
  out.printExposureEv = params.printExposureEv;
  out.filmPushPullMode = static_cast<int32_t>(params.filmPushPullMode);
  out.filmPushPullStops = params.filmPushPullStops;
  out.printPushPullMode = 0;
  out.printPushPullStops = params.printPushPullStops;
  out.negativeBleachBypassAmount = params.negativeBleachBypassAmount;
  out.negativeLeucoCyanCoupling = params.negativeLeucoCyanCoupling;
  out.printBleachBypassAmount = params.printBleachBypassAmount;
  out.filmGamma = params.filmPushPullMode == PushPullMode::Experimental
    ? params.filmGamma
    : params.filmGamma * filmPushPullGamma(params.filmPushPullStops);
  out.printGamma = params.printGamma * printPushPullGamma(params.printPushPullStops);
  out.printShadowShape = params.printShadowShape;
  out.printHighlightShape = params.printHighlightShape;
  out.filterC = params.filterC;
  out.filterMShift = params.filterMShift;
  out.filterYShift = params.filterYShift;
  out.enlargerScale = enlargerScale(params);
  out.enlargerOffsetXPercent = params.enlargerOffsetXPercent;
  out.enlargerOffsetYPercent = params.enlargerOffsetYPercent;
  out.preflashExposure = params.preflashExposure;
  out.preflashMFilterShift = params.preflashMFilterShift;
  out.preflashYFilterShift = params.preflashYFilterShift;
  out.printerLightsR = params.printerLightsR;
  out.printerLightsG = params.printerLightsG;
  out.printerLightsB = params.printerLightsB;
  out.printerLightsGang = params.printerLightsGang ? 1u : 0u;
  out.printerLightCalibration = params.printerLightCalibration ? 1u : 0u;
  out.dirCouplersAmount = params.dirCouplersAmount;
  out.dirCouplersDiffusionUm = params.dirCouplersDiffusionUm;
  out.dirCouplersDiffusionTailUm = params.dirCouplersDiffusionTailUm;
  out.dirCouplersDiffusionTailWeight = params.dirCouplersDiffusionTailWeight;
  out.grainEnabled = params.grainEnabled ? 1u : 0u;
  out.grainModel = static_cast<int32_t>(params.grainModel);
  out.filmFormat = static_cast<int32_t>(params.filmFormat);
  out.grainAmount = params.grainAmount;
  out.grainSaturation = params.grainSaturation;
  out.grainSublayersEnabled = params.grainSublayersEnabled ? 1u : 0u;
  out.grainSubLayerCount = std::max(1, params.grainSubLayerCount);
  out.grainParticleAreaUm2 = params.grainParticleAreaUm2;
  out.grainParticleScaleR = params.grainParticleScaleR;
  out.grainParticleScaleG = params.grainParticleScaleG;
  out.grainParticleScaleB = params.grainParticleScaleB;
  out.grainParticleScaleLayer0 = params.grainParticleScaleLayer0;
  out.grainParticleScaleLayer1 = params.grainParticleScaleLayer1;
  out.grainParticleScaleLayer2 = params.grainParticleScaleLayer2;
  out.grainDensityMinR = params.grainDensityMinR;
  out.grainDensityMinG = params.grainDensityMinG;
  out.grainDensityMinB = params.grainDensityMinB;
  out.grainUniformityR = params.grainUniformityR;
  out.grainUniformityG = params.grainUniformityG;
  out.grainUniformityB = params.grainUniformityB;
  out.grainFinalBlurUm = params.grainFinalBlurUm;
  out.grainBlurDyeCloudsUm = params.grainBlurDyeCloudsUm;
  out.grainMicroStructureScale = params.grainMicroStructureScale;
  out.grainMicroStructureSigmaNm = params.grainMicroStructureSigmaNm;
  out.grainSeed = params.grainSeed;
  out.grainAnimate = params.grainAnimate ? 1u : 0u;
  const int32_t filmReferencePixels = std::max(width, height);
  out.filmPixelSizeUm = filmFormatMm(params.filmFormat) * 1000.0f /
    static_cast<float>(std::max(filmReferencePixels, 1)) / out.enlargerScale;
  const float grainSynthesisQuality = std::clamp(params.grainSynthesisQuality, 0.25f, 4.0f);
  const float grainSynthesisSize = std::clamp(params.grainSynthesisSize, 0.25f, 4.0f);
  const float grainSynthesisSharpness = std::max(params.grainSynthesisSharpness, 0.25f);
  out.grainSynthesisSamples = std::clamp(
    static_cast<int32_t>(std::lround(static_cast<float>(params.grainSynthesisSamples) * grainSynthesisQuality)),
    1,
    1024
  );
  out.grainSynthesisAmount = std::clamp(params.grainSynthesisAmount, 0.0f, 3.0f);
  out.grainSynthesisMeanRadiusUm = params.grainSynthesisMeanRadiusUm * grainSynthesisSize;
  out.grainSynthesisRadiusStdDevRatio = params.grainSynthesisRadiusStdDevRatio;
  out.grainSynthesisObservationSigmaUm = params.grainSynthesisObservationSigmaUm / grainSynthesisSharpness;
  out.grainSynthesisCellSizeRatio = params.grainSynthesisCellSizeRatio;
  out.grainSynthesisMaxRadiusQuantile = params.grainSynthesisMaxRadiusQuantile;
  out.grainSynthesisCoverageEpsilon = params.grainSynthesisCoverageEpsilon;
  out.grainSynthesisMaxGrainsPerCell = std::clamp(params.grainSynthesisMaxGrainsPerCell, 1, 128);
  out.grainSynthesisRadiusScaleR = params.grainSynthesisRadiusScaleR;
  out.grainSynthesisRadiusScaleG = params.grainSynthesisRadiusScaleG;
  out.grainSynthesisRadiusScaleB = params.grainSynthesisRadiusScaleB;
  out.grainSynthesisLayerScale0 = params.grainSynthesisLayerScale0;
  out.grainSynthesisLayerScale1 = params.grainSynthesisLayerScale1;
  out.grainSynthesisLayerScale2 = params.grainSynthesisLayerScale2;
  out.grainSynthesisLayered = params.grainSynthesisLayered ? 1u : 0u;
  out.halationEnabled = params.halationEnabled ? 1u : 0u;
  out.scatterAmount = params.scatterAmount;
  out.scatterScale = params.scatterScale;
  out.halationAmount = params.halationAmount;
  out.halationScale = params.halationScale;
  out.halationStrengthR = params.halationStrengthR;
  out.halationStrengthG = params.halationStrengthG;
  out.halationStrengthB = params.halationStrengthB;
  out.halationFirstSigmaUmR = params.halationFirstSigmaUmR;
  out.halationFirstSigmaUmG = params.halationFirstSigmaUmG;
  out.halationFirstSigmaUmB = params.halationFirstSigmaUmB;
  out.halationBoostEv = params.halationBoostEv;
  out.halationBoostRange = params.halationBoostRange;
  out.halationProtectEv = params.halationProtectEv;
  out.cameraDiffusionEnabled = params.cameraDiffusionEnabled ? 1u : 0u;
  out.cameraDiffusionFamily = static_cast<int32_t>(params.cameraDiffusionFamily);
  out.cameraDiffusionStrength = params.cameraDiffusionStrength;
  out.cameraDiffusionSpatialScale = params.cameraDiffusionSpatialScale;
  out.cameraDiffusionHaloWarmth = params.cameraDiffusionHaloWarmth;
  out.cameraDiffusionCoreIntensity = params.cameraDiffusionCoreIntensity;
  out.cameraDiffusionCoreSize = params.cameraDiffusionCoreSize;
  out.cameraDiffusionHaloIntensity = params.cameraDiffusionHaloIntensity;
  out.cameraDiffusionHaloSize = params.cameraDiffusionHaloSize;
  out.cameraDiffusionBloomIntensity = params.cameraDiffusionBloomIntensity;
  out.cameraDiffusionBloomSize = params.cameraDiffusionBloomSize;
  out.printDiffusionEnabled = params.printDiffusionEnabled ? 1u : 0u;
  out.printDiffusionFamily = static_cast<int32_t>(params.printDiffusionFamily);
  out.printDiffusionStrength = params.printDiffusionStrength;
  out.printDiffusionSpatialScale = params.printDiffusionSpatialScale;
  out.printDiffusionHaloWarmth = params.printDiffusionHaloWarmth;
  out.printDiffusionCoreIntensity = params.printDiffusionCoreIntensity;
  out.printDiffusionCoreSize = params.printDiffusionCoreSize;
  out.printDiffusionHaloIntensity = params.printDiffusionHaloIntensity;
  out.printDiffusionHaloSize = params.printDiffusionHaloSize;
  out.printDiffusionBloomIntensity = params.printDiffusionBloomIntensity;
  out.printDiffusionBloomSize = params.printDiffusionBloomSize;
  out.scannerEnabled = params.scannerEnabled ? 1u : 0u;
  out.scannerWhiteCorrection = params.scannerWhiteCorrection ? 1u : 0u;
  out.scannerBlackCorrection = params.scannerBlackCorrection ? 1u : 0u;
  out.scannerWhiteLevel = params.scannerWhiteLevel;
  out.scannerBlackLevel = params.scannerBlackLevel;
  out.glarePercent = params.glarePercent;
  out.glareRoughness = params.glareRoughness;
  out.glareBlur = params.glareBlur;
  out.scannerBlurSigmaPx = scannerSigmaUmFromMtf50(params.scannerMtf50LpMm) / std::max(out.filmPixelSizeUm, 1.0e-6f);
  out.scannerUnsharpSigmaPx = std::max(params.scannerUnsharpRadiusUm, 0.0f) / std::max(out.filmPixelSizeUm, 1.0e-6f);
  out.scannerUnsharpAmount = params.scannerUnsharpAmount;
  out.time = static_cast<float>(time);
  return out;
}

std::vector<KernelDiffusionComponent> makeCameraDiffusionComponents(
  const RenderParams &params,
  float pixelSizeUm,
  KernelDiffusionInfo &info,
  float clusterSigmaRatio
) {
  info = KernelDiffusionInfo{};
  if (!params.cameraDiffusionEnabled) {
    return {};
  }
  const DiffusionSettings settings{
    params.cameraDiffusionFamily,
    params.cameraDiffusionStrength,
    params.cameraDiffusionSpatialScale,
    params.cameraDiffusionHaloWarmth,
    params.cameraDiffusionCoreIntensity,
    params.cameraDiffusionCoreSize,
    params.cameraDiffusionHaloIntensity,
    params.cameraDiffusionHaloSize,
    params.cameraDiffusionBloomIntensity,
    params.cameraDiffusionBloomSize
  };
  return makeDiffusionComponents(settings, pixelSizeUm, info, clusterSigmaRatio);
}

std::vector<KernelDiffusionComponent> makePrintDiffusionComponents(
  const RenderParams &params,
  float pixelSizeUm,
  KernelDiffusionInfo &info,
  float clusterSigmaRatio
) {
  info = KernelDiffusionInfo{};
  if (!params.printDiffusionEnabled) {
    return {};
  }
  const DiffusionSettings settings{
    params.printDiffusionFamily,
    params.printDiffusionStrength,
    params.printDiffusionSpatialScale,
    params.printDiffusionHaloWarmth,
    params.printDiffusionCoreIntensity,
    params.printDiffusionCoreSize,
    params.printDiffusionHaloIntensity,
    params.printDiffusionHaloSize,
    params.printDiffusionBloomIntensity,
    params.printDiffusionBloomSize
  };
  return makeDiffusionComponents(settings, pixelSizeUm, info, clusterSigmaRatio);
}

KernelDirInfo makeDirInfo(const ProfileCurveSet &filmCurves, const RenderParams &params) {
  const float amount = std::max(params.dirCouplersAmount, 0.0f);
  const float sameLayer = std::max(params.dirCouplersInhibitionSameLayer, 0.0f);
  const float interlayer = std::max(params.dirCouplersInhibitionInterlayer, 0.0f);
  KernelDirInfo info{};
  info.matrix00 = params.dirCouplersGammaSameLayerR * sameLayer * amount;
  info.matrix11 = params.dirCouplersGammaSameLayerG * sameLayer * amount;
  info.matrix22 = params.dirCouplersGammaSameLayerB * sameLayer * amount;
  info.matrix01 = params.dirCouplersGammaRToG * interlayer * amount;
  info.matrix02 = params.dirCouplersGammaRToB * interlayer * amount;
  info.matrix10 = params.dirCouplersGammaGToR * interlayer * amount;
  info.matrix12 = params.dirCouplersGammaGToB * interlayer * amount;
  info.matrix20 = params.dirCouplersGammaBToR * interlayer * amount;
  info.matrix21 = params.dirCouplersGammaBToG * interlayer * amount;
  const std::array<float, 3> maxima = densityCurveMaximums(filmCurves);
  info.densityMax0 = maxima[0];
  info.densityMax1 = maxima[1];
  info.densityMax2 = maxima[2];
  return info;
}

KernelGaussianBlurInfo makeGaussianBlurInfo(float sigma, uint32_t radiusLimit) {
  KernelGaussianBlurInfo info{};
  if (sigma <= 1.0e-4f) {
    return info;
  }
  const uint32_t radius = std::min<uint32_t>(
    static_cast<uint32_t>(std::ceil(3.0f * sigma)),
    radiusLimit
  );
  info.radius = radius;
  info.active = 1u;
  const double sigmaDouble = static_cast<double>(std::max(sigma, 1.0e-6f));
  const double invSigma2 = 1.0 / std::max(sigmaDouble * sigmaDouble, 1.0e-8);
  info.firstWeight = static_cast<float>(std::exp(-0.5 * invSigma2));
  info.firstRatio = static_cast<float>(std::exp(-1.5 * invSigma2));
  info.ratioStep = static_cast<float>(std::exp(-invSigma2));
  double weight = info.firstWeight;
  double ratio = info.firstRatio;
  double weightSum = 1.0;
  for (uint32_t offset = 1u; offset <= radius; ++offset) {
    weightSum += 2.0 * weight;
    weight *= ratio;
    ratio *= info.ratioStep;
  }
  info.invWeightSum = static_cast<float>(1.0 / std::max(weightSum, 1.0e-8));
  return info;
}

std::array<KernelGaussianBlurInfo, 3> makeDirTailBlurInfos(const KernelParams &params) {
  constexpr std::array<float, 3> kTailSigmaScale = {0.5360f, 1.5236f, 2.7684f};
  std::array<KernelGaussianBlurInfo, 3> infos{};
  const float pixelSizeUm = std::max(params.filmPixelSizeUm, 1.0e-6f);
  const float tailUm = std::max(params.dirCouplersDiffusionTailUm, 0.0f);
  for (size_t component = 0; component < infos.size(); ++component) {
    infos[component] = makeGaussianBlurInfo(tailUm * kTailSigmaScale[component] / pixelSizeUm, 256u);
  }
  return infos;
}

std::vector<float> makeDirCorrectedDensityCurves(
  const ProfileCurveSet &filmCurves,
  const KernelDirInfo &dirInfo
) {
  const bool positive = std::strcmp(filmCurves.type, "positive") == 0;
  std::vector<float> corrected(static_cast<size_t>(filmCurves.exposureCount) * 3u, 0.0f);
  for (uint32_t receiver = 0u; receiver < 3u; ++receiver) {
    std::vector<float> logExposure0(filmCurves.exposureCount, 0.0f);
    for (uint32_t i = 0u; i < filmCurves.exposureCount; ++i) {
      const float d0 = filmCurves.densityCurves[i * 3u];
      const float d1 = filmCurves.densityCurves[i * 3u + 1u];
      const float d2 = filmCurves.densityCurves[i * 3u + 2u];
      const float silver0 = positive ? dirInfo.densityMax0 - d0 : d0;
      const float silver1 = positive ? dirInfo.densityMax1 - d1 : d1;
      const float silver2 = positive ? dirInfo.densityMax2 - d2 : d2;
      float amount = 0.0f;
      if (receiver == 0u) {
        amount = silver0 * dirInfo.matrix00 + silver1 * dirInfo.matrix10 + silver2 * dirInfo.matrix20;
      } else if (receiver == 1u) {
        amount = silver0 * dirInfo.matrix01 + silver1 * dirInfo.matrix11 + silver2 * dirInfo.matrix21;
      } else {
        amount = silver0 * dirInfo.matrix02 + silver1 * dirInfo.matrix12 + silver2 * dirInfo.matrix22;
      }
      logExposure0[i] = filmCurves.logExposure[i] - amount;
    }
    for (uint32_t i = 0u; i < filmCurves.exposureCount; ++i) {
      corrected[i * 3u + receiver] = interpLinear(
        logExposure0,
        filmCurves.densityCurves,
        receiver,
        filmCurves.logExposure[i]
      );
    }
  }
  return corrected;
}

bool StaticProfileResourceData::validFor(const RenderParams &params) const {
  return film == params.film &&
         paper == params.paper &&
         rgbToRawMethod == params.rgbToRawMethod &&
         cameraUvFilterEnabled == params.cameraUvFilterEnabled &&
         cameraUvCutNm == params.cameraUvCutNm &&
         cameraIrFilterEnabled == params.cameraIrFilterEnabled &&
         cameraIrCutNm == params.cameraIrCutNm &&
         filmCurves &&
         paperCurves &&
         !logExposure.empty() &&
         !densityCurves.empty() &&
         !wavelengths.empty() &&
         !logSensitivity.empty() &&
         !bandpassHanatos.empty() &&
         !hanatosRawResponse.empty() &&
         !inputToReferenceXyz.empty() &&
         !inputToSrgb.empty() &&
         !colorDecodeLut.empty() &&
         !colorTransferKind.empty() &&
         !paperLogExposure.empty() &&
         !paperDensityCurves.empty() &&
         !filmChannelDensity.empty() &&
         !filmBaseDensity.empty() &&
         !paperLogSensitivity.empty() &&
         !thKg3Illuminant.empty() &&
         !customEnlargerFilters.empty() &&
         !neutralPrintFilters.empty() &&
         !academyPrinterDensityData.empty() &&
         !paperScanDensityData.empty() &&
         !scanIlluminantsAndCmfs.empty() &&
         !scanToOutputRgbData.empty() &&
         !colorEncodeLut.empty();
}

void StaticProfileResourceData::reset() {
  *this = StaticProfileResourceData{};
}

bool buildStaticProfileResourceData(
  const RenderParams &params,
  const std::vector<float> &hanatosSpectraData,
  StaticProfileResourceData &resources,
  std::string &error
) {
  if (resources.validFor(params)) {
    return true;
  }

  StaticProfileResourceData next{};
  const ProfileCurveSet *filmCurves = selectedFilmCurves(params);
  const ProfileCurveSet *paperCurves = selectedPaperCurves(params);
  if (!filmCurves || filmCurves->exposureCount == 0 || !filmCurves->logExposure || !filmCurves->densityCurves) {
    error = "Unable to locate generated film density curves.";
    return false;
  }
  if (!paperCurves || paperCurves->exposureCount == 0 || !paperCurves->logExposure || !paperCurves->densityCurves) {
    error = "Unable to locate generated paper density curves.";
    return false;
  }
  if (filmCurves->wavelengthCount == 0 || !filmCurves->wavelengths || !filmCurves->logSensitivity) {
    error = "Unable to locate generated film spectral data.";
    return false;
  }
  if (params.rgbToRawMethod == RgbToRawMethod::Hanatos2025 && !filmCurves->bandpassHanatos2025) {
    error = "Unable to locate archived Hanatos 2025 film bandpass data.";
    return false;
  }
  if (params.rgbToRawMethod == RgbToRawMethod::Hanatos2026 &&
      (!filmCurves->hanatos2026WindowParams || !filmCurves->referenceIlluminantSpectrum)) {
    error = "Unable to locate generated Hanatos 2026 film adaptation data.";
    return false;
  }
  if (paperCurves->wavelengthCount != filmCurves->wavelengthCount || !paperCurves->logSensitivity) {
    error = "Unable to locate generated paper spectral data.";
    return false;
  }
  if (!filmCurves->channelDensity || !filmCurves->baseDensity || !filmCurves->densityCurveMinimum ||
      !filmCurves->densityCurveLayers || !filmCurves->densityCurveLayerMaxima) {
    error = "Unable to locate generated film spectral density data.";
    return false;
  }
  if (!filmCurves->halationStrength || !filmCurves->halationFirstSigmaUm) {
    error = "Unable to locate generated film halation data.";
    return false;
  }
  if (!filmCurves->scanIlluminant || !filmCurves->scanToOutputRgb || !standardObserverCmfs()) {
    error = "Unable to locate generated film scan data.";
    return false;
  }
  if (!filmCurves->inputToReferenceXyz || !filmCurves->inputToSrgb ||
      !filmCurves->mallettBasisIlluminant || filmCurves->mallettRawMidgrayGreen <= 0.0f) {
    error = "Unable to locate generated RGB-to-raw reference data.";
    return false;
  }
  if (!paperCurves->channelDensity || !paperCurves->baseDensity ||
      !paperCurves->scanIlluminant || !paperCurves->scanToOutputRgb) {
    error = "Unable to locate generated paper scan data.";
    return false;
  }
  if (!thKg3Illuminant() || !customEnlargerFilters() || !neutralPrintFilters() || !academyPrinterDensityData()) {
    error = "Unable to locate generated print exposure data.";
    return false;
  }
  if (!colorDecodeLuts() || !colorEncodeLuts() || !colorTransferKinds() || !inputMeterXyzMatrices()) {
    error = "Unable to locate generated color transform data.";
    return false;
  }

  const HanatosSpectraLutInfo &hanatos = hanatosSpectraLutInfo();
  const size_t expectedHanatos =
    static_cast<size_t>(hanatos.width) * static_cast<size_t>(hanatos.height) *
    static_cast<size_t>(hanatos.wavelengthCount);
  if (expectedHanatos == 0 || hanatosSpectraData.size() < expectedHanatos) {
    error = "Unable to locate Hanatos spectra LUT resource.";
    return false;
  }

  next.film = params.film;
  next.paper = params.paper;
  next.rgbToRawMethod = params.rgbToRawMethod;
  next.cameraUvFilterEnabled = params.cameraUvFilterEnabled;
  next.cameraUvCutNm = params.cameraUvCutNm;
  next.cameraIrFilterEnabled = params.cameraIrFilterEnabled;
  next.cameraIrCutNm = params.cameraIrCutNm;
  next.filmCurves = filmCurves;
  next.paperCurves = paperCurves;
  next.curveInfo = {filmCurves->exposureCount, 0u, 0u, 0u};
  next.paperCurveInfo = {paperCurves->exposureCount, 0u, 0u, 0u};
  next.colorInfo = {
    kSpektraColorSpaceCount,
    kSpektraColorTransferLutSize,
    colorDecodeLutMin(),
    colorDecodeLutMax(),
    colorEncodeLutMin(),
    colorEncodeLutMax(),
    0.0f,
    0.0f
  };
  next.spectralInfo.filmWavelengthCount = filmCurves->wavelengthCount;
  next.spectralInfo.hanatosWidth = hanatos.width;
  next.spectralInfo.hanatosHeight = hanatos.height;
  next.spectralInfo.hanatosWavelengthCount = hanatos.wavelengthCount;
  next.spectralInfo.filmCount = kSpektraFilmCount;
  next.spectralInfo.paperCount = kSpektraPaperCount;
  next.spectralInfo.filmPositive = (filmCurves->type && std::strcmp(filmCurves->type, "positive") == 0) ? 1u : 0u;
  next.spectralInfo.mallettRawMidgrayGreen = filmCurves->mallettRawMidgrayGreen;
  next.spectralInfo.filmDensityCurveMinimum0 = filmCurves->densityCurveMinimum[0];
  next.spectralInfo.filmDensityCurveMinimum1 = filmCurves->densityCurveMinimum[1];
  next.spectralInfo.filmDensityCurveMinimum2 = filmCurves->densityCurveMinimum[2];
  const std::array<float, 3> filmDensityCurveMaximum = densityCurveMaximums(*filmCurves);
  next.spectralInfo.filmDensityCurveMaximum0 = filmDensityCurveMaximum[0];
  next.spectralInfo.filmDensityCurveMaximum1 = filmDensityCurveMaximum[1];
  next.spectralInfo.filmDensityCurveMaximum2 = filmDensityCurveMaximum[2];
  const std::array<float, 3> paperDensityCurveMaximum = densityCurveMaximums(*paperCurves);
  next.spectralInfo.paperDensityCurveMaximum0 = paperDensityCurveMaximum[0];
  next.spectralInfo.paperDensityCurveMaximum1 = paperDensityCurveMaximum[1];
  next.spectralInfo.paperDensityCurveMaximum2 = paperDensityCurveMaximum[2];

  const size_t wavelengthCount = filmCurves->wavelengthCount;
  const size_t filmExposureCount = filmCurves->exposureCount;
  const size_t paperExposureCount = paperCurves->exposureCount;
  const size_t sensitivityCount = wavelengthCount * 3u;
  const size_t neutralPrintFilterCount =
    static_cast<size_t>(kSpektraPaperCount) * static_cast<size_t>(kSpektraFilmCount) * 3u;
  const size_t inputMatrixCount = static_cast<size_t>(kSpektraColorSpaceCount) * 9u;
  const size_t transferLutCount =
    static_cast<size_t>(kSpektraColorSpaceCount) * static_cast<size_t>(kSpektraColorTransferLutSize);

  const std::vector<float> baseFilmSensitivityLinear =
    makeLinearSensitivity(filmCurves->logSensitivity, filmCurves->wavelengthCount);
  const std::vector<float> filmSensitivityLinear =
    applyCameraBandPass(*filmCurves, baseFilmSensitivityLinear, params);
  const std::vector<float> paperSensitivityLinear =
    makeLinearSensitivity(paperCurves->logSensitivity, paperCurves->wavelengthCount);
  next.mallettBasisIlluminant = makeMallettRawMatrix(*filmCurves, filmSensitivityLinear);
  next.hanatosRawResponse = makeHanatosRawResponse(
    *filmCurves,
    filmSensitivityLinear,
    hanatosSpectraData,
    hanatos,
    params.rgbToRawMethod
  );

  assignFloats(next.logExposure, filmCurves->logExposure, filmExposureCount);
  assignFloats(next.densityCurves, filmCurves->densityCurves, filmExposureCount * 3u);
  assignFloats(next.wavelengths, filmCurves->wavelengths, wavelengthCount);
  assignFloats(next.logSensitivity, filmSensitivityLinear.data(), filmSensitivityLinear.size());
  if (filmCurves->bandpassHanatos2025) {
    assignFloats(next.bandpassHanatos, filmCurves->bandpassHanatos2025, sensitivityCount);
  } else {
    next.bandpassHanatos.assign(sensitivityCount, 1.0f);
  }
  assignFloats(next.inputToReferenceXyz, filmCurves->inputToReferenceXyz, inputMatrixCount);
  assignFloats(next.inputToSrgb, filmCurves->inputToSrgb, inputMatrixCount);
  assignFloats(next.colorDecodeLut, colorDecodeLuts(), transferLutCount);
  assignUInts(next.colorTransferKind, colorTransferKinds(), kSpektraColorSpaceCount);
  assignFloats(next.paperLogExposure, paperCurves->logExposure, paperExposureCount);
  assignFloats(next.paperDensityCurves, paperCurves->densityCurves, paperExposureCount * 3u);
  assignFloats(next.filmChannelDensity, filmCurves->channelDensity, sensitivityCount);
  assignFloats(next.filmBaseDensity, filmCurves->baseDensity, wavelengthCount);
  assignFloats(next.paperLogSensitivity, paperSensitivityLinear.data(), paperSensitivityLinear.size());
  assignFloats(next.thKg3Illuminant, thKg3Illuminant(), wavelengthCount);
  assignFloats(next.customEnlargerFilters, customEnlargerFilters(), sensitivityCount);
  assignFloats(next.neutralPrintFilters, neutralPrintFilters(), neutralPrintFilterCount);
  assignFloats(next.academyPrinterDensityData, academyPrinterDensityData(), sensitivityCount + neutralPrintFilterCount);
  assignFloats(next.colorEncodeLut, colorEncodeLuts(), transferLutCount);

  next.paperScanDensityData.reserve(wavelengthCount * 4u + filmExposureCount * 9u + 9u);
  next.paperScanDensityData.insert(next.paperScanDensityData.end(), paperCurves->channelDensity, paperCurves->channelDensity + sensitivityCount);
  next.paperScanDensityData.insert(next.paperScanDensityData.end(), paperCurves->baseDensity, paperCurves->baseDensity + wavelengthCount);
  next.paperScanDensityData.insert(next.paperScanDensityData.end(), filmCurves->densityCurveLayers, filmCurves->densityCurveLayers + filmExposureCount * 9u);
  next.paperScanDensityData.insert(next.paperScanDensityData.end(), filmCurves->densityCurveLayerMaxima, filmCurves->densityCurveLayerMaxima + 9u);

  next.scanIlluminantsAndCmfs.reserve(wavelengthCount * 5u);
  next.scanIlluminantsAndCmfs.insert(next.scanIlluminantsAndCmfs.end(), filmCurves->scanIlluminant, filmCurves->scanIlluminant + wavelengthCount);
  next.scanIlluminantsAndCmfs.insert(next.scanIlluminantsAndCmfs.end(), paperCurves->scanIlluminant, paperCurves->scanIlluminant + wavelengthCount);
  next.scanIlluminantsAndCmfs.insert(next.scanIlluminantsAndCmfs.end(), standardObserverCmfs(), standardObserverCmfs() + wavelengthCount * 3u);

  next.scanToOutputRgbData.reserve(static_cast<size_t>(kSpektraColorSpaceCount) * 18u);
  next.scanToOutputRgbData.insert(next.scanToOutputRgbData.end(), filmCurves->scanToOutputRgb, filmCurves->scanToOutputRgb + inputMatrixCount);
  next.scanToOutputRgbData.insert(next.scanToOutputRgbData.end(), paperCurves->scanToOutputRgb, paperCurves->scanToOutputRgb + inputMatrixCount);

  if (!next.validFor(params)) {
    error = "Unable to create static renderer resources.";
    return false;
  }
  resources = std::move(next);
  error.clear();
  return true;
}

bool validateRgbaFloatOrHalfImages(
  const ImageView &source,
  const MutableImageView &destination,
  std::string &error
) {
  if (!source.data || !destination.data) {
    error = "Renderer received a null image pointer.";
    return false;
  }
  if (source.components != 4 || destination.components != 4) {
    error = "SpektraFilm requires RGBA input and output clips.";
    return false;
  }
  if ((source.bytesPerComponent != 2 && source.bytesPerComponent != 4) ||
      (destination.bytesPerComponent != 2 && destination.bytesPerComponent != 4)) {
    error = "SpektraFilm requires 16-bit float or 32-bit float RGBA images.";
    return false;
  }
  if (source.rowBytes <= 0 || destination.rowBytes <= 0 ||
      source.width < 0 || source.height < 0 ||
      destination.width < 0 || destination.height < 0) {
    error = "Renderer received an invalid image layout.";
    return false;
  }
  return true;
}

EffectiveRenderWindow intersectRenderWindow(
  const ImageView &source,
  const MutableImageView &destination,
  const RenderWindow &window
) {
  EffectiveRenderWindow out{};
  out.x1 = std::max({window.x1, source.x1, destination.x1});
  out.y1 = std::max({window.y1, source.y1, destination.y1});
  out.x2 = std::min({window.x2, source.x1 + source.width, destination.x1 + destination.width});
  out.y2 = std::min({window.y2, source.y1 + source.height, destination.y1 + destination.height});
  return out;
}

float halfToFloat(uint16_t value) {
  const uint32_t bits = halfToFloatBits(value);
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(out));
  return out;
}

uint16_t floatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  const uint32_t mantissa = bits & 0x007fffffu;
  if (exponent <= 0) {
    if (exponent < -10) {
      return static_cast<uint16_t>(sign);
    }
    const uint32_t shifted = (mantissa | 0x00800000u) >> static_cast<uint32_t>(1 - exponent);
    return static_cast<uint16_t>(sign | ((shifted + 0x00001000u) >> 13));
  }
  if (exponent >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(
    sign |
    (static_cast<uint32_t>(exponent) << 10) |
    ((mantissa + 0x00001000u) >> 13)
  );
}

void copySourceToFloatStaging(
  const ImageView &source,
  const EffectiveRenderWindow &window,
  int32_t width,
  int32_t height,
  float *destination
) {
  const auto *base = static_cast<const unsigned char *>(source.data);
  for (int32_t y = 0; y < height; ++y) {
    const auto *row = base +
      static_cast<ptrdiff_t>(window.y1 + y - source.y1) * source.rowBytes +
      static_cast<ptrdiff_t>(window.x1 - source.x1) * source.components * source.bytesPerComponent;
    for (int32_t x = 0; x < width; ++x) {
      float *dst = destination + (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
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
}

void copyFloatStagingToDestination(
  const float *source,
  const MutableImageView &destination,
  const EffectiveRenderWindow &window,
  int32_t width,
  int32_t height
) {
  auto *base = static_cast<unsigned char *>(destination.data);
  for (int32_t y = 0; y < height; ++y) {
    auto *row = base +
      static_cast<ptrdiff_t>(window.y1 + y - destination.y1) * destination.rowBytes +
      static_cast<ptrdiff_t>(window.x1 - destination.x1) * destination.components * destination.bytesPerComponent;
    for (int32_t x = 0; x < width; ++x) {
      const float *src = source + (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4u;
      if (destination.bytesPerComponent == 4) {
        auto *dst = reinterpret_cast<float *>(row + static_cast<size_t>(x) * 4u * sizeof(float));
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = src[3];
      } else {
        auto *dst = reinterpret_cast<uint16_t *>(row + static_cast<size_t>(x) * 4u * sizeof(uint16_t));
        dst[0] = floatToHalf(src[0]);
        dst[1] = floatToHalf(src[1]);
        dst[2] = floatToHalf(src[2]);
        dst[3] = floatToHalf(src[3]);
      }
    }
  }
}

} // namespace spektrafilm
