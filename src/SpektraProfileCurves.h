#pragma once

#include <cstdint>

#if __has_include("SpektraGeneratedProfileCounts.h")
#  include "SpektraGeneratedProfileCounts.h"
#endif

namespace spektrafilm {

constexpr uint32_t kSpektraColorSpaceCount = 29u;
constexpr uint32_t kSpektraColorTransferLutSize = 4096u;
constexpr uint32_t kSpektraOutputGamutCompressionStride = 18u;
constexpr uint32_t kSpektraOutputGamutCompressionElementCount =
  kSpektraColorSpaceCount * kSpektraOutputGamutCompressionStride;

#ifndef SPEKTRA_GENERATED_PROFILE_COUNTS
constexpr uint32_t kSpektraFilmCount = 20u;
constexpr uint32_t kSpektraPaperCount = 7u;
constexpr int32_t kSpektraDefaultFilmIndex = 2;
constexpr int32_t kSpektraDefaultPaperIndex = 3;
constexpr bool kSpektraAcademyPrinterDensityEnabled = true;
#endif

struct ProfileCurveSet {
  const char *stock = nullptr;
  const char *name = nullptr;
  const char *type = nullptr;
  const char *referenceIlluminant = nullptr;
  uint32_t wavelengthCount = 0;
  uint32_t exposureCount = 0;
  const float *wavelengths = nullptr;
  const float *logSensitivity = nullptr;
  const float *bandpassHanatos2025 = nullptr;
  const float *hanatos2026WindowParams = nullptr;
  const float *referenceIlluminantSpectrum = nullptr;
  const float *inputToReferenceXyz = nullptr;
  const float *inputToSrgb = nullptr;
  const float *mallettBasisIlluminant = nullptr;
  float mallettRawMidgrayGreen = 1.0f;
  const float *logExposure = nullptr;
  const float *densityCurves = nullptr;
  const float *channelDensity = nullptr;
  const float *baseDensity = nullptr;
  const float *densityCurveMinimum = nullptr;
  const float *densityCurveLayers = nullptr;
  const float *densityCurveLayerMaxima = nullptr;
  const float *halationStrength = nullptr;
  const float *halationFirstSigmaUm = nullptr;
  const float *dirGammaSameLayerRgb = nullptr;
  const float *dirGammaRToGb = nullptr;
  const float *dirGammaGToRb = nullptr;
  const float *dirGammaBToRg = nullptr;
  const float *scanIlluminant = nullptr;
  const float *scanToOutputRgb = nullptr;
};

struct HanatosSpectraLutInfo {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t wavelengthCount = 0;
  uint32_t elementCount = 0;
};

const ProfileCurveSet *filmProfileCurves(int32_t index);
const ProfileCurveSet *paperProfileCurves(int32_t index);
const HanatosSpectraLutInfo &hanatosSpectraLutInfo();
const float *inputMeterXyzMatrices();
const uint32_t *colorTransferKinds();
const float *colorTransferParams();
const char *colorSpaceLabel(int32_t index);
const float *colorDecodeLuts();
const float *colorEncodeLuts();
const float *standardObserverCmfs();
const float *thKg3Illuminant();
const float *customEnlargerFilters();
const float *neutralPrintFilters();
const float *academyPrinterDensityResponsivities();
const float *academyPrinterDensityNeutralOffsets();
const float *academyPrinterDensityData();
const float *academyPrinterDensityInfluxSpectrum();
float colorDecodeLutMin();
float colorDecodeLutMax();
float colorEncodeLutMin();
float colorEncodeLutMax();

} // namespace spektrafilm
