// SpektraDevStreakField.h
// Host-side generator for developer-streak fields (calibrated against measured reference renders).
// Produces a small, frame-relative grid of per-channel exposure offsets in STOPS.
// Stateless: the field for frame N depends only on (seed, N) so out-of-order /
// parallel rendering in Resolve is safe. Cost: ~1 ms per frame on one CPU core.
//
// The shader (SpektraDevStreak.comp) upsamples the grid bicubically and adds
//   logRaw += stops * log10(2) * toeWeight(logRaw)
// to the film log-exposure buffer right before SpektraCurveDevelop, so both
// development (and DIR re-development) and grain see the streaked latent image.
#pragma once
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace spektrafilm {

struct DevStreakSettings {
  bool enabled = false;
  float strength = 0.5f;          // UI 0..1 slider (see strengthCurve)
  float formatLongEdgeMm = 24.89f; // gate width, drives amplitude via gauge table
  uint32_t seed = 1u;
  int64_t frame = 0;               // integer frame index (use 0 when not animated)
  // Advanced (1.0 = calibrated default)
  float amount = 1.0f;             // extra multiplier on the whole effect
  float widthScale = 1.0f;         // >1 = wider streaks (scales horizontal spectrum)
  float lengthScale = 1.0f;        // >1 = streaks stay coherent over more of the frame height
  float flicker = 1.0f;            // multiplier on the per-frame global (DC) component
  float colorVariation = 1.0f;     // 0 = neutral streaks, 1 = calibrated inter-layer decorrelation
  float temporalHold = 1.0f;       // multiplier on the slow AR component's time constant
};

// ---- Calibration constants measured from reference renders (Super 35, full calibrated strength) ----
namespace devstreak {
constexpr int kGridX = 64, kGridY = 36, kDom = 4;         // grid samples per frame, synthesis domain factor
constexpr float kPxP = 1.65f, kPxKc = 7.0f, kPxK0 = 0.6f;  // Px(k)=k^-p exp(-(k/kc)^2) k^4/(k^4+k0^4), k=cycles/frame width
constexpr float kLy = 0.50f, kNu = 1.0f;                   // vertical Matern(nu) length, fraction of frame height
constexpr float kArW[2] = {0.379f, 0.621f};                 // temporal: fast + slow AR(1) variance weights
constexpr float kArPhi[2] = {0.200f, 0.753f};
constexpr float kAmpSpatial[3] = {0.01443f, 0.01375f, 0.01529f}; // stops per unit field (R,G,B)
constexpr float kAmpDc[3] = {0.0078f, 0.0046f, 0.0107f};          // stops, per-frame global component
constexpr float kBiasPerRms = -0.30f;                             // mean darkening = -0.30 * field rms
constexpr float kRmsPerUnit = 0.715f;
// Input (latent-image) layer correlation. Slightly below the measured output values (.72/.36/.75)
// because Spektrafilm's print stage adds inter-channel crosstalk; measured end-to-end it matches.
constexpr float kCorrSpatial[3][3] = {{1.f, .67f, .28f}, {.67f, 1.f, .67f}, {.28f, .67f, 1.f}};
constexpr float kCorrDc[3][3] = {{1.f, .05f, .00f}, {.05f, 1.f, .40f}, {.00f, .40f, 1.f}};

// Strength slider: UI 0..1 is stretched 1.7x onto the calibrated response curve (knee at q = 0.5),
// so the UI 50 % default gives 0.775x the calibration amplitude and UI 100 % gives 2.05x.
constexpr float kStrengthSliderStretch = 1.7f;
inline float strengthCurve(float p) {  // calibrated response, knee at q = 0.5
  const float q = std::clamp(p, 0.0f, 1.0f) * kStrengthSliderStretch;
  return q <= 0.5f ? 0.5f * q : 0.25f + 1.5f * (q - 0.5f);
}

// Gauge (gate long edge, mm) -> amplitude multiplier relative to Super 35.
// Measured: 8mm 1.72, 16mm 1.95, S35 1.00, 35mm FF 0.72, 65mm 0.50. Log-log interpolation.
inline float gaugeScale(float mm) {
  static const float x[] = {4.80f, 10.26f, 24.89f, 35.0f, 52.48f};
  static const float y[] = {1.72f, 1.95f, 1.00f, 0.72f, 0.50f};
  const int n = 5;
  mm = std::max(mm, 1.0f);
  if (mm <= x[0]) return y[0];
  int i = 0;
  while (i < n - 2 && mm > x[i + 1]) ++i;  // beyond last point: extrapolate last segment
  const float t = std::log(mm / x[i]) / std::log(x[i + 1] / x[i]);
  return std::exp(std::log(y[i]) + t * (std::log(y[i + 1]) - std::log(y[i])));
}

inline uint64_t splitmix64(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
// Counter-based standard normal pair (Box-Muller)
inline void normalPair(uint64_t key, float &a, float &b) {
  const uint64_t h1 = splitmix64(key), h2 = splitmix64(key ^ 0xD1B54A32D192ED03ull);
  const double u1 = ((h1 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  const double u2 = ((h2 >> 11) + 0.5) * (1.0 / 9007199254740992.0);
  const double r = std::sqrt(-2.0 * std::log(u1));
  a = float(r * std::cos(6.283185307179586 * u2));
  b = float(r * std::sin(6.283185307179586 * u2));
}
inline uint64_t mixKey(uint64_t seed, int64_t frame, uint64_t stream, uint64_t index) {
  return splitmix64(splitmix64(splitmix64(seed * 0x100000001B3ull ^ uint64_t(frame)) ^ stream) ^ index);
}
// Stateless temporal process: sum of AR(1) components expressed as finite weighted
// sums of per-frame white noise. Returns a complex unit-variance-per-component sample.
inline void temporalNormal(uint64_t seed, int64_t frame, uint64_t stream, uint64_t index,
                           float hold, float &re, float &im) {
  re = im = 0.f;
  for (int c = 0; c < 2; ++c) {
    float phi = kArPhi[c];
    if (c == 1 && hold != 1.0f) phi = std::pow(phi, 1.0f / std::max(hold, 0.05f));
    const int n = phi > 0.f ? std::min(40, int(std::ceil(std::log(1e-2f) / std::log(phi)))) : 0;
    const float scale = std::sqrt(kArW[c] * (1.f - phi * phi));
    float w = 1.f;
    for (int j = 0; j <= n; ++j) {
      float a, b;
      normalPair(mixKey(seed, frame - j, stream * 2 + c, index), a, b);
      re += scale * w * a;
      im += scale * w * b;
      w *= phi;
    }
  }
}
inline void cholesky3(const float c[3][3], float L[3][3]) {
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) L[i][j] = 0.f;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j) {
      float s = c[i][j];
      for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
      L[i][j] = (i == j) ? std::sqrt(std::max(s, 1e-9f)) : s / L[j][j];
    }
}
}  // namespace devstreak

// Gate long edge in mm for Spektrafilm's FilmFormat enum (same table as the plugin's filmFormatMm).
inline float devStreakFormatLongEdgeMm(int32_t filmFormat) {
  switch (filmFormat) {
    case 0: return 4.8f;    // 8mm
    case 1: return 5.79f;   // Super 8
    case 2: return 10.26f;  // 16mm
    case 3: return 12.52f;  // Super 16
    case 5: return 24.89f;  // Super 35
    case 6: return 52.48f;  // 65mm
    case 7: return 70.41f;  // 70mm / IMAX
    case 4:
    default: return 35.0f;  // 35mm
  }
}

// Output: grid[(y*kGridX + x)*3 + c], exposure offset in stops (already includes
// strength, gauge scale, DC flicker and mean bias). Empty vector when disabled.
inline std::vector<float> generateDevStreakField(const DevStreakSettings &s) {
  using namespace devstreak;
  std::vector<float> grid;
  const float k = strengthCurve(s.strength) * gaugeScale(s.formatLongEdgeMm) * std::max(s.amount, 0.f);
  if (!s.enabled || k <= 0.f) return grid;
  grid.assign(size_t(kGridX) * kGridY * 3, 0.f);

  const int NX = kGridX * kDom, NY = kGridY * kDom;
  // Spectral filter over bins (fftfreq convention), truncated where negligible. Cached per (width, length).
  struct Bin { int ix, iy; float kx, ky, f; };
  const float lengthL = kLy * std::max(s.lengthScale, 0.05f);
  const float widthS = std::max(s.widthScale, 0.05f);
  struct FilterCache { float widthS = -1.f, lengthL = -1.f; double sumF2 = 0.0; std::vector<Bin> bins; std::vector<int> cols; };
  thread_local FilterCache cache;
  if (cache.widthS != widthS || cache.lengthL != lengthL) {
    cache.bins.clear(); cache.cols.clear(); cache.sumF2 = 0.0;
    float fmax = 0.f;
    std::vector<Bin> all;
    all.reserve(size_t(NX) * NY);
    for (int iy = 0; iy < NY; ++iy) {
      const float ky = float(iy < (NY + 1) / 2 ? iy : iy - NY) / kDom;   // cycles / frame height
      const float py = std::pow(1.f + std::pow(6.2831853f * ky * lengthL, 2.f), -(kNu + 0.5f));
      for (int ix = 1; ix < NX; ++ix) {                                   // no kx = 0 (DC handled separately)
        const float kx = float(ix < (NX + 1) / 2 ? ix : ix - NX) / kDom * widthS;  // cycles / frame width
        const float ka = std::fabs(kx);
        const float px = std::pow(ka, -kPxP) * std::exp(-(ka / kPxKc) * (ka / kPxKc)) *
                         std::pow(ka, 4.f) / (std::pow(ka, 4.f) + std::pow(kPxK0, 4.f));
        const float f = std::sqrt(px * py);
        cache.sumF2 += double(f) * f;
        fmax = std::max(fmax, f);
        all.push_back({ix, iy, kx / widthS, ky, f});
      }
    }
    for (const Bin &b : all)
      if (b.f > 1e-2f * fmax) cache.bins.push_back(b);  // keeps 99.7 % of the energy
    for (const Bin &b : cache.bins)
      if (std::find(cache.cols.begin(), cache.cols.end(), b.ix) == cache.cols.end()) cache.cols.push_back(b.ix);
    cache.widthS = widthS; cache.lengthL = lengthL;
  }
  const std::vector<Bin> &bins = cache.bins;
  const std::vector<int> &cols = cache.cols;
  const double sumF2 = cache.sumF2;
  const float norm = float(1.0 / std::sqrt(2.0 * sumF2));

  float Ls[3][3], Ld[3][3];
  float cs[3][3], cd[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      const float cv = std::clamp(s.colorVariation, 0.f, 1.f);
      cs[i][j] = i == j ? 1.f : 1.f - cv * (1.f - kCorrSpatial[i][j]);
      cd[i][j] = i == j ? 1.f : 1.f - cv * (1.f - kCorrDc[i][j]);
    }
  cholesky3(cs, Ls);
  cholesky3(cd, Ld);

  // Separable inverse DFT restricted to the cropped [0,kGridX) x [0,kGridY) window.
  // Stage 1: for each bin column ix, accumulate over iy into rows y.
  const double twoPi = 6.283185307179586;
  std::vector<std::array<float, 6>> tmp(cols.size() * kGridY);  // complex per channel
  for (auto &t : tmp) t.fill(0.f);
  for (size_t bi = 0; bi < bins.size(); ++bi) {
    const Bin &b = bins[bi];
    float z[3][2];
    for (int c = 0; c < 3; ++c)
      devstreak::temporalNormal(s.seed, s.frame, 1 + c, uint64_t(b.iy) * NX + b.ix, s.temporalHold, z[c][0], z[c][1]);
    float zc[3][2];
    for (int i = 0; i < 3; ++i) {
      zc[i][0] = zc[i][1] = 0.f;
      for (int j = 0; j <= i; ++j) { zc[i][0] += Ls[i][j] * z[j][0]; zc[i][1] += Ls[i][j] * z[j][1]; }
    }
    const size_t col = size_t(std::find(cols.begin(), cols.end(), b.ix) - cols.begin());  // cols is small (<200)
    for (int y = 0; y < kGridY; ++y) {
      const double ph = twoPi * double(b.iy) * y / NY;
      const float cr = float(std::cos(ph)), si = float(std::sin(ph));
      auto &t = tmp[col * kGridY + y];
      for (int c = 0; c < 3; ++c) {
        t[c * 2 + 0] += b.f * (zc[c][0] * cr - zc[c][1] * si);
        t[c * 2 + 1] += b.f * (zc[c][0] * si + zc[c][1] * cr);
      }
    }
  }
  // Stage 2: over columns into x, keep real part.
  for (size_t ci = 0; ci < cols.size(); ++ci) {
    const int ix = cols[ci];
    for (int x = 0; x < kGridX; ++x) {
      const double ph = twoPi * double(ix) * x / NX;
      const float cr = float(std::cos(ph)), si = float(std::sin(ph));
      for (int y = 0; y < kGridY; ++y) {
        const auto &t = tmp[ci * kGridY + y];
        float *g = &grid[(size_t(y) * kGridX + x) * 3];
        for (int c = 0; c < 3; ++c) g[c] += (t[c * 2] * cr - t[c * 2 + 1] * si) * norm;
      }
    }
  }
  // Global per-frame component + bias, then amplitudes.
  float dcz[3];
  for (int c = 0; c < 3; ++c) {
    float re, im;
    devstreak::temporalNormal(s.seed, s.frame, 7, uint64_t(c), s.temporalHold, re, im);
    dcz[c] = re;
  }
  float dc[3] = {0.f, 0.f, 0.f};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j <= i; ++j) dc[i] += Ld[i][j] * dcz[j];
  // Mean darkening is kept NEUTRAL (same in all layers): a static colour bias
  // (e.g. cool shadows / warm highlights) is a global grade, not a streak property, and is excluded.
  const float neutralBias = kBiasPerRms * kRmsPerUnit * k * (kAmpSpatial[0] + kAmpSpatial[1] + kAmpSpatial[2]) / 3.f;
  for (int c = 0; c < 3; ++c) {
    const float aS = kAmpSpatial[c] * k;
    const float offset = kAmpDc[c] * k * std::max(s.flicker, 0.f) * dc[c] + neutralBias;
    for (size_t p = 0; p < size_t(kGridX) * kGridY; ++p) grid[p * 3 + c] = grid[p * 3 + c] * aS + offset;
  }
  return grid;
}

// Same as generateDevStreakField but memoises the last result per thread, so tiled
// renders of one frame (several calls with identical settings) generate it once.
inline const std::vector<float> &cachedDevStreakField(const DevStreakSettings &s) {
  struct Entry { bool valid = false; DevStreakSettings key; std::vector<float> grid; };
  thread_local Entry entry;
  const auto same = [](const DevStreakSettings &a, const DevStreakSettings &b) {
    return a.enabled == b.enabled && a.strength == b.strength && a.formatLongEdgeMm == b.formatLongEdgeMm &&
           a.seed == b.seed && a.frame == b.frame && a.amount == b.amount && a.widthScale == b.widthScale &&
           a.lengthScale == b.lengthScale && a.flicker == b.flicker && a.colorVariation == b.colorVariation &&
           a.temporalHold == b.temporalHold;
  };
  if (!entry.valid || !same(entry.key, s)) {
    entry.grid = generateDevStreakField(s);
    entry.key = s;
    entry.valid = true;
  }
  return entry.grid;
}

}  // namespace spektrafilm
