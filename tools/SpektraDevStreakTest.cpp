// SpektraDevStreakTest: renders a flat S-Log3 / S-Gamut3.Cine patch through the Vulkan
// renderer for N consecutive frames and dumps float32 RGB frames for analysis.
//   SpektraDevStreakTest <out.bin> <frames> <width> <height> <slog3 code> <streak 0|1> <strength 0..1> <filmFormat> [grain 0|1]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include "SpektraVulkanRenderer.h"

int main(int argc, char **argv) {
  if (argc < 9) {
    std::fprintf(stderr, "usage: %s out.bin frames width height code streak strength format [grain]\n", argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const int frames = std::atoi(argv[2]);
  const int width = std::atoi(argv[3]);
  const int height = std::atoi(argv[4]);
  const float code = static_cast<float>(std::atof(argv[5]));
  const bool streak = std::atoi(argv[6]) != 0;
  const float strength = static_cast<float>(std::atof(argv[7]));
  const int format = std::atoi(argv[8]);
  const bool grain = argc > 9 && std::atoi(argv[9]) != 0;

  spektrafilm::VulkanRenderer renderer;
  if (!renderer.isAvailable()) {
    std::fprintf(stderr, "renderer unavailable: %s\n", renderer.lastError().c_str());
    return 1;
  }
  const int rowBytes = width * 4 * 4;
  std::vector<float> src(static_cast<size_t>(width) * height * 4, code);
  for (size_t i = 3; i < src.size(); i += 4) src[i] = 1.0f;
  std::vector<float> dst(src.size(), 0.0f);
  spektrafilm::ImageView sv{}; sv.data = src.data(); sv.width = width; sv.height = height; sv.rowBytes = rowBytes; sv.components = 4; sv.bytesPerComponent = 4;
  spektrafilm::MutableImageView dv{}; dv.data = dst.data(); dv.width = width; dv.height = height; dv.rowBytes = rowBytes; dv.components = 4; dv.bytesPerComponent = 4;
  spektrafilm::RenderWindow window{0, 0, width, height};

  spektrafilm::RenderParams params{};
  params.inputColorSpace = spektrafilm::ColorSpace::SonySLog3SGamut3Cine;
  params.filmFormat = static_cast<spektrafilm::FilmFormat>(format);
  params.devStreakEnabled = streak;
  params.devStreakStrength = strength;
  params.devStreakSeed = 7;
  params.devStreakAnimate = true;
  params.grainEnabled = grain;
  params.grainAnimate = true;

  FILE *f = std::fopen(out.c_str(), "wb");
  for (int t = 0; t < frames; ++t) {
    if (!renderer.render(sv, dv, window, params, static_cast<double>(t))) {
      std::fprintf(stderr, "render failed at frame %d: %s\n", t, renderer.lastError().c_str());
      return 1;
    }
    for (size_t i = 0; i < dst.size(); i += 4) std::fwrite(&dst[i], sizeof(float), 3, f);
  }
  std::fclose(f);
  std::fprintf(stderr, "ok %d frames\n", frames);
  return 0;
}
