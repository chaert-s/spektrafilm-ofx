#include "SpektraCudaRenderer.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int32_t kWidth = 64;
constexpr int32_t kHeight = 40;
constexpr int32_t kComponents = 4;

std::vector<float> makeSource() {
  std::vector<float> pixels(static_cast<size_t>(kWidth) * kHeight * kComponents);
  for (int32_t y = 0; y < kHeight; ++y) {
    for (int32_t x = 0; x < kWidth; ++x) {
      const size_t base = (static_cast<size_t>(y) * kWidth + x) * kComponents;
      pixels[base + 0] = 0.05f + 0.8f * static_cast<float>(x) / static_cast<float>(kWidth - 1);
      pixels[base + 1] = 0.03f + 0.7f * static_cast<float>(y) / static_cast<float>(kHeight - 1);
      pixels[base + 2] = 0.1f + 0.4f * static_cast<float>((x + y) % 17) / 16.0f;
      pixels[base + 3] = 1.0f;
    }
  }
  return pixels;
}

bool validOutput(const std::vector<float> &pixels, const char *label) {
  for (size_t i = 0; i < pixels.size(); ++i) {
    if (!std::isfinite(pixels[i])) {
      std::cerr << label << " produced a non-finite value at " << i << "\n";
      return false;
    }
    if ((i % kComponents) == 3 && std::abs(pixels[i] - 1.0f) > 1.0e-5f) {
      std::cerr << label << " changed alpha at " << i << "\n";
      return false;
    }
  }
  return true;
}

bool hasPass(const spektrafilm::RendererDiagnostics &diagnostics, const std::string &name) {
  for (const auto &pass : diagnostics.passes) {
    if (pass.name == name) {
      return true;
    }
  }
  return false;
}

float maxRgbDifference(const std::vector<float> &a, const std::vector<float> &b) {
  float maximum = 0.0f;
  for (size_t offset = 0; offset + 2u < a.size() && offset + 2u < b.size(); offset += 4u) {
    maximum = std::max(maximum, std::abs(a[offset] - b[offset]));
    maximum = std::max(maximum, std::abs(a[offset + 1u] - b[offset + 1u]));
    maximum = std::max(maximum, std::abs(a[offset + 2u] - b[offset + 2u]));
  }
  return maximum;
}

spektrafilm::RenderParams baseParams() {
  spektrafilm::RenderParams params{};
  params.inputColorSpace = spektrafilm::ColorSpace::DavinciIntermediateWideGamut;
  params.renderOutput = spektrafilm::RenderOutputMode::FinalPreview;
  params.grainEnabled = false;
  return params;
}

bool renderHost(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  const spektrafilm::RenderParams params = baseParams();
  if (!renderer.render(src, dst, window, params, 1.0)) {
    std::cerr << "Host-memory CUDA render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.backendName == "cuda" && diagnostics.passCount > 1 && validOutput(destination, "host render");
}

bool renderDevice(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  const size_t bytes = source.size() * sizeof(float);
  void *deviceSource = nullptr;
  void *deviceDestination = nullptr;
  cudaError_t status = cudaMalloc(&deviceSource, bytes);
  if (status == cudaSuccess) {
    status = cudaMalloc(&deviceDestination, bytes);
  }
  if (status == cudaSuccess) {
    status = cudaMemcpy(deviceSource, source.data(), bytes, cudaMemcpyHostToDevice);
  }
  if (status != cudaSuccess) {
    cudaFree(deviceDestination);
    cudaFree(deviceSource);
    std::cerr << "CUDA device-image setup failed: " << cudaGetErrorString(status) << "\n";
    return false;
  }

  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{
    deviceSource, 0, 0, kWidth, kHeight, rowBytes, kComponents, 4, spektrafilm::ImageMemoryDomain::CudaDevice};
  const spektrafilm::MutableImageView dst{
    deviceDestination, 0, 0, kWidth, kHeight, rowBytes, kComponents, 4, spektrafilm::ImageMemoryDomain::CudaDevice};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  const spektrafilm::RenderParams params = baseParams();
  const bool rendered = renderer.render(src, dst, window, params, 2.0);

  std::vector<float> destination(source.size(), -1.0f);
  if (rendered) {
    status = cudaMemcpy(destination.data(), deviceDestination, bytes, cudaMemcpyDeviceToHost);
  }
  cudaFree(deviceDestination);
  cudaFree(deviceSource);
  if (!rendered || status != cudaSuccess) {
    std::cerr << "CUDA device-image render failed: "
              << (rendered ? cudaGetErrorString(status) : renderer.lastError()) << "\n";
    return false;
  }

  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.backendName == "cuda" &&
    diagnostics.cudaTransferMode.find("host-cuda-device") != std::string::npos &&
    diagnostics.cudaHostToDeviceMs == 0.0 &&
    diagnostics.cudaDeviceToHostMs == 0.0 &&
    validOutput(destination, "device render");
}

bool renderSpatial(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.cameraDiffusionEnabled = true;
  params.printDiffusionEnabled = true;
  params.halationEnabled = true;
  params.dirCouplersAmount = 0.3f;
  params.scannerEnabled = true;
  if (!renderer.render(src, dst, window, params, 3.0)) {
    std::cerr << "CUDA spatial render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.halationPath &&
    diagnostics.cameraDiffusionPath &&
    diagnostics.printDiffusionPath &&
    diagnostics.dirPath &&
    diagnostics.finalPostProcessPath &&
    validOutput(destination, "spatial render");
}

bool renderCameraDiffusionDir(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> baseline(source.size(), -1.0f);
  std::vector<float> coupled(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.cameraDiffusionEnabled = true;
  params.cameraDiffusionStrength = 0.2f;
  params.cameraDiffusionSpatialScale = 0.5f;
  params.dirCouplersDiffusionUm = 7.0f;
  params.dirCouplersDiffusionTailUm = 233.0f;
  params.dirCouplersDiffusionTailWeight = 0.0f;
  params.dirCouplersAmount = 0.0f;
  spektrafilm::MutableImageView baselineDst{baseline.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  if (!renderer.render(src, baselineDst, window, params, 3.1)) {
    std::cerr << "CUDA camera-diffusion DIR baseline failed: " << renderer.lastError() << "\n";
    return false;
  }

  params.dirCouplersAmount = 0.171f;
  spektrafilm::MutableImageView coupledDst{coupled.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  if (!renderer.render(src, coupledDst, window, params, 3.1)) {
    std::cerr << "CUDA camera-diffusion DIR render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  const float difference = maxRgbDifference(baseline, coupled);
  if (!diagnostics.cameraDiffusionPath ||
      !diagnostics.dirPath ||
      !hasPass(diagnostics, "cuda_dir_redevelop") ||
      difference <= 1.0e-6f ||
      difference > 4.0f) {
    std::cerr << "CUDA camera-diffusion DIR response is invalid. max_difference=" << difference << "\n";
    return false;
  }
  return validOutput(coupled, "camera-diffusion DIR render");
}

bool renderDirFilmDensity(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.renderOutput = spektrafilm::RenderOutputMode::FilmDensityCmy;
  params.dirCouplersAmount = 0.25f;
  params.dirCouplersDiffusionUm = 20.0f;
  if (!renderer.render(src, dst, window, params, 3.2)) {
    std::cerr << "CUDA DIR FilmDensityCmy render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.dirPath &&
    hasPass(diagnostics, "cuda_dir_blur_y") &&
    hasPass(diagnostics, "cuda_dir_redevelop") &&
    validOutput(destination, "DIR FilmDensityCmy render");
}

bool renderCameraDiffusionFilmLog(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  // Keep FilmLogRaw on the same camera-diffusion path as the Metal and Vulkan backends.
  std::vector<float> baseline(source.size(), -1.0f);
  std::vector<float> diffused(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.renderOutput = spektrafilm::RenderOutputMode::FilmLogRaw;
  params.dirCouplersAmount = 0.3f;
  params.cameraDiffusionEnabled = false;
  spektrafilm::MutableImageView baselineDst{baseline.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  if (!renderer.render(src, baselineDst, window, params, 3.25)) {
    std::cerr << "CUDA camera-diffusion baseline render failed: " << renderer.lastError() << "\n";
    return false;
  }

  params.cameraDiffusionEnabled = true;
  params.cameraDiffusionStrength = 1.0f;
  params.cameraDiffusionSpatialScale = 1.0f;
  spektrafilm::MutableImageView diffusedDst{diffused.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  if (!renderer.render(src, diffusedDst, window, params, 3.25)) {
    std::cerr << "CUDA camera-diffusion FilmLogRaw render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  const float difference = maxRgbDifference(baseline, diffused);
  if (!diagnostics.cameraDiffusionPath ||
      diagnostics.dirPath ||
      !hasPass(diagnostics, "cuda_camera_diffusion_resolve") ||
      difference <= 1.0e-6f) {
    std::cerr << "CUDA camera diffusion did not affect FilmLogRaw. max_difference=" << difference << "\n";
    return false;
  }
  return validOutput(diffused, "camera-diffusion FilmLogRaw render");
}

bool renderProcessNegative(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.process = spektrafilm::ProcessMode::ProcessNegative;
  params.printDiffusionEnabled = true;
  params.scannerEnabled = false;
  if (!renderer.render(src, dst, window, params, 4.0)) {
    std::cerr << "CUDA process-negative render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return hasPass(diagnostics, "cuda_print_raw_from_negative_light") &&
    hasPass(diagnostics, "cuda_final_from_process_negative") &&
    diagnostics.printDiffusionPath &&
    validOutput(destination, "process-negative render");
}

bool renderRcmAces(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.process = spektrafilm::ProcessMode::ScanNegative;
  params.scanNegativeInvert = true;
  params.outputRole = spektrafilm::OutputRole::Rcm;
  params.outputColorSpace = spektrafilm::ColorSpace::AcesCg;
  params.scannerEnabled = false;
  if (!renderer.render(src, dst, window, params, 5.0)) {
    std::cerr << "CUDA RCM/ACES render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.backendName == "cuda" && diagnostics.passCount > 1 &&
    validOutput(destination, "rcm aces render");
}

bool renderColorAdaptation(spektrafilm::Renderer &renderer, const std::vector<float> &source) {
  std::vector<float> destination(source.size(), -1.0f);
  const int32_t rowBytes = kWidth * kComponents * static_cast<int32_t>(sizeof(float));
  const spektrafilm::ImageView src{source.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::MutableImageView dst{destination.data(), 0, 0, kWidth, kHeight, rowBytes, kComponents, 4};
  const spektrafilm::RenderWindow window{0, 0, kWidth, kHeight};
  spektrafilm::RenderParams params = baseParams();
  params.colorAdaptation = true;
  params.colorAdaptationInputCompression = true;
  params.colorAdaptationCurveSmoothing = true;
  params.colorAdaptationOutputLightnessCompression = true;
  params.colorAdaptationOutputChromaCompression = true;
  params.outputColorSpace = spektrafilm::ColorSpace::Srgb;
  if (!renderer.render(src, dst, window, params, 6.0)) {
    std::cerr << "CUDA color-adaptation render failed: " << renderer.lastError() << "\n";
    return false;
  }
  const auto &diagnostics = renderer.lastDiagnostics();
  return diagnostics.backendName == "cuda" && diagnostics.passCount > 1 &&
    validOutput(destination, "color-adaptation render");
}

} // namespace

int main() {
  auto renderer = spektrafilm::createCudaRenderer();
  if (!renderer || !renderer->isAvailable()) {
    std::cerr << "SpektraFilm CUDA renderer is unavailable";
    if (renderer) {
      std::cerr << ": " << renderer->lastError();
    }
    std::cerr << "\n";
    return 1;
  }

  const std::vector<float> source = makeSource();
  if (!renderHost(*renderer, source) ||
      !renderDevice(*renderer, source) ||
      !renderSpatial(*renderer, source) ||
      !renderCameraDiffusionDir(*renderer, source) ||
      !renderDirFilmDensity(*renderer, source) ||
      !renderCameraDiffusionFilmLog(*renderer, source) ||
      !renderProcessNegative(*renderer, source) ||
      !renderRcmAces(*renderer, source) ||
      !renderColorAdaptation(*renderer, source)) {
    return 2;
  }

  const auto &diagnostics = renderer->lastDiagnostics();
  std::cout << "SpektraFilm CUDA harness passed. device=\"" << diagnostics.deviceName
            << "\" passes=" << diagnostics.passCount
            << " kernel_ms=" << diagnostics.cudaKernelMs
            << " transfer=" << diagnostics.cudaTransferMode << "\n";
  return 0;
}
