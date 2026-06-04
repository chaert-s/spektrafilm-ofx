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
  if (!renderHost(*renderer, source) || !renderDevice(*renderer, source) || !renderSpatial(*renderer, source)) {
    return 2;
  }

  const auto &diagnostics = renderer->lastDiagnostics();
  std::cout << "SpektraFilm CUDA harness passed. device=\"" << diagnostics.deviceName
            << "\" passes=" << diagnostics.passCount
            << " kernel_ms=" << diagnostics.cudaKernelMs
            << " transfer=" << diagnostics.cudaTransferMode << "\n";
  return 0;
}
