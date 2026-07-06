#pragma once

#include <memory>
#include <string>

#include "SpektraRenderer.h"

namespace spektrafilm {

using VulkanPassDiagnostics = RendererPassDiagnostics;
using VulkanRenderDiagnostics = RendererDiagnostics;

class VulkanRenderer final : public Renderer {
public:
  VulkanRenderer();
  ~VulkanRenderer() override;

  VulkanRenderer(const VulkanRenderer &) = delete;
  VulkanRenderer &operator=(const VulkanRenderer &) = delete;

  bool isAvailable() const override;
  const VulkanRenderDiagnostics &lastDiagnostics() const override;
  const std::string &lastError() const override;
  void releaseTransientResources() override;

  bool render(
    const ImageView &source,
    const MutableImageView &destination,
    const RenderWindow &window,
    const RenderParams &params,
    double time
  ) override;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  VulkanRenderDiagnostics lastRenderDiagnostics_;
  std::string lastRenderError_;
};

std::unique_ptr<Renderer> createVulkanRenderer();

} // namespace spektrafilm
