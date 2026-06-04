#include "SpektraRenderer.h"
#include "SpektraCudaRenderer.h"

namespace spektrafilm {

std::unique_ptr<Renderer> createNativeRenderer() {
  return createCudaRenderer();
}

} // namespace spektrafilm
