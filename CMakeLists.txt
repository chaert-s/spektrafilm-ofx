cmake_minimum_required(VERSION 3.24)

if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin" AND
    (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL ""))
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
endif()

project(SpektraFilmOFX LANGUAGES CXX)

if(WIN32)
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(SPEKTRAFILM_USE_VULKAN OFF)
if(APPLE)
  enable_language(OBJCXX)
elseif(WIN32)
  set(SPEKTRAFILM_USE_VULKAN ON)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _SPEKTRAFILM_LINUX_PROCESSOR)
  if(NOT _SPEKTRAFILM_LINUX_PROCESSOR MATCHES "^(x86_64|amd64)$")
    message(FATAL_ERROR "SpektraFilm Linux developer builds currently support x86_64/amd64 only.")
  endif()
  set(SPEKTRAFILM_USE_VULKAN ON)
else()
  message(FATAL_ERROR "SpektraFilm OFX currently supports macOS Metal, Windows CUDA/Vulkan, and Linux x86_64 Vulkan builds.")
endif()

# Define platform-specific version variables that can be overridden via command line
set(SPEKTRAFILM_MACOS_VERSION "0.2.8" CACHE STRING "Version number for macOS release")
set(SPEKTRAFILM_WINDOWS_VERSION "0.1.9" CACHE STRING "Version number for Windows release")
set(SPEKTRAFILM_LINUX_VERSION "${SPEKTRAFILM_WINDOWS_VERSION}" CACHE STRING "Version number for Linux developer builds")

# Set the active version based on the target OS
if(APPLE)
  set(SPEKTRAFILM_VERSION "${SPEKTRAFILM_MACOS_VERSION}")
elseif(WIN32)
  set(SPEKTRAFILM_VERSION "${SPEKTRAFILM_WINDOWS_VERSION}")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(SPEKTRAFILM_VERSION "${SPEKTRAFILM_LINUX_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(APPLE)
  set(CMAKE_OBJCXX_STANDARD 17)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

option(SPEKTRAFILM_NATIVE_FAST_MATH "Build native macOS renderer code with -ffast-math. Candidate optimization; validate with parity before release." OFF)
option(SPEKTRAFILM_METAL_FAST_MATH "Build Metal shaders with -ffast-math. Candidate optimization; validate with parity before release." OFF)
option(SPEKTRAFILM_METAL_PROFILE_SOURCES "Build Metal shader library with source/debug info for Xcode Metal profiling. Do not enable for release builds." OFF)
option(SPEKTRAFILM_OFX_METAL_GPU_BUFFERS "Advertise OFX Metal GPU buffer renders on macOS. CPU rendering remains the fallback." ON)
option(SPEKTRAFILM_RELEASE_LTO "Enable interprocedural optimization for Release native targets when supported." ON)
option(SPEKTRAFILM_LINUX_STATIC_STDLIB "Static-link the Linux OFX plugin C++ runtime to avoid host stdlib symbol interposition." ON)
option(SPEKTRAFILM_ENABLE_CUDA "Build the optional Windows CUDA backend. Vulkan remains the default Windows backend." OFF)
set(SPEKTRAFILM_METAL_LANGUAGE_STANDARD "macos-metal2.3" CACHE STRING "Metal language standard used to build SpektraFilm.metallib.")
set(SPEKTRAFILM_METAL_OPTIMIZATION_LEVEL "-O3" CACHE STRING "Optimization level passed to the Metal shader compiler.")
set(SPEKTRAFILM_METAL_EXTRA_FLAGS "" CACHE STRING "Extra flags passed to the Metal shader compiler.")
set(SPEKTRAFILM_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA GPU architectures for SpektraFilm CUDA targets.")
if(APPLE)
  set(SPEKTRAFILM_MACOS_LOCAL_CODESIGN_IDENTITY "-" CACHE STRING "Code signing identity for local macOS OFX bundles. Set empty to disable. Release packaging re-signs staged bundles with Developer ID.")
endif()

if(NOT DEFINED Python3_EXECUTABLE)
  set(_SPEKTRAFILM_VENV_PYTHON_CANDIDATES)
  if(WIN32)
    list(APPEND _SPEKTRAFILM_VENV_PYTHON_CANDIDATES
      "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/Scripts/python.exe"
      "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/Scripts/python"
    )
  else()
    list(APPEND _SPEKTRAFILM_VENV_PYTHON_CANDIDATES
      "${CMAKE_CURRENT_SOURCE_DIR}/../../.venv/bin/python"
    )
  endif()

  foreach(_SPEKTRAFILM_VENV_PYTHON IN LISTS _SPEKTRAFILM_VENV_PYTHON_CANDIDATES)
    if(EXISTS "${_SPEKTRAFILM_VENV_PYTHON}")
      set(Python3_EXECUTABLE "${_SPEKTRAFILM_VENV_PYTHON}" CACHE FILEPATH "Python interpreter for SpektraFilm data generation")
      break()
    endif()
  endforeach()
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import numpy, scipy, colour"
  RESULT_VARIABLE SPEKTRAFILM_PYTHON_DEPS_RESULT
  OUTPUT_QUIET
  ERROR_VARIABLE SPEKTRAFILM_PYTHON_DEPS_ERROR
)
if(NOT SPEKTRAFILM_PYTHON_DEPS_RESULT EQUAL 0)
  message(FATAL_ERROR
    "SpektraFilm OFX build-time Python dependencies are missing for ${Python3_EXECUTABLE}.\n"
    "Install them with:\n"
    "  ${Python3_EXECUTABLE} -m pip install numpy scipy colour-science\n"
    "Python error:\n${SPEKTRAFILM_PYTHON_DEPS_ERROR}"
  )
endif()

set(_SPEKTRAFILM_VULKAN_SDK "")
if(SPEKTRAFILM_USE_VULKAN AND DEFINED ENV{VULKAN_SDK})
  file(TO_CMAKE_PATH "$ENV{VULKAN_SDK}" _SPEKTRAFILM_VULKAN_SDK)
endif()

if(SPEKTRAFILM_RELEASE_LTO AND (APPLE OR SPEKTRAFILM_USE_VULKAN))
  include(CheckIPOSupported)
  check_ipo_supported(RESULT SPEKTRAFILM_RELEASE_LTO_SUPPORTED OUTPUT SPEKTRAFILM_RELEASE_LTO_ERROR LANGUAGES CXX)
  if(NOT SPEKTRAFILM_RELEASE_LTO_SUPPORTED)
    message(WARNING "Release LTO is unavailable and will be disabled: ${SPEKTRAFILM_RELEASE_LTO_ERROR}")
  endif()
endif()

if(APPLE)
  find_package(PNG REQUIRED)
elseif(SPEKTRAFILM_USE_VULKAN)
  if(_SPEKTRAFILM_VULKAN_SDK)
    list(PREPEND CMAKE_PREFIX_PATH "${_SPEKTRAFILM_VULKAN_SDK}")
    set(Vulkan_ROOT "${_SPEKTRAFILM_VULKAN_SDK}" CACHE PATH "Vulkan SDK root")

    if(NOT Vulkan_INCLUDE_DIR AND EXISTS "${_SPEKTRAFILM_VULKAN_SDK}/Include/vulkan/vulkan.h")
      set(Vulkan_INCLUDE_DIR "${_SPEKTRAFILM_VULKAN_SDK}/Include" CACHE PATH "Vulkan include directory")
    endif()

    if(NOT Vulkan_LIBRARY AND EXISTS "${_SPEKTRAFILM_VULKAN_SDK}/Lib/vulkan-1.lib")
      set(Vulkan_LIBRARY "${_SPEKTRAFILM_VULKAN_SDK}/Lib/vulkan-1.lib" CACHE FILEPATH "Vulkan loader import library")
    elseif(NOT Vulkan_LIBRARY AND EXISTS "${_SPEKTRAFILM_VULKAN_SDK}/lib/libvulkan.so")
      set(Vulkan_LIBRARY "${_SPEKTRAFILM_VULKAN_SDK}/lib/libvulkan.so" CACHE FILEPATH "Vulkan loader library")
    elseif(NOT Vulkan_LIBRARY AND EXISTS "${_SPEKTRAFILM_VULKAN_SDK}/lib64/libvulkan.so")
      set(Vulkan_LIBRARY "${_SPEKTRAFILM_VULKAN_SDK}/lib64/libvulkan.so" CACHE FILEPATH "Vulkan loader library")
    endif()
  endif()

  find_package(Vulkan REQUIRED)
endif()

set(SPEKTRAFILM_OFX_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/openfx" CACHE PATH "Vendored or external OpenFX SDK root")
set(SPEKTRAFILM_OFX_HEADERS "" CACHE PATH "Directory containing OpenFX API headers such as ofxImageEffect.h")
if(NOT SPEKTRAFILM_OFX_HEADERS)
  if(DEFINED ENV{OFX_HEADER_DIR})
    set(SPEKTRAFILM_OFX_HEADERS "$ENV{OFX_HEADER_DIR}")
  else()
    set(SPEKTRAFILM_OFX_HEADERS "${SPEKTRAFILM_OFX_ROOT}/include")
  endif()
endif()

set(SPEKTRAFILM_OFX_SUPPORT_HEADERS "${SPEKTRAFILM_OFX_ROOT}/Support" CACHE PATH "Directory containing OpenFX plugin support headers")
set(SPEKTRAFILM_OFX_HOST_SUPPORT_HEADERS "${SPEKTRAFILM_OFX_ROOT}/HostSupport" CACHE PATH "Directory containing OpenFX host support headers")
set(SPEKTRAFILM_OFX_EXAMPLE_HEADERS "${SPEKTRAFILM_OFX_ROOT}/Examples/include" CACHE PATH "Directory containing OpenFX example utility headers")

set(SPEKTRAFILM_GENERATED_PROFILE_CURVES "${CMAKE_CURRENT_BINARY_DIR}/generated/SpektraGeneratedProfileCurves.cpp")
set(SPEKTRAFILM_GENERATED_PROFILE_COUNTS "${CMAKE_CURRENT_BINARY_DIR}/generated/SpektraGeneratedProfileCounts.h")
set(SPEKTRAFILM_GENERATED_HANATOS_LUT "${CMAKE_CURRENT_BINARY_DIR}/generated/SpektraHanatos2025Spectra.f32")
set(SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION "${CMAKE_CURRENT_BINARY_DIR}/generated/SpektraOutputGamutCompression.f32")
set(SPEKTRAFILM_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Resources/data")
set(SPEKTRAFILM_PARITY_PROFILE_DIR "${SPEKTRAFILM_DATA_DIR}/profiles/archive")
set(SPEKTRAFILM_PARITY_GENERATED_PROFILE_CURVES "${CMAKE_CURRENT_BINARY_DIR}/generated_parity/SpektraGeneratedProfileCurves.cpp")
set(SPEKTRAFILM_PARITY_GENERATED_PROFILE_COUNTS "${CMAKE_CURRENT_BINARY_DIR}/generated_parity/SpektraGeneratedProfileCounts.h")
set(SPEKTRAFILM_PARITY_GENERATED_HANATOS_LUT "${CMAKE_CURRENT_BINARY_DIR}/generated_parity/SpektraHanatos2025Spectra.f32")
set(SPEKTRAFILM_PARITY_GENERATED_OUTPUT_GAMUT_COMPRESSION "${CMAKE_CURRENT_BINARY_DIR}/generated_parity/SpektraOutputGamutCompression.f32")
set(SPEKTRAFILM_HANATOS_LUT_DATA "${SPEKTRAFILM_DATA_DIR}/luts/spectral_upsampling/irradiance_xy_tc.npy")
file(GLOB SPEKTRAFILM_PROFILE_JSON CONFIGURE_DEPENDS
  "${SPEKTRAFILM_DATA_DIR}/profiles/*.json"
)
file(GLOB SPEKTRAFILM_ARCHIVE_PROFILE_JSON CONFIGURE_DEPENDS
  "${SPEKTRAFILM_DATA_DIR}/profiles/archive/*.json"
)
file(GLOB_RECURSE SPEKTRAFILM_FILTER_DATA CONFIGURE_DEPENDS
  "${SPEKTRAFILM_DATA_DIR}/filters/*"
)
file(GLOB_RECURSE SPEKTRAFILM_STANDARD_DATA CONFIGURE_DEPENDS
  "${SPEKTRAFILM_DATA_DIR}/standards/*"
)
file(GLOB_RECURSE SPEKTRAFILM_LUT_DATA CONFIGURE_DEPENDS
  "${SPEKTRAFILM_DATA_DIR}/luts/*"
)

add_custom_command(
  OUTPUT
          "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
          "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
          "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
          "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
  COMMAND "${CMAKE_COMMAND}" -E env "SPEKTRAFILM_DATA_DIR=${SPEKTRAFILM_DATA_DIR}"
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_profile_curves.py"
          "--output"
          "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
          "--counts-output"
          "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
          "--hanatos-output"
          "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
          "--output-gamut-compression-output"
          "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
  DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_profile_curves.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/ofx_stock_lists.py"
          "${SPEKTRAFILM_HANATOS_LUT_DATA}"
          ${SPEKTRAFILM_PROFILE_JSON}
          ${SPEKTRAFILM_ARCHIVE_PROFILE_JSON}
          ${SPEKTRAFILM_FILTER_DATA}
          ${SPEKTRAFILM_STANDARD_DATA}
          ${SPEKTRAFILM_LUT_DATA}
  VERBATIM
)

add_custom_command(
  OUTPUT
          "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_CURVES}"
          "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_COUNTS}"
          "${SPEKTRAFILM_PARITY_GENERATED_HANATOS_LUT}"
          "${SPEKTRAFILM_PARITY_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
  COMMAND "${CMAKE_COMMAND}" -E env "SPEKTRAFILM_DATA_DIR=${SPEKTRAFILM_DATA_DIR}"
          "SPEKTRAFILM_PROFILE_DIR=${SPEKTRAFILM_PARITY_PROFILE_DIR}"
          "SPEKTRAFILM_ARCHIVE_PROFILE_DIR=${SPEKTRAFILM_PARITY_PROFILE_DIR}"
          "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_profile_curves.py"
          "--output"
          "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_CURVES}"
          "--counts-output"
          "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_COUNTS}"
          "--hanatos-output"
          "${SPEKTRAFILM_PARITY_GENERATED_HANATOS_LUT}"
          "--output-gamut-compression-output"
          "${SPEKTRAFILM_PARITY_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
  DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/generate_profile_curves.py"
          "${CMAKE_CURRENT_SOURCE_DIR}/tools/ofx_stock_lists.py"
          "${SPEKTRAFILM_HANATOS_LUT_DATA}"
          ${SPEKTRAFILM_ARCHIVE_PROFILE_JSON}
          ${SPEKTRAFILM_FILTER_DATA}
          ${SPEKTRAFILM_STANDARD_DATA}
          ${SPEKTRAFILM_LUT_DATA}
  VERBATIM
)

set(SPEKTRAFILM_COMMON_PLUGIN_SOURCES
  src/SpektraFilmPlugin.cpp
  src/SpektraRenderCore.cpp
  "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
  "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
)

if(APPLE)
  set(SPEKTRAFILM_PLATFORM_RENDERER_SOURCES
    src/SpektraMetalRenderer.mm
  )

  set(SPEKTRAFILM_METAL_AIR "${CMAKE_CURRENT_BINARY_DIR}/SpektraFilm.air")
  set(SPEKTRAFILM_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/SpektraFilm.metallib")
  set(SPEKTRAFILM_METAL_MODULE_CACHE "${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache")
  set(SPEKTRAFILM_METAL_COMPILE_FLAGS)
  if(SPEKTRAFILM_METAL_OPTIMIZATION_LEVEL)
    list(APPEND SPEKTRAFILM_METAL_COMPILE_FLAGS "${SPEKTRAFILM_METAL_OPTIMIZATION_LEVEL}")
  endif()
  if(SPEKTRAFILM_METAL_FAST_MATH)
    list(APPEND SPEKTRAFILM_METAL_COMPILE_FLAGS -ffast-math)
  endif()
  if(SPEKTRAFILM_METAL_PROFILE_SOURCES)
    list(APPEND SPEKTRAFILM_METAL_COMPILE_FLAGS -frecord-sources)
  endif()
  if(SPEKTRAFILM_METAL_EXTRA_FLAGS)
    separate_arguments(SPEKTRAFILM_METAL_EXTRA_FLAGS_LIST NATIVE_COMMAND "${SPEKTRAFILM_METAL_EXTRA_FLAGS}")
    list(APPEND SPEKTRAFILM_METAL_COMPILE_FLAGS ${SPEKTRAFILM_METAL_EXTRA_FLAGS_LIST})
  endif()

  add_custom_command(
    OUTPUT "${SPEKTRAFILM_METAL_AIR}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPEKTRAFILM_METAL_MODULE_CACHE}"
    COMMAND xcrun -sdk macosx metal
            -std=${SPEKTRAFILM_METAL_LANGUAGE_STANDARD}
            -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
            -fmodules-cache-path="${SPEKTRAFILM_METAL_MODULE_CACHE}"
            ${SPEKTRAFILM_METAL_COMPILE_FLAGS}
            -c "${CMAKE_CURRENT_SOURCE_DIR}/shaders/SpektraFilm.metal"
            -o "${SPEKTRAFILM_METAL_AIR}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/shaders/SpektraFilm.metal"
    VERBATIM
  )

  if(SPEKTRAFILM_METAL_PROFILE_SOURCES)
    add_custom_command(
      OUTPUT "${SPEKTRAFILM_METALLIB}"
      COMMAND xcrun -sdk macosx metal
              -frecord-sources
              "${SPEKTRAFILM_METAL_AIR}"
              -o "${SPEKTRAFILM_METALLIB}"
      DEPENDS "${SPEKTRAFILM_METAL_AIR}"
      VERBATIM
    )
  else()
    add_custom_command(
      OUTPUT "${SPEKTRAFILM_METALLIB}"
      COMMAND xcrun -sdk macosx metallib
              "${SPEKTRAFILM_METAL_AIR}"
              -o "${SPEKTRAFILM_METALLIB}"
      DEPENDS "${SPEKTRAFILM_METAL_AIR}"
      VERBATIM
    )
  endif()

  add_custom_target(SpektraFilmMetalLibrary ALL DEPENDS "${SPEKTRAFILM_METALLIB}")
elseif(SPEKTRAFILM_USE_VULKAN)
  if(WIN32 AND SPEKTRAFILM_ENABLE_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(TARGET CUDA::cudart_static)
      set(SPEKTRAFILM_CUDA_RUNTIME_TARGET CUDA::cudart_static)
    else()
      set(SPEKTRAFILM_CUDA_RUNTIME_TARGET CUDA::cudart)
    endif()
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(SPEKTRAFILM_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES "${SPEKTRAFILM_CUDA_ARCHITECTURES}" CACHE STRING "CUDA GPU architectures" FORCE)
    elseif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA GPU architectures" FORCE)
    endif()
  endif()

  set(SPEKTRAFILM_PLATFORM_RENDERER_SOURCES
    src/SpektraVulkanRenderer.cpp
  )
  if(WIN32)
    list(PREPEND SPEKTRAFILM_PLATFORM_RENDERER_SOURCES
      src/SpektraNativeRenderer.cpp
    )
  endif()
  if(WIN32 AND SPEKTRAFILM_ENABLE_CUDA)
    list(PREPEND SPEKTRAFILM_PLATFORM_RENDERER_SOURCES
      src/SpektraCudaRenderer.cpp
      src/SpektraCudaKernels.cu
    )
  endif()

  set(_SPEKTRAFILM_VULKAN_TOOL_HINTS)
  if(_SPEKTRAFILM_VULKAN_SDK)
    list(APPEND _SPEKTRAFILM_VULKAN_TOOL_HINTS "${_SPEKTRAFILM_VULKAN_SDK}/Bin")
    list(APPEND _SPEKTRAFILM_VULKAN_TOOL_HINTS "${_SPEKTRAFILM_VULKAN_SDK}/bin")
  endif()

  find_program(SPEKTRAFILM_GLSLC_EXECUTABLE
    NAMES glslc
    HINTS ${_SPEKTRAFILM_VULKAN_TOOL_HINTS}
  )
  find_program(SPEKTRAFILM_GLSLANG_VALIDATOR_EXECUTABLE
    NAMES glslangValidator
    HINTS ${_SPEKTRAFILM_VULKAN_TOOL_HINTS}
  )

  set(SPEKTRAFILM_VULKAN_SHADER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraCopy.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraFormatConvert.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraFilmExposure.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraCurveDevelop.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraPrintScan.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraHalation.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraDiffusion.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraDir.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraScannerPost.comp"
    "${CMAKE_CURRENT_SOURCE_DIR}/shaders/vulkan/SpektraGrain.comp"
  )
  set(SPEKTRAFILM_VULKAN_SPV_FILES)
  set(SPEKTRAFILM_VULKAN_COPY_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraCopy.comp.spv")
  set(SPEKTRAFILM_VULKAN_FORMAT_CONVERT_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraFormatConvert.comp.spv")
  set(SPEKTRAFILM_VULKAN_FILM_EXPOSURE_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraFilmExposure.comp.spv")
  set(SPEKTRAFILM_VULKAN_CURVE_DEVELOP_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraCurveDevelop.comp.spv")
  set(SPEKTRAFILM_VULKAN_PRINT_SCAN_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraPrintScan.comp.spv")
  set(SPEKTRAFILM_VULKAN_HALATION_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraHalation.comp.spv")
  set(SPEKTRAFILM_VULKAN_DIFFUSION_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraDiffusion.comp.spv")
  set(SPEKTRAFILM_VULKAN_DIR_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraDir.comp.spv")
  set(SPEKTRAFILM_VULKAN_SCANNER_POST_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraScannerPost.comp.spv")
  set(SPEKTRAFILM_VULKAN_GRAIN_SPV "${CMAKE_CURRENT_BINARY_DIR}/vulkan/SpektraGrain.comp.spv")
  if(NOT SPEKTRAFILM_GLSLC_EXECUTABLE AND NOT SPEKTRAFILM_GLSLANG_VALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "A Vulkan GLSL compiler is required. Install the Vulkan SDK so glslc or glslangValidator is on PATH.")
  endif()
  foreach(vulkan_shader IN LISTS SPEKTRAFILM_VULKAN_SHADER_SOURCES)
    get_filename_component(vulkan_shader_name "${vulkan_shader}" NAME)
    set(vulkan_spv "${CMAKE_CURRENT_BINARY_DIR}/vulkan/${vulkan_shader_name}.spv")
    list(APPEND SPEKTRAFILM_VULKAN_SPV_FILES "${vulkan_spv}")
    if(SPEKTRAFILM_GLSLC_EXECUTABLE)
      add_custom_command(
        OUTPUT "${vulkan_spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/vulkan"
        COMMAND "${SPEKTRAFILM_GLSLC_EXECUTABLE}"
                -fshader-stage=compute
                "${vulkan_shader}"
                -o "${vulkan_spv}"
        DEPENDS "${vulkan_shader}"
        VERBATIM
      )
    else()
      add_custom_command(
        OUTPUT "${vulkan_spv}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/vulkan"
        COMMAND "${SPEKTRAFILM_GLSLANG_VALIDATOR_EXECUTABLE}"
                -V
                "${vulkan_shader}"
                -o "${vulkan_spv}"
        DEPENDS "${vulkan_shader}"
        VERBATIM
      )
    endif()
  endforeach()
  add_custom_target(SpektraFilmVulkanShaders ALL DEPENDS ${SPEKTRAFILM_VULKAN_SPV_FILES})
endif()

set(SPEKTRAFILM_PLUGIN_SOURCES
  ${SPEKTRAFILM_COMMON_PLUGIN_SOURCES}
  ${SPEKTRAFILM_PLATFORM_RENDERER_SOURCES}
)

set(SPEKTRAFILM_WEBSITE_DOWNLOAD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../website/public/downloads")
set(SPEKTRAFILM_DEFAULT_USER_MANUAL_PDF "${CMAKE_CURRENT_SOURCE_DIR}/../../docs/user_manual/main.pdf")
set(SPEKTRAFILM_USER_MANUAL_PDF "${SPEKTRAFILM_DEFAULT_USER_MANUAL_PDF}" CACHE FILEPATH "PDF manual copied into OFX bundles")
if(NOT EXISTS "${SPEKTRAFILM_USER_MANUAL_PDF}")
  set(SPEKTRAFILM_USER_MANUAL_PDF "${CMAKE_CURRENT_SOURCE_DIR}/documentation/spektrafilm_reference_guide.pdf")
endif()
set(SPEKTRAFILM_OFX_INSTALL_INSTRUCTIONS "${CMAKE_CURRENT_SOURCE_DIR}/install_instructions.txt")
set(SPEKTRAFILM_OFX_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Legal/SPEKTRAFILM_OFX_LICENSE.txt")
set(SPEKTRAFILM_OFX_LUT_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/Legal/SPEKTRAFILM_OFX_LUT_LICENSE.txt")
set(SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES "${CMAKE_CURRENT_SOURCE_DIR}/Legal/THIRD_PARTY_NOTICES.txt")

if(WIN32)
  if(DEFINED ENV{CommonProgramFiles})
    set(_SPEKTRAFILM_WINDOWS_INSTALL_ROOT "$ENV{CommonProgramFiles}/OFX/Plugins")
  else()
    set(_SPEKTRAFILM_WINDOWS_INSTALL_ROOT "C:/Program Files/Common Files/OFX/Plugins")
  endif()
  set(SPEKTRAFILM_WINDOWS_OFX_INSTALL_ROOT "${_SPEKTRAFILM_WINDOWS_INSTALL_ROOT}" CACHE PATH "Windows OFX plugin install directory")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(SPEKTRAFILM_LINUX_OFX_INSTALL_ROOT "/usr/OFX/Plugins" CACHE PATH "Linux OFX plugin install directory")
endif()

function(apply_spektra_macos_native_options target)
  if(APPLE)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>"
    )
    if(SPEKTRAFILM_RELEASE_LTO AND SPEKTRAFILM_RELEASE_LTO_SUPPORTED)
      set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
      target_compile_options(${target} PRIVATE
        "$<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:OBJCXX>>:-flto=thin>"
      )
    endif()
    if(SPEKTRAFILM_NATIVE_FAST_MATH)
      target_compile_options(${target} PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:-ffast-math>"
        "$<$<COMPILE_LANGUAGE:OBJCXX>:-ffast-math>"
      )
      target_compile_definitions(${target} PRIVATE SPEKTRAFILM_NATIVE_FAST_MATH=1)
    endif()
  endif()
endfunction()

function(apply_spektra_release_lto target)
  if(SPEKTRAFILM_RELEASE_LTO AND SPEKTRAFILM_RELEASE_LTO_SUPPORTED)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endfunction()

function(configure_spektra_target target artifact_name display_name plugin_id flavor)
  set(SPEKTRAFILM_BUNDLE_EXECUTABLE "${artifact_name}.ofx")
  set(SPEKTRAFILM_BUNDLE_IDENTIFIER "${plugin_id}")
  set(SPEKTRAFILM_BUNDLE_NAME "${display_name}")
  set(SPEKTRAFILM_PLUGIN_ICON_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/${artifact_name}.png")
  set(SPEKTRAFILM_RESOLVE_ICON_NAME "${plugin_id}.png")
  set(SPEKTRAFILM_MANIFEST_ID "${plugin_id}")
  set(SPEKTRAFILM_MANIFEST_NAME "${display_name}")
  set(SPEKTRAFILM_MANIFEST_VERSION "${SPEKTRAFILM_VERSION}")
  if(APPLE)
    set(SPEKTRAFILM_MANIFEST_RENDERER "Metal")
  elseif(WIN32 AND SPEKTRAFILM_ENABLE_CUDA)
    set(SPEKTRAFILM_MANIFEST_RENDERER "CUDA/Vulkan")
  else()
    set(SPEKTRAFILM_MANIFEST_RENDERER "Vulkan")
  endif()

  set(plugin_manifest "${CMAKE_CURRENT_BINARY_DIR}/${target}-plugin_manifest.json")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Resources/plugin_manifest.json.in" "${plugin_manifest}" @ONLY)

  add_library(${target} MODULE ${SPEKTRAFILM_PLUGIN_SOURCES})

  target_include_directories(${target} PRIVATE
    "${SPEKTRAFILM_OFX_HEADERS}"
    "${SPEKTRAFILM_OFX_SUPPORT_HEADERS}"
    "${SPEKTRAFILM_OFX_HOST_SUPPORT_HEADERS}"
    "${SPEKTRAFILM_OFX_EXAMPLE_HEADERS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  target_compile_definitions(${target} PRIVATE
    SPEKTRAFILM_VERSION_STRING="${SPEKTRAFILM_VERSION}"
    SPEKTRAFILM_PLUGIN_IDENTIFIER="${plugin_id}"
    SPEKTRAFILM_PLUGIN_LABEL="${display_name}"
    SPEKTRAFILM_PLUGIN_FLAVOR=${flavor}
  )

  if(SPEKTRAFILM_USE_VULKAN)
    apply_spektra_release_lto(${target})
    if(WIN32)
      target_compile_definitions(${target} PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
      )
      if(SPEKTRAFILM_ENABLE_CUDA)
        target_compile_definitions(${target} PRIVATE SPEKTRAFILM_ENABLE_CUDA=1)
      endif()
    endif()
    target_link_libraries(${target} PRIVATE
      Vulkan::Vulkan
    )
    if(WIN32)
      target_link_libraries(${target} PRIVATE shell32 user32)
      if(SPEKTRAFILM_ENABLE_CUDA)
        target_link_libraries(${target} PRIVATE ${SPEKTRAFILM_CUDA_RUNTIME_TARGET})
        set_target_properties(${target} PROPERTIES
          CUDA_SEPARABLE_COMPILATION OFF
          CUDA_RESOLVE_DEVICE_SYMBOLS ON
        )
      endif()
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      target_compile_options(${target} PRIVATE
        -fvisibility=hidden
        -fvisibility-inlines-hidden
      )
      target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
      target_link_options(${target} PRIVATE
        -Wl,-Bsymbolic-functions
        -Wl,--exclude-libs,ALL
      )
      if(SPEKTRAFILM_LINUX_STATIC_STDLIB)
        target_link_options(${target} PRIVATE
          -static-libstdc++
          -static-libgcc
        )
      endif()
    else()
      target_link_libraries(${target} PRIVATE ${CMAKE_DL_LIBS})
    endif()
    set_target_properties(${target} PROPERTIES
      PREFIX ""
      SUFFIX ".ofx"
    )
  elseif(APPLE)
    if(SPEKTRAFILM_OFX_METAL_GPU_BUFFERS)
      target_compile_definitions(${target} PRIVATE SPEKTRAFILM_OFX_METAL_GPU_BUFFERS=1)
    endif()
    apply_spektra_macos_native_options(${target})
    target_link_libraries(${target} PRIVATE
      "-framework ApplicationServices"
      "-framework Foundation"
      "-framework Metal"
      "-framework MetalKit"
      "-framework MetalPerformanceShaders"
      "-framework QuartzCore"
    )

    set(info_plist "${CMAKE_CURRENT_BINARY_DIR}/${target}-Info.plist")
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/Resources/Info.plist.in" "${info_plist}" @ONLY)
    set_target_properties(${target} PROPERTIES
      BUNDLE TRUE
      # OUTPUT_NAME includes the OFX executable suffix; "bundle" keeps the
      # outer artifact as <artifact>.ofx.bundle.
      BUNDLE_EXTENSION "bundle"
      MACOSX_BUNDLE_INFO_PLIST "${info_plist}"
      OUTPUT_NAME "${artifact_name}.ofx"
      PREFIX ""
    )
  endif()

  set(${target}_PLUGIN_MANIFEST "${plugin_manifest}" PARENT_SCOPE)
  set(${target}_PLUGIN_ICON_SOURCE "${SPEKTRAFILM_PLUGIN_ICON_SOURCE}" PARENT_SCOPE)
  set(${target}_RESOLVE_ICON_NAME "${SPEKTRAFILM_RESOLVE_ICON_NAME}" PARENT_SCOPE)
endfunction()

function(add_macos_bundle_resources target artifact_name plugin_manifest plugin_icon_source resolve_icon_name package_for_website)
  set(bundle_resource_deps
    "${SPEKTRAFILM_METALLIB}"
    "${plugin_manifest}"
    "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
    "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
    "${SPEKTRAFILM_USER_MANUAL_PDF}"
    "${SPEKTRAFILM_OFX_LICENSE}"
    "${SPEKTRAFILM_OFX_LUT_LICENSE}"
    "${SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.png"
    "${plugin_icon_source}"
  )

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/Legal"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons"
    COMMAND "${CMAKE_COMMAND}" -E rm -f
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/MacOS/${artifact_name}"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_METALLIB}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/SpektraFilm.metallib"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${plugin_manifest}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/plugin_manifest.json"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/SpektraOutputGamutCompression.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_USER_MANUAL_PDF}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/manual.pdf"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_LICENSE}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/Legal/SPEKTRAFILM_OFX_LICENSE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_LUT_LICENSE}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/Legal/SPEKTRAFILM_OFX_LUT_LICENSE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/Legal/THIRD_PARTY_NOTICES.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.svg"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm_flow.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.svg"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.svg"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm_dev.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.png"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm_flow.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.png"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.png"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/icons/spektrafilm_dev.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${plugin_icon_source}"
            "$<TARGET_BUNDLE_CONTENT_DIR:${target}>/Resources/${resolve_icon_name}"
    VERBATIM
  )

  add_custom_target(${target}BundleResources ALL DEPENDS ${target} ${bundle_resource_deps})
  add_dependencies(${target} SpektraFilmMetalLibrary)

  if(SPEKTRAFILM_MACOS_LOCAL_CODESIGN_IDENTITY)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND /usr/bin/codesign
              --force
              --sign "${SPEKTRAFILM_MACOS_LOCAL_CODESIGN_IDENTITY}"
              "$<TARGET_BUNDLE_DIR:${target}>"
      VERBATIM
    )
  endif()

  if(package_for_website)
    set(download_zip "${SPEKTRAFILM_WEBSITE_DOWNLOAD_DIR}/${artifact_name}-OFX-macOS.zip")
    set(download_stage "${CMAKE_CURRENT_BINARY_DIR}/${target}-download")
    add_custom_target(${target}DownloadZip
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${SPEKTRAFILM_WEBSITE_DOWNLOAD_DIR}"
      COMMAND "${CMAKE_COMMAND}" -E rm -f "${download_zip}"
      COMMAND "${CMAKE_COMMAND}" -E rm -rf "${download_stage}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${download_stage}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${download_stage}/Legal"
      COMMAND "${CMAKE_COMMAND}" -E copy_directory
              "$<TARGET_BUNDLE_DIR:${target}>"
              "${download_stage}/${artifact_name}.ofx.bundle"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_OFX_INSTALL_INSTRUCTIONS}"
              "${download_stage}/install_instructions.txt"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_USER_MANUAL_PDF}"
              "${download_stage}/manual.pdf"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_OFX_LICENSE}"
              "${download_stage}/Legal/SPEKTRAFILM_OFX_LICENSE.txt"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_OFX_LUT_LICENSE}"
              "${download_stage}/Legal/SPEKTRAFILM_OFX_LUT_LICENSE.txt"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES}"
              "${download_stage}/Legal/THIRD_PARTY_NOTICES.txt"
      COMMAND "${CMAKE_COMMAND}" -E env COPYFILE_DISABLE=1 /usr/bin/ditto
              -c
              -k
              --norsrc
              "${download_stage}"
              "${download_zip}"
      DEPENDS ${target}BundleResources "${SPEKTRAFILM_OFX_INSTALL_INSTRUCTIONS}" "${SPEKTRAFILM_USER_MANUAL_PDF}"
      VERBATIM
    )
  endif()

  install(TARGETS ${target}
    LIBRARY DESTINATION "/Library/OFX/Plugins"
    BUNDLE DESTINATION "/Library/OFX/Plugins"
  )
endfunction()

function(add_vulkan_bundle_resources target artifact_name plugin_manifest plugin_icon_source resolve_icon_name binary_subdir install_root)
  set(bundle_root "${CMAKE_CURRENT_BINARY_DIR}/${artifact_name}.ofx.bundle")
  set(contents_dir "${bundle_root}/Contents")
  set(binary_dir "${contents_dir}/${binary_subdir}")
  set(resources_dir "${contents_dir}/Resources")
  set(bundle_resource_deps
    ${SPEKTRAFILM_VULKAN_SPV_FILES}
    "${plugin_manifest}"
    "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
    "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
    "${SPEKTRAFILM_USER_MANUAL_PDF}"
    "${SPEKTRAFILM_OFX_LICENSE}"
    "${SPEKTRAFILM_OFX_LUT_LICENSE}"
    "${SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES}"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.svg"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.png"
    "${plugin_icon_source}"
  )

  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${binary_dir}"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${resources_dir}/Legal"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${resources_dir}/icons"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${resources_dir}/shaders"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "$<TARGET_FILE:${target}>"
            "${binary_dir}/${artifact_name}.ofx"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_COPY_SPV}"
            "${resources_dir}/shaders/SpektraCopy.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_FORMAT_CONVERT_SPV}"
            "${resources_dir}/shaders/SpektraFormatConvert.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_FILM_EXPOSURE_SPV}"
            "${resources_dir}/shaders/SpektraFilmExposure.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_CURVE_DEVELOP_SPV}"
            "${resources_dir}/shaders/SpektraCurveDevelop.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_PRINT_SCAN_SPV}"
            "${resources_dir}/shaders/SpektraPrintScan.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_HALATION_SPV}"
            "${resources_dir}/shaders/SpektraHalation.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_DIFFUSION_SPV}"
            "${resources_dir}/shaders/SpektraDiffusion.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_DIR_SPV}"
            "${resources_dir}/shaders/SpektraDir.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_SCANNER_POST_SPV}"
            "${resources_dir}/shaders/SpektraScannerPost.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_GRAIN_SPV}"
            "${resources_dir}/shaders/SpektraGrain.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${plugin_manifest}"
            "${resources_dir}/plugin_manifest.json"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "${resources_dir}/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "${resources_dir}/SpektraOutputGamutCompression.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_USER_MANUAL_PDF}"
            "${resources_dir}/manual.pdf"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_LICENSE}"
            "${resources_dir}/Legal/SPEKTRAFILM_OFX_LICENSE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_LUT_LICENSE}"
            "${resources_dir}/Legal/SPEKTRAFILM_OFX_LUT_LICENSE.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_OFX_THIRD_PARTY_NOTICES}"
            "${resources_dir}/Legal/THIRD_PARTY_NOTICES.txt"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.svg"
            "${resources_dir}/icons/spektrafilm_flow.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.svg"
            "${resources_dir}/icons/spektrafilm.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.svg"
            "${resources_dir}/icons/spektrafilm_dev.svg"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_flow.png"
            "${resources_dir}/icons/spektrafilm_flow.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm.png"
            "${resources_dir}/icons/spektrafilm.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/Resources/icons/spektrafilm_dev.png"
            "${resources_dir}/icons/spektrafilm_dev.png"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${plugin_icon_source}"
            "${resources_dir}/${resolve_icon_name}"
    VERBATIM
  )

  add_dependencies(${target} SpektraFilmVulkanShaders)
  add_custom_target(${target}BundleResources ALL DEPENDS ${target} SpektraFilmVulkanShaders ${bundle_resource_deps})
  install(DIRECTORY "${bundle_root}" DESTINATION "${install_root}")
endfunction()

function(add_windows_bundle_resources target artifact_name plugin_manifest plugin_icon_source resolve_icon_name package_for_website)
  add_vulkan_bundle_resources(
    ${target}
    "${artifact_name}"
    "${plugin_manifest}"
    "${plugin_icon_source}"
    "${resolve_icon_name}"
    "Win64"
    "${SPEKTRAFILM_WINDOWS_OFX_INSTALL_ROOT}"
  )
endfunction()

function(add_linux_bundle_resources target artifact_name plugin_manifest plugin_icon_source resolve_icon_name package_for_website)
  add_vulkan_bundle_resources(
    ${target}
    "${artifact_name}"
    "${plugin_manifest}"
    "${plugin_icon_source}"
    "${resolve_icon_name}"
    "Linux-x86-64"
    "${SPEKTRAFILM_LINUX_OFX_INSTALL_ROOT}"
  )
endfunction()

function(add_spektra_ofx_plugin target artifact_name display_name plugin_id flavor package_for_website)
  configure_spektra_target(${target} "${artifact_name}" "${display_name}" "${plugin_id}" "${flavor}")
  if(APPLE)
    add_macos_bundle_resources(
      ${target}
      "${artifact_name}"
      "${${target}_PLUGIN_MANIFEST}"
      "${${target}_PLUGIN_ICON_SOURCE}"
      "${${target}_RESOLVE_ICON_NAME}"
      "${package_for_website}"
    )
  elseif(WIN32)
    add_windows_bundle_resources(
      ${target}
      "${artifact_name}"
      "${${target}_PLUGIN_MANIFEST}"
      "${${target}_PLUGIN_ICON_SOURCE}"
      "${${target}_RESOLVE_ICON_NAME}"
      "${package_for_website}"
    )
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_linux_bundle_resources(
      ${target}
      "${artifact_name}"
      "${${target}_PLUGIN_MANIFEST}"
      "${${target}_PLUGIN_ICON_SOURCE}"
      "${${target}_RESOLVE_ICON_NAME}"
      "${package_for_website}"
    )
  endif()
endfunction()

add_spektra_ofx_plugin(spektrafilm_flow spektrafilm_flow "spektrafilm flow" org.spektrafilm.flow 0 TRUE)
add_spektra_ofx_plugin(spektrafilm spektrafilm spektrafilm org.spektrafilm 1 TRUE)
add_spektra_ofx_plugin(spektrafilm_dev spektrafilm_dev "spektrafilm dev" org.spektrafilm.dev 2 FALSE)

if(APPLE)
  add_library(SpektraFilmNativeCore STATIC
    src/SpektraAppBridge.mm
    src/SpektraMetalRenderer.mm
    "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
  )

  target_include_directories(SpektraFilmNativeCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  apply_spektra_macos_native_options(SpektraFilmNativeCore)

  add_dependencies(SpektraFilmNativeCore SpektraFilmMetalLibrary)

  add_executable(SpektraFilmPerfHarness
    tools/SpektraMetalPerfHarness.mm
    src/SpektraMetalRenderer.mm
    "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
  )

  add_executable(SpektraFilmEvaluationHarness
    tools/SpektraMetalEvaluationHarness.mm
    src/SpektraMetalRenderer.mm
    "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
  )

  add_executable(SpektraFilmParityHarness
    tools/SpektraMetalParityHarness.mm
    src/SpektraMetalRenderer.mm
    "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_PARITY_GENERATED_PROFILE_COUNTS}"
  )

  add_executable(SpektraVariantGenerator
    tools/SpektraVariantGenerator.mm
    src/SpektraMetalRenderer.mm
    "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
  )

  target_include_directories(SpektraFilmPerfHarness PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  target_include_directories(SpektraFilmEvaluationHarness PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  target_include_directories(SpektraFilmParityHarness PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated_parity"
  )

  target_include_directories(SpektraVariantGenerator PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  apply_spektra_macos_native_options(SpektraFilmPerfHarness)
  apply_spektra_macos_native_options(SpektraFilmEvaluationHarness)
  apply_spektra_macos_native_options(SpektraFilmParityHarness)
  apply_spektra_macos_native_options(SpektraVariantGenerator)

  target_link_libraries(SpektraFilmPerfHarness PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
    "-framework QuartzCore"
  )

  target_link_libraries(SpektraFilmEvaluationHarness PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
    "-framework QuartzCore"
  )

  target_link_libraries(SpektraFilmParityHarness PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
    "-framework QuartzCore"
  )

  target_link_libraries(SpektraVariantGenerator PRIVATE
    PNG::PNG
    "-framework CoreGraphics"
    "-framework Foundation"
    "-framework ImageIO"
    "-framework Metal"
    "-framework MetalKit"
    "-framework MetalPerformanceShaders"
    "-framework QuartzCore"
  )

  add_dependencies(SpektraFilmPerfHarness SpektraFilmMetalLibrary)
  add_dependencies(SpektraFilmEvaluationHarness SpektraFilmMetalLibrary)
  add_dependencies(SpektraFilmParityHarness SpektraFilmMetalLibrary)
  add_dependencies(SpektraVariantGenerator SpektraFilmMetalLibrary)

  add_custom_command(TARGET SpektraFilmPerfHarness POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_METALLIB}"
            "$<TARGET_FILE_DIR:SpektraFilmPerfHarness>/SpektraFilm.metallib"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "$<TARGET_FILE_DIR:SpektraFilmPerfHarness>/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_FILE_DIR:SpektraFilmPerfHarness>/SpektraOutputGamutCompression.f32"
    VERBATIM
  )

  add_custom_command(TARGET SpektraFilmEvaluationHarness POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_METALLIB}"
            "$<TARGET_FILE_DIR:SpektraFilmEvaluationHarness>/SpektraFilm.metallib"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "$<TARGET_FILE_DIR:SpektraFilmEvaluationHarness>/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_FILE_DIR:SpektraFilmEvaluationHarness>/SpektraOutputGamutCompression.f32"
    VERBATIM
  )

  add_custom_command(TARGET SpektraFilmParityHarness POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_METALLIB}"
            "$<TARGET_FILE_DIR:SpektraFilmParityHarness>/SpektraFilm.metallib"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_PARITY_GENERATED_HANATOS_LUT}"
            "$<TARGET_FILE_DIR:SpektraFilmParityHarness>/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_PARITY_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_FILE_DIR:SpektraFilmParityHarness>/SpektraOutputGamutCompression.f32"
    VERBATIM
  )

  add_custom_command(TARGET SpektraVariantGenerator POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_METALLIB}"
            "$<TARGET_FILE_DIR:SpektraVariantGenerator>/SpektraFilm.metallib"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "$<TARGET_FILE_DIR:SpektraVariantGenerator>/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_FILE_DIR:SpektraVariantGenerator>/SpektraOutputGamutCompression.f32"
    VERBATIM
  )
elseif(SPEKTRAFILM_USE_VULKAN)
  if(WIN32 AND SPEKTRAFILM_ENABLE_CUDA)
    add_executable(SpektraCudaHarness
      tools/SpektraCudaHarness.cpp
      src/SpektraRenderCore.cpp
      src/SpektraCudaRenderer.cpp
      src/SpektraCudaKernels.cu
      "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
      "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
    )

    target_include_directories(SpektraCudaHarness PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_BINARY_DIR}/generated"
    )
    target_compile_definitions(SpektraCudaHarness PRIVATE
      NOMINMAX
      WIN32_LEAN_AND_MEAN
      _CRT_SECURE_NO_WARNINGS
      SPEKTRAFILM_ENABLE_CUDA=1
    )
    target_link_libraries(SpektraCudaHarness PRIVATE ${SPEKTRAFILM_CUDA_RUNTIME_TARGET})
    set_target_properties(SpektraCudaHarness PROPERTIES
      CUDA_SEPARABLE_COMPILATION OFF
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    add_custom_command(TARGET SpektraCudaHarness POST_BUILD
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
              "$<TARGET_FILE_DIR:SpektraCudaHarness>/SpektraHanatos2025Spectra.f32"
      COMMAND "${CMAKE_COMMAND}" -E copy_if_different
              "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
              "$<TARGET_FILE_DIR:SpektraCudaHarness>/SpektraOutputGamutCompression.f32"
      VERBATIM
    )
  endif()

  add_executable(SpektraVulkanCopyHarness
    tools/SpektraVulkanCopyHarness.cpp
    src/SpektraRenderCore.cpp
    src/SpektraVulkanRenderer.cpp
    "${SPEKTRAFILM_GENERATED_PROFILE_CURVES}"
    "${SPEKTRAFILM_GENERATED_PROFILE_COUNTS}"
  )

  target_include_directories(SpektraVulkanCopyHarness PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
  )

  if(WIN32)
    target_compile_definitions(SpektraVulkanCopyHarness PRIVATE
      NOMINMAX
      WIN32_LEAN_AND_MEAN
      _CRT_SECURE_NO_WARNINGS
    )
  endif()

  target_link_libraries(SpektraVulkanCopyHarness PRIVATE Vulkan::Vulkan)
  apply_spektra_release_lto(SpektraVulkanCopyHarness)
  if(NOT WIN32)
    target_link_libraries(SpektraVulkanCopyHarness PRIVATE ${CMAKE_DL_LIBS})
  endif()
  add_dependencies(SpektraVulkanCopyHarness SpektraFilmVulkanShaders)

  add_custom_command(TARGET SpektraVulkanCopyHarness POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_COPY_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraCopy.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_FORMAT_CONVERT_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraFormatConvert.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_FILM_EXPOSURE_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraFilmExposure.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_CURVE_DEVELOP_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraCurveDevelop.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_PRINT_SCAN_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraPrintScan.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_HALATION_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraHalation.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_DIFFUSION_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraDiffusion.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_DIR_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraDir.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_SCANNER_POST_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraScannerPost.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_VULKAN_GRAIN_SPV}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/shaders/SpektraGrain.comp.spv"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_HANATOS_LUT}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/SpektraHanatos2025Spectra.f32"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${SPEKTRAFILM_GENERATED_OUTPUT_GAMUT_COMPRESSION}"
            "$<TARGET_FILE_DIR:SpektraVulkanCopyHarness>/SpektraOutputGamutCompression.f32"
    VERBATIM
  )
endif()
