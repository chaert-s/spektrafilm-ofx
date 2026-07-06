# spektrafilm OFX

spektrafilm OFX is a native OpenFX plugin project built from the `spektrafilm`
film-simulation codebase. It is intended for host applications such as DaVinci
Resolve or Nuke and provides Metal/Vulkan-accelerated film, print, scan, grain, halation,
diffusion, color-management, and LUT-export workflows on macOS and Windows.

## Build Requirements

The macOS source build expects:

1. macOS.
2. Xcode Command Line Tools or Xcode.
3. Apple's Metal toolchain available through `xcrun`.
4. CMake `3.24` or newer.
5. Python with the OFX build-time table generation dependencies installed:
   `numpy`, `scipy`, and `colour-science`.
6. libpng discoverable by CMake for the variant generator target.
7. OpenFX SDK headers (OFX_Release_1.5.1).

The Windows source build expects:

1. Windows 10 or newer.
2. Visual Studio 2022 C++ build tools, Ninja, or another CMake-supported C++17 toolchain.
3. Vulkan SDK with the Vulkan loader, headers, and `glslc` or `glslangValidator`.
4. CMake `3.24` or newer.
5. Python with the OFX build-time table generation dependencies installed:
   `numpy`, `scipy`, and `colour-science`.
6. OpenFX SDK headers (OFX_Release_1.5.1).

The Linux developer source build expects:

1. Linux x86_64.
2. GCC, Clang, Ninja, or another CMake-supported C++17 toolchain.
3. Vulkan loader and development headers plus `glslc` or `glslangValidator`
   from the Vulkan SDK or distribution packages.
4. CMake `3.24` or newer.
5. Python with the OFX build-time table generation dependencies installed:
   `numpy`, `scipy`, and `colour-science`.
6. OpenFX SDK headers (OFX_Release_1.5.1).

The OFX build prefers the repository virtual environment at `../../.venv/bin/python`
on Unix-like platforms and `../../.venv/Scripts/python.exe` on Windows when it
exists. Otherwise CMake falls back to the Python interpreter found by
`find_package(Python3)`. CMake checks for the build-time Python packages during
configure and prints the matching `pip install` command if they are missing.

## Setup From a Fresh Checkout
For ease of use, I developed this project from within Andrea's spektrafilm repository root. To follow the below instructions, pull the latest version of spektrafilm and place the contents of this repo at OFX/SpektraFilm.

From the spektrafilm root, create or sync the Python environment first. This
project uses the Python package for build-time table generation.

Using `uv`:

```sh
uv sync --extra dev
```

Or with a manually managed Python 3.13 environment:

```sh
python -m pip install -e ".[dev]"
```

For an OFX-only build environment, the full GUI/image stack is not required:

```sh
python -m pip install numpy scipy colour-science
```

Then build the OFX project:

```sh
cd OFX/SpektraFilm
./build_macos.sh
```

The script configures CMake, builds the plugin targets, and produces local
`.ofx.bundle` outputs. It does not sign with Developer ID, notarize, or write
the public macOS installer ZIP.

To create the signed public macOS release ZIP, first create a notarytool
keychain profile, then run the release packager with your Developer ID
identities:

```sh
xcrun notarytool store-credentials spektrafilm-notary

SPEKTRAFILM_DEVELOPER_ID_APP="Developer ID Application: Your Name (TEAMID)" \
SPEKTRAFILM_DEVELOPER_ID_INSTALLER="Developer ID Installer: Your Name (TEAMID)" \
SPEKTRAFILM_NOTARY_PROFILE="spektrafilm-notary" \
./tools/package_macos_release.sh
```

The output is a single notarized installer package that installs both public
macOS plugins into `/Library/OFX/Plugins`.

On Windows, run the PowerShell build script instead:

```powershell
cd OFX\SpektraFilm
.\build_windows.ps1
```

The Windows script builds local `Contents\Win64` OFX bundles for development.
It does not create the public website ZIP.

To create the public Windows release ZIP, run the release packager:

```powershell
.\tools\package_windows_release.ps1
```

The generated zip contains both public Windows `.ofx.bundle` directories, `install.bat`,
manual, install instructions, and legal notices. The batch script can be
inspected before running; it elevates, removes old public spektrafilm bundles,
and copies the new bundles into
`C:\Program Files\Common Files\OFX\Plugins`.


On Linux, use the manual CMake flow. The build creates local
`Contents/Linux-x86-64` OFX bundles for development and emits
`SpektraVulkanCopyHarness` for the same Vulkan smoke coverage. Linux release
packaging is not provided in this phase.

## Development Notes

The plugin is still an active development project.

Have fun and thank you for creating with spektrafilm!
