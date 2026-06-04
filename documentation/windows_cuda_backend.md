# Windows CUDA Backend

Windows builds are CUDA-only and require an NVIDIA GPU, compatible driver, and
CUDA Toolkit at build time.

DaVinci Resolve can supply CUDA device pointers through its host GPU extension,
including versions using the pre-OFX-1.5 extension path. Those frames stay on
the GPU for the complete processing graph. Hosts that supply ordinary OFX CPU
images still work through the CUDA renderer's pinned host staging path.

Use the repository-local Ninja tree:

```powershell
.\build_windows_visible_ninja.bat SpektraCudaHarness spektrafilm_flow spektrafilm spektrafilm_dev
.\Ninja\SpektraCudaHarness.exe
```

Runtime diagnostics report `backend=cuda`, the CUDA device, transfer mode,
per-pass timings when enabled, and whether source and destination were no-copy.
