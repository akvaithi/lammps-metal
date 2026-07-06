# LAMMPS Metal Port (ReaxFF)

This repository serves as the active development branch for porting the complex **ReaxFF** force field (via `fix qeq/reaxff/gpu` and `pair reaxff/gpu`) to Apple Silicon using the Metal API. 

## Overview
ReaxFF is an empirical, bond-order dependent force field that requires many intricate computational stages, including a Charge Equilibration (QEq) solver.

This repository builds upon the `lammps-metal-core` UCL translation layer and injects the ReaxFF-specific kernels.

### Current Status
- **Core Metal Infrastructure**: The base `UCL` wrapper for Metal (`mtl_cuda_stubs.h`, `metal-cpp-impl.cpp`) is complete and stable.
- **QEq Matrix-Vector Initialization**: The core kernel `k_qeq_matvec` for the Conjugate Gradient solver in `fix qeq/reaxff/gpu` has been successfully implemented in `lib/gpu/reaxff.metal`.
- **Timer Fix**: We resolved a critical segmentation fault occurring during `ReaxFF` initialization where unallocated timers were being inadvertently polled.
- **Testing**: Using `-pk gpu 1 neigh no`, the ReaxFF GPU instantiation securely passes through initialization and successfully launches the `k_qeq_matvec` metal kernels during a simulation run without crashing.
- **Pending**: ReaxFF Force calculation (`pair reaxff/gpu`) requires further kernel translation. It currently falls back or exits with "GPU_FORCE mode only" if neighbor lists are enabled on the GPU.

## Build Instructions
To build this version of LAMMPS with the experimental ReaxFF Metal support:
```bash
mkdir build && cd build
cmake ../cmake -D PKG_GPU=yes -D PKG_REAXFF=yes -D GPU_API=metal 
make -j8
```
*(Note: standard LAMMPS CMake options apply. Ensure you have the `metal-cpp` headers available in your include path if not bundled).*

## Validation / Testing
To test the QEq solver execution:
```bash
cd examples/reaxff
../../build/lmp -sf gpu -pk gpu 1 neigh no -in in.reaxff.rdx.gpu
```
*(You must use `neigh no` to prevent LAMMPS from assigning neighbor list construction to the GPU, which triggers the unimplemented GPU force branches currently).*
