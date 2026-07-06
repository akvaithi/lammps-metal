# Metal Port — Status & Debugging Log

This repo is the **unified** LAMMPS-on-Apple-Metal port: it contains the core
UCL→Metal translation layer (`lib/gpu/geryon/mtl_*`, `metal-cpp-impl.cpp`), the
LJ pair style (`lj.metal`, `lal_lj.*`), and the experimental ReaxFF Metal work
(`reaxff.metal`, `lal_reaxff.*`, `pair_reaxff_gpu`, `fix_qeq_reaxff_gpu`). It
supersedes `lammps-metal-core` and `lammps-metal-reaxff`.

## Build (macOS, Apple Silicon)

```bash
mkdir build && cd build
cmake ../cmake -D PKG_GPU=yes -D PKG_REAXFF=yes -D GPU_API=metal \
      -D BUILD_MPI=no -D CMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

`cmake` now auto-runs `lib/gpu/generate_metals.py` to produce the generated
`*_cubin.h` kernel-string headers (they are gitignored). No manual step needed.

Run the LJ parity example (CPU neighbor build is required — see below):

```bash
cd ../examples/melt
../../build/lmp -in in.melt                          # CPU baseline
../../build/lmp -sf gpu -pk gpu 1 neigh no -in in.melt   # Metal
```

## Fixes applied in this consolidation

1. **cmake couldn't configure** — `lammps-metal-core`'s `GPU.cmake` referenced
   `lal_reaxff.cpp`/`_ext.cpp`, which only exist in the ReaxFF tree. In the
   unified repo those sources are present, so the reference is valid.
2. **Fresh clone couldn't build** — the generated `*_cubin.h` headers and the
   `cmake/etc/profile.d/*.in` install templates are gitignored (the `*.d` rule
   also matched the `profile.d/` directory). `generate_metals.py` is now wired
   into cmake and also emits `device_cubin.h`, `lj_cubin.h`, `reaxff_cubin.h`
   from the `.metal` files (previously only atom/neighbor were generated); the
   `.gitignore` now keeps `cmake/etc/profile.d/`.
3. **~250 lines/run of debug spam** — the per-kernel-launch `printf` tracing in
   `mtl_cuda_stubs.h` is gated behind `-DUCL_METAL_DEBUG` (off by default).
4. **Reduction correctness** (`lj.metal`) — out-of-range threads (`ii >= inum`)
   now contribute 0 to the threadgroup energy/virial reduction instead of
   returning early and leaving their shared-memory slot uninitialized.
5. **Dispatch semantics** (`mtl_cuda_stubs.h`) — `cuLaunchKernel` now uses
   `dispatchThreadgroups` (CUDA grid-of-blocks semantics: full, uniform
   threadgroups) instead of `dispatchThreads`. This keeps every threadgroup at
   the power-of-two block size and makes `threadgroups_per_grid == gridDimX ==
   the host's red_blocks`.

## LJ parity: current state (NOT yet at parity)

`lj/cut/gpu` on Metal does **not** yet match the CPU baseline. Detailed
bisection (see below) localized the remaining bug precisely.

### What is verified CORRECT (host-side reads of the shared Metal buffers)
- **Positions** `x_` — exact FCC lattice, float4 layout, host-cast (no GPU cast
  kernel in this path).
- **Neighbor list** — indices point to valid ghost images at correct periodic
  distances; `numj`, packed offsets, and `t_per_atom=1` stride all correct.
- **Coefficients** — `lj1[3]=(48,24,6.25,0)`, `lj3[3]=(4,4,0,0)`, `sp_lj=(1,0,0,0)`
  are exactly right for ε=σ=1, cut=2.5.
- **Algorithm** — replicating the kernel's exact inner loop *on the host* over
  the same buffers yields the correct per-atom energy **−6.77337** (matches CPU).

### The remaining bug: scalar kernel arguments arrive corrupted
The compiled Metal kernel produces per-atom energies ~3–8×10³ too large and
**varying per atom** (impossible on a perfect lattice → it is reading
atom-dependent wrong offsets). Echoing the kernel's received `inum` back out
gives values like `5e26` — impossible for a real `int`. So the **by-value
scalar arguments** (`inum`, `nbor_pitch`, …) passed through
`UCL_Kernel::run → cuLaunchKernel → setBytes` are being bound wrong, which
corrupts every neighbor offset (`dev_nbor[ii + nbor_pitch]`).

A follow-on symptom: writing even a constant to `engv[0]` from the kernel does
not appear at the host readback, while per-atom writes to `engv[1..inum-1]` do —
suggesting an `engv` buffer binding/offset issue in the same argument-binding
layer (`resolve_metal_buffer` / the setBytes-vs-setBuffer path in
`mtl_cuda_stubs.h::cuLaunchKernel`).

### Where to look next
`lib/gpu/geryon/mtl_cuda_stubs.h` `cuLaunchKernel` argument loop, and
`lib/gpu/geryon/mtl_kernel.h` `add_arg` / `ucl_arg_kludge.h` — verify that:
- each by-value scalar (`argSizes[i]==4`, `argIsDevicePtr[i]==false`) is copied
  with `setBytes` from a still-live address at launch time, and
- device-pointer args resolve to the correct distinct `MTL::Buffer` (no two
  args aliasing to the same buffer at offset 0).
Reproduce with `-DUCL_METAL_DEBUG` (prints each arg's bound buffer/offset) on
`examples/melt/in.melt` with `-pk gpu 1 neigh no`.

## GPU neighbor build is a stub — use `neigh no`
The GPU-side neighbor kernels (`neighbor_gpu` in `generate_metals.py`) are empty
no-ops, so the default (GPU) neighbor build never populates the list. Always run
Metal with `-pk gpu 1 neigh no` (CPU neighbor build) until the GPU neighbor
kernels are implemented.

## ReaxFF on Metal
`reaxff/gpu` + `qeq/reaxff/gpu` register and the QEq matvec kernel launches, but
the ReaxFF **force** kernels are not ported. ReaxFF cannot run a real simulation
on Metal yet, and it rides on the same argument-binding layer that LJ parity
depends on — so fixing the LJ scalar-argument bug above is the prerequisite.
