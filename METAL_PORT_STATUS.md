# Metal Port — Status

This repo is the **unified** LAMMPS-on-Apple-Metal port: it contains the core
UCL→Metal translation layer (`lib/gpu/geryon/mtl_*`, `metal-cpp-impl.cpp`), the
LJ pair style (`lj.metal`, `lal_lj.*`), and the experimental ReaxFF Metal work
(`reaxff.metal`, `lal_reaxff.*`, `pair_reaxff_gpu`, `fix_qeq_reaxff_gpu`). It
supersedes `lammps-metal-core` and `lammps-metal-reaxff`.

## TL;DR

- ✅ **`lj/cut/gpu` on Metal reaches CPU parity.** The 3d LJ melt matches the CPU
  baseline to single-precision tolerance (E_pair −6.7733687 vs −6.7733681 at
  step 0; trajectories track across the full 250-step run).
- ⏳ ReaxFF: `reaxff/gpu` + `qeq/reaxff/gpu` register and the QEq matvec kernel
  runs, but the ReaxFF **force** kernels are not ported yet.
- Fresh clone builds with no manual steps (cmake generates the kernel headers).

## Build (macOS, Apple Silicon)

```bash
mkdir build && cd build
cmake ../cmake -D PKG_GPU=yes -D PKG_REAXFF=yes -D GPU_API=metal \
      -D BUILD_MPI=no -D CMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

`GPU_API=metal` forces single precision automatically (see below). cmake also
auto-runs `lib/gpu/generate_metals.py` to produce the generated `*_cubin.h`
kernel-string headers (they are gitignored).

Run the LJ parity example — **use `neigh no`** (CPU neighbor build; the GPU
neighbor kernels are still stubs):

```bash
cd ../examples/melt
../../build/lmp -in in.melt                               # CPU baseline
../../build/lmp -sf gpu -pk gpu 1 neigh no -in in.melt    # Metal (matches)
```

## The bug that blocked LJ parity: precision mismatch (fixed)

Apple Metal has **no fp64**, so the shaders always use 32-bit `numtyp`/`acctyp`.
But the default LAMMPS GPU build is `mixed` precision, which makes the **host**
C++ use `acctyp = double`. The host therefore allocated / read back the `engv`
(energy-virial) and `force` answer buffers as 8-byte doubles, while the kernels
wrote them as 4-byte floats. Every accumulator was read at the wrong stride and
reinterpreted, producing large, atom-dependent garbage energies/forces (and a
telltale `E_pair ≈ -2.7e21`).

**Fix:** the `GPU_API=metal` cmake branch now forces `GPU_PREC=single`
(`SINGLE_SINGLE`) and warns if another precision was requested, so host and
device agree on 32-bit. This is in `cmake/Modules/Packages/GPU.cmake`.

(The earlier hypothesis that scalar kernel arguments were being corrupted was a
red herring: instrumenting the bind showed `inum`, `nbor_pitch`, etc. arrive
correct — the corruption was entirely the float/double size mismatch on the
answer buffers.)

## Other fixes in this consolidation

- **cmake configures/builds from a clean clone** — generated `*_cubin.h` headers
  (now including `device`/`lj`/`reaxff`) are produced by `generate_metals.py`,
  wired into cmake; restored the `cmake/etc/profile.d/*.in` install templates
  (the `*.d` gitignore rule was swallowing the `profile.d/` directory).
- **Reduction correctness** (`lj.metal`) — out-of-range threads (`ii >= inum`)
  contribute 0 to the threadgroup energy/virial reduction instead of returning
  early (which left their shared-memory slot uninitialized).
- **Dispatch semantics** (`mtl_cuda_stubs.h`) — `cuLaunchKernel` uses
  `dispatchThreadgroups` (full, uniform threadgroups: CUDA grid-of-blocks
  semantics) so `threadgroups_per_grid == gridDimX == the host's red_blocks`.
- ~250 lines/run of per-launch `printf` tracing gated behind `-DUCL_METAL_DEBUG`.

## Known limitations / next steps

1. **GPU neighbor build is a stub.** The `neighbor_gpu` kernels in
   `generate_metals.py` are empty no-ops, so the default GPU neighbor build does
   nothing — always run Metal with `-pk gpu 1 neigh no` (CPU neighbor build).
   Implementing the GPU binning/neighbor kernels is the next core-infra task.
2. **ReaxFF forces.** Only the QEq matvec kernel is ported. The bond-order and
   force kernels in `reaxff.metal` need to be written. They now sit on a working,
   parity-verified single-precision foundation.
3. **More pair styles.** Only `lj/cut/gpu` is wired into the Metal `GPU_SOURCES`.
   Porting additional `pair_*_gpu` styles is mostly adding their `.metal` kernel
   and `lal_*` glue.

## Upstreaming notes
Goal is to contribute this Metal backend to LAMMPS. Before a PR: implement GPU
neighbor kernels (remove the `neigh no` requirement), add a couple more pair
styles + a small regression test comparing Metal vs CPU within single-precision
tolerance, and clean up the `mtl_*` headers.
