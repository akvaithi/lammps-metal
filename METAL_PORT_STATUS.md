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
- ✅ **GPU neighbor build works** — no more `neigh no` needed for non-bonded
  systems. The cell-list `calc_neigh_list_cell` kernel is ported; binning runs on
  the host (hybrid mode), the neighbor build runs on the GPU (`Neighbor list
  builds = 0` on the CPU side confirms it), and results match CPU.
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

Run the LJ parity example (GPU neighbor build; `neigh no` no longer required):

```bash
cd ../examples/melt
../../build/lmp -in in.melt                       # CPU baseline
../../build/lmp -sf gpu -pk gpu 1 -in in.melt     # Metal (matches)
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

## GPU neighbor build (how it works on Metal)

Apple Metal has no device radix sort, so LAMMPS falls back to **hybrid** neighbor
mode (`gpu_nbor==2`, forced in `lal_device.cpp`): the host computes cell ids,
cell counts (prefix sum) and bins the atoms, then the GPU runs
`calc_neigh_list_cell` (the `LAL_USE_OLD_NEIGHBOR` shared-memory version) to build
the packed neighbor list. That one kernel is implemented in
`generate_metals.py`'s `neighbor_gpu` program; `calc_cell_id` /
`kernel_calc_cell_counts` (full-GPU-binning path) and `transpose` /
`kernel_special` (special-bond handling) remain stubs.

## Known limitations / next steps

1. **Molecular (special-bond) systems.** `kernel_special` is still a stub, so
   systems with bond/angle exclusions (`special_bonds`) won't have those pairs
   masked on the GPU. Non-bonded systems (LJ, and ReaxFF, which has no fixed
   topology) are unaffected. Porting `kernel_special` + `transpose` is the next
   neighbor-side task.
2. **ReaxFF forces.** Only the QEq matvec kernel is ported. The bond-order and
   force kernels in `reaxff.metal` need to be written. They now sit on a working,
   parity-verified single-precision foundation with a working GPU neighbor list.
3. **More pair styles.** Only `lj/cut/gpu` is wired into the Metal `GPU_SOURCES`.
   Porting additional `pair_*_gpu` styles is mostly adding their `.metal` kernel
   and `lal_*` glue.

## Upstreaming notes
Goal is to contribute this Metal backend to LAMMPS. Before a PR: implement GPU
neighbor kernels (remove the `neigh no` requirement), add a couple more pair
styles + a small regression test comparing Metal vs CPU within single-precision
tolerance, and clean up the `mtl_*` headers.
