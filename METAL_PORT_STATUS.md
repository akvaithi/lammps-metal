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
- ✅ **GPU neighbor build works**, including **molecular / special-bond systems**.
  `calc_neigh_list_cell` (build), `transpose` and `kernel_special` (1-2/1-3/1-4
  masking) are ported; binning runs on the host (hybrid mode), the rest on the GPU
  (`Neighbor list builds = 0` on the CPU side confirms it). A bonded test with
  `special_bonds lj 0 0 0` matches CPU to single-precision tolerance.
- ✅ **ReaxFF QEq (charge equilibration) is GPU-accelerated.** `pair reaxff/gpu`
  runs real ReaxFF (forces on CPU) with the QEq matvec offloaded to Metal
  (`fix qeq/reaxff/gpu`); a 17k-atom Fluorocarbon test matches all-CPU ReaxFF to
  single-precision tolerance. Just `-sf gpu -pk gpu 1` (no `newton off`/`neigh no`).
- ⏳ ReaxFF **force** kernels are not ported yet (forces still run on CPU; see below).
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
the packed neighbor list, and `transpose` + `kernel_special` mask special
(1-2/1-3/1-4) bonded pairs. All three are implemented in `generate_metals.py`'s
`neighbor_gpu` program. `calc_cell_id` / `kernel_calc_cell_counts` (the full-GPU
binning path, `gpu_nbor==1`) stay stubs — Metal never uses them.

Note: `BLOCK_CELL_2D` is set to 16 in `device.metal` (was 256) so the `transpose`
threadgroup tile (`N x N+1`) fits Metal's 32 KB threadgroup-memory limit.

## Known limitations / next steps

1. **ReaxFF forces.** `pair reaxff/gpu` now computes forces on the **CPU** (via
   the parent `PairReaxFF`) and offloads only the **QEq matvec** to the GPU — a
   real, working configuration (GPU-accelerated charge equilibration + CPU ReaxFF
   forces), verified against all-CPU ReaxFF. The force kernels themselves
   (`lal_reaxff.cpp::loop()`) are still a stub. Mainline LAMMPS has no `reaxff/gpu`
   in the GPU package (only KOKKOS does), so porting the forces is a from-scratch
   effort (bond-order + ~8 energy terms), not a port — the largest remaining task.
   Note the QEq matvec is single precision (Metal has no fp64): the CG solver runs
   in double on the CPU but each A·x is float, so charges match CPU to ~1e-6.
2. **More pair styles.** Only `lj/cut/gpu` is wired into the Metal `GPU_SOURCES`.
   Porting additional `pair_*_gpu` styles is mostly adding their `.metal` kernel
   and `lal_*` glue; charged styles (`coul`) also exercise the `BaseCharge` path
   that ReaxFF's QEq needs.

## Upstreaming notes
Goal is to contribute this Metal backend to LAMMPS. Before a PR: implement GPU
neighbor kernels (remove the `neigh no` requirement), add a couple more pair
styles + a small regression test comparing Metal vs CPU within single-precision
tolerance, and clean up the `mtl_*` headers.
