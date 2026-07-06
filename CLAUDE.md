# CLAUDE.md — LAMMPS-on-Apple-Metal port

Guidance for working in this repo. Read this + `METAL_PORT_STATUS.md` before touching code.

## What this is
A port of LAMMPS's **GPU package** (the geryon/UCL layer, normally CUDA/OpenCL)
to **Apple Metal** via metal-cpp, so ReaxFF and other pair styles can run on
Apple-Silicon GPUs. It is the acceleration companion to the
`coke-graphitization-sim` project (Fe-catalyzed graphitization ReaxFF), but is a
general LAMMPS backend, not coke-specific. This is the **unified** repo
(github.com/akvaithi/lammps-metal); it supersedes the now-archived
`lammps-metal-core` and `lammps-metal-reaxff`.

## STATUS — PAUSED (2026-07-06)
Development is intentionally paused. Rationale: the remaining work (a from-scratch
port of the ReaxFF **force** terms — see below) is a 3–6 month build, and it is
**not** on the critical path for the coke sim, whose real blocker is *simulated
timescale* (ps vs. the hour-scale real process), which no GPU speedup fixes. The
high-value, already-working slice (LJ + GPU neighbor build + QEq offload) is a
clean upstreamable milestone; finishing ReaxFF forces is the opposite (highest
effort, narrowest payoff). Resume only if a compute-bound need reappears.

## What works (all validated to single-precision tolerance vs CPU)
- **`lj/cut/gpu`** on Metal — full parity on the 3d LJ melt.
- **GPU neighbor build**, including molecular / special-bond systems (no `neigh no`
  needed). Hybrid mode: host bins, GPU builds the list (`calc_neigh_list_cell`,
  `transpose`, `kernel_special`).
- **ReaxFF QEq** offloaded (`fix qeq/reaxff/gpu`): `pair reaxff/gpu` runs real
  ReaxFF with **forces on the CPU** and only the QEq matvec on Metal. Verified on
  the 17k-atom FC example.
- **ReaxFF nonbonded (vdW + Coulomb) ENERGY** on GPU, validated (17k-atom FC,
  1.77M pairs: e_vdW rel 1.2e-7, e_ele 5.8e-6). Energy only — forces still CPU.

## In progress (parked)
- **ReaxFF nonbonded FORCES**: kernel `k_reaxff_nb_force` + `ReaxFFGPU::nonbonded_force`
  are written but **not built/wired/validated** — parked on branch
  `wip-nonbonded-forces` (pushed to unified), off `main`, per the discipline below.
- Everything else in `REAXFF_FORCES_PLAN.md` (BO, bonds, angles, torsion, hbond)
  is unstarted. Mainline LAMMPS has **no** `reaxff/gpu` in the GPU package (only
  KOKKOS), so those are a from-scratch build, not a port.

## Non-negotiable discipline
- **Never commit unvalidated physics to `main`.** Every ReaxFF term must be checked
  against CPU (`data->my_en.*`) and, for forces, finite-difference of the validated
  energy, before it lands. Unvalidated WIP goes on a branch.
- **Metal is fp32-only** (no fp64). This is why `GPU_API=metal` forces
  `GPU_PREC=single` in `cmake/Modules/Packages/GPU.cmake` — do not revert it (a
  host-double / device-float mismatch was the LJ-parity bug). Precision-sensitive
  ReaxFF terms (BO corrections, angles/torsion) may not hold tolerance in fp32;
  validate with loose (~1e-4) tolerance and watch energy conservation.
- **Runtime Metal shader compilation gets no `-D` defines** — hardcode constants in
  the `.metal` sources, not via preprocessor flags. `cmake` auto-runs
  `lib/gpu/generate_metals.py` to emit the gitignored `*_cubin.h` kernel-string
  headers; a fresh clone builds with no manual steps.
- `C_ele` is a ReaxFF macro (332.06371) — never name a C++ variable `C_ele`.

## Build
```bash
mkdir build && cd build
cmake ../cmake -D PKG_GPU=yes -D PKG_REAXFF=yes -D GPU_API=metal \
      -D BUILD_MPI=no -D CMAKE_BUILD_TYPE=Release
cmake --build . -j8
```

## Canonical remote
`unified` → github.com/akvaithi/lammps-metal (this is `main`'s home). The local
`origin` still points at the old archived `lammps-metal-reaxff` repo and lags on
purpose — push to `unified`, not `origin`.

## Key docs
- `METAL_PORT_STATUS.md` — detailed status, the precision-bug write-up, how the GPU
  neighbor build works, upstreaming notes.
- `REAXFF_FORCES_PLAN.md` — term-by-term roadmap + validation harness for the
  remaining ReaxFF force work (read before writing any force kernel).

## Author
Arun Vaithianathan — akvaithi.page — TAMU NETL/ARPA-E graphite-from-coke project.
