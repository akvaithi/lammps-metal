# Porting ReaxFF forces to Metal — implementation plan

**Status:** not started. `pair reaxff/gpu` currently computes forces on the CPU
(via the parent `PairReaxFF`) and offloads only the QEq matvec. This document is
the roadmap for moving the ReaxFF **force/energy** computation itself onto the GPU.

Read this before writing any force kernel. ReaxFF is intricate and tightly
coupled; the difference between "looks right" and "is right" is invisible without
the per-term validation harness described below. Do it term by term, validated,
in the order given.

## Why this is a build, not a port

Mainline LAMMPS has **no** `reaxff/gpu` in the GPU/geryon package — only the
KOKKOS backend (`pair reaxff/kk`) accelerates ReaxFF, and that is thousands of
lines. So there is no CUDA/OpenCL kernel in this tree to translate; the force
kernels must be written from the CPU physics in `src/REAXFF/reaxff_*.cpp`. Budget
accordingly (this is a multi-month effort, not a session).

## The computation, and its data dependencies

`reaxff_forces.cpp::Compute_Forces` runs these in order; each writes into
`data->my_en.*` (see `reaxff_types.h:244`) and into per-atom forces:

| # | Term | Source | Energy | Notes |
|---|------|--------|--------|-------|
| 0 | **Bond order (BO)** | `reaxff_bond_orders.cpp` | — | Foundation. Per bond: uncorrected sigma/pi/pipi BO, then overcoordination corrections. Everything bonded depends on it. |
| 1 | Bonds | `reaxff_bonds.cpp` | `e_bond` | Bond energy from corrected BO. |
| 2 | Lone pair / over- / under-coord | `reaxff_multi_body.cpp` (`Atom_Energy`) | `e_lp,e_ov,e_un` | Per-atom, uses BO sums. |
| 3 | Valence angles (3-body) | `reaxff_valence_angles.cpp` | `e_ang,e_pen,e_coa` | Needs the bond list + BO; builds a three-body list. |
| 4 | Torsion angles (4-body) | `reaxff_torsion_angles.cpp` | `e_tor,e_conj` | Needs the three-body list. |
| 5 | Hydrogen bonds | `reaxff_hydrogen_bonds.cpp` | `e_hb` | Needs an H-bond list. |
| 6 | **vdW + Coulomb (nonbonded)** | `reaxff_nonbonded.cpp` | `e_vdW,e_ele` | Pairwise over the far-neighbor list; uses QEq charges + Taper. Most GPU-amenable. |

Shared prerequisites that must live on the GPU before any term:
- **The far-neighbor list** (`reax_list` FAR_NBRS): ReaxFF builds its own list
  with distances/`dvec` cached, and the bonded terms build the **bond list**,
  **three-body list**, and **hbond list** on top of it. These are not the
  geryon neighbor list; getting them (or an equivalent) on the GPU is the bulk of
  the infrastructure work.
- **Parameters:** `reax_param.tbp[ti][tj]` (two-body: D, alpha, r_vdW, gamma_w,
  gamma, ecore/acore/rcore, p_bo*, …), `sbp[ti]` (single-body), `thbp`/`fbp`
  (three-/four-body), `gp.l[...]` global params, and `workspace->Tap[0..7]`.
  Flatten into GPU-resident float arrays once at init.
- **Charges** `q` from QEq (already produced each step), positions, types.

## Precision

Metal is fp32 only (see `METAL_PORT_STATUS.md`). ReaxFF CPU is fp64. The BO
corrections and angle/torsion terms involve large cancellations and `exp`/`pow`
chains that are precision-sensitive; expect to validate against CPU with a
**loose** tolerance (~1e-4 relative on per-term energy) and to watch energy
conservation over a trajectory. If a term won't hold tolerance in fp32, that term
may need a Kahan-style compensation or to stay on the CPU.

## Validation harness (build this FIRST)

Do not implement a term without being able to check it in isolation.

1. Expose the CPU per-term energies. `data->my_en` already holds them; add them to
   `PairReaxFF::extract` (`pair_reaxff.cpp:701`) or print them under a debug flag,
   so a plain `pair reaxff` run emits `e_vdW`, `e_bond`, … each step.
2. For the term under development, compute it on the GPU **and** on the CPU in the
   same run, and compare the scalar term energy (and, term by term, per-atom
   forces via `f` diff) at step 0 on a small system. `examples/reaxff/FC` (17k
   atoms, C/F, uses QEq) and the smaller `examples/reaxff/{AB,CHO}` are good.
3. Only once the term matches CPU to tolerance, subtract it from the CPU path
   (compute everything-except-this-term on CPU, this term on GPU) and confirm the
   total energy and forces still match. This "one term at a time on the GPU"
   staging keeps every commit validated.

## Recommended order (each step independently validated + committed)

1. **Infrastructure:** get the far-neighbor list + flattened `tbp`/`sbp`/`gp`/`Tap`
   + `q` onto the GPU, reachable from `lal_reaxff.cpp::loop()`. No physics yet;
   validate by round-tripping the data back and diffing against the CPU structs.
2. **Nonbonded (vdW + Coulomb).** Port `vdW_Coulomb_Energy` (`reaxff_nonbonded.cpp`)
   — a self-contained pairwise kernel over the far-neighbor list. This is the
   natural first physics term: no bonded lists, only `tbp` + `Tap` + `q`. Validate
   `e_vdW+e_ele` and the pairwise forces, then offload it (CPU does everything
   else). Delivers a real, correct partial GPU offload.
3. **Bond order (BO).** The gateway to all bonded terms; build the GPU bond list.
4. **Bonds**, then **Atom_Energy** (lp/over/under) — both are per-bond/per-atom
   reductions off the BO, relatively contained.
5. **Valence angles**, then **Torsion**, then **Hydrogen bonds** — the 3-/4-body
   terms, most complex, need the three-body/hbond lists on the GPU.

Ship steps 1–2 first: they turn `pair reaxff/gpu` into "QEq + nonbonded on GPU,
bonded on CPU", which is already a meaningful, fully-correct acceleration and a
clean upstreamable milestone before tackling the bonded machinery.

## Where the code goes
- Kernels: `lib/gpu/reaxff.metal` (added to the `neighbor_gpu`-style generated
  header by `generate_metals.py`, same as `lj.metal`).
- Host glue / launches: `lib/gpu/lal_reaxff.cpp` (`loop()` is the empty stub to
  fill), `lal_reaxff.h`.
- LAMMPS entry: `src/GPU/pair_reaxff_gpu.cpp::compute()` — replace the current
  `PairReaxFF::compute()` fallback with the GPU path as terms come online (keep the
  CPU fallback for un-ported terms).
