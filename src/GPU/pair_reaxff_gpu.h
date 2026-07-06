/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS
// clang-format off
PairStyle(reaxff/gpu,PairReaxFFGPU);
// clang-format on
#else

#ifndef LMP_PAIR_REAXFF_GPU_H
#define LMP_PAIR_REAXFF_GPU_H

#include "pair_reaxff.h"

namespace LAMMPS_NS {

class PairReaxFFGPU : public PairReaxFF {
 public:
  PairReaxFFGPU(class LAMMPS *);
  ~PairReaxFFGPU() override;
  void compute(int, int) override;
  void init_style() override;
  double memory_usage() override;

  enum { GPU_FORCE, GPU_NEIGH, GPU_HYB_NEIGH };

 private:
  int gpu_mode;
  double cpu_time;
};

}    // namespace LAMMPS_NS
#endif
#endif
