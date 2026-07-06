/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(qeq/reaxff/gpu,FixQEqReaxFFGPU);
// clang-format on
#else

#ifndef LMP_FIX_QEQ_REAXFF_GPU_H
#define LMP_FIX_QEQ_REAXFF_GPU_H

#include "fix_qeq_reaxff.h"

namespace LAMMPS_NS {

class FixQEqReaxFFGPU : public FixQEqReaxFF {
 public:
  FixQEqReaxFFGPU(class LAMMPS *, int, char **);
  ~FixQEqReaxFFGPU() override;
  void init() override;
  void sparse_matvec(sparse_matrix *, double *, double *) override;
};

}    // namespace LAMMPS_NS
#endif
#endif
