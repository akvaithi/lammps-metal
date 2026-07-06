/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "fix_qeq_reaxff_gpu.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "update.h"
#include "gpu_extra.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixQEqReaxFFGPU::FixQEqReaxFFGPU(LAMMPS *lmp, int narg, char **arg) :
  FixQEqReaxFF(lmp, narg, arg)
{
  if (matrix_free)
    error->all(FLERR, "Matrix-free QEq is not supported in the GPU package");
}

/* ---------------------------------------------------------------------- */

FixQEqReaxFFGPU::~FixQEqReaxFFGPU()
{
}

/* ---------------------------------------------------------------------- */

void FixQEqReaxFFGPU::init()
{
  FixQEqReaxFF::init();
}

/* ---------------------------------------------------------------------- */

extern void reaxff_qeq_matvec(int nn, int *ilist, int *mask, double *eta, int *type,
                              int *firstnbr, int *numnbrs, int *jlist, double *val,
                              double *x, double *b, int ntypes, int nall, int m_fill, int groupbit);

void FixQEqReaxFFGPU::sparse_matvec(sparse_matrix *A, double *x, double *b)
{
  int nall = atom->nlocal + atom->nghost;
  reaxff_qeq_matvec(nn, ilist, atom->mask, eta, atom->type, A->firstnbr, A->numnbrs,
                    A->jlist, A->val, x, b, atom->ntypes, nall, m_fill, groupbit);
}
