/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org
------------------------------------------------------------------------- */

#include "pair_reaxff_gpu.h"
#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "gpu_extra.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "suffix.h"
#include "update.h"

// External functions from GPU library for ReaxFF
int reaxff_gpu_init(const int ntypes, const int nlocal, const int nall, const int max_nbors,
                    int &gpu_mode, FILE *screen);
void reaxff_gpu_clear();
void reaxff_gpu_compute(const int ago, const int inum, const int nall, double **host_x,
                        int *host_type, int *ilist, int *numj, int **firstneigh,
                        const bool eflag, const bool vflag, const bool eatom, const bool vatom,
                        int &host_start, const double cpu_time, bool &success);
double reaxff_gpu_bytes();

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairReaxFFGPU::PairReaxFFGPU(LAMMPS *lmp) : PairReaxFF(lmp), gpu_mode(GPU_FORCE)
{
  respa_enable = 0;
  cpu_time = 0.0;
  suffix_flag |= Suffix::GPU;
  GPU_EXTRA::gpu_ready(lmp->modify, lmp->error);
}

/* ----------------------------------------------------------------------
   free all arrays
------------------------------------------------------------------------- */

PairReaxFFGPU::~PairReaxFFGPU()
{
  reaxff_gpu_clear();
}

/* ---------------------------------------------------------------------- */

void PairReaxFFGPU::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  int nall = atom->nlocal + atom->nghost;
  int inum, host_start;

  bool success = true;
  int *ilist, *numneigh, **firstneigh;
  
  if (gpu_mode != GPU_FORCE) {
    error->all(FLERR, "ReaxFF GPU currently only supports GPU_FORCE mode");
  } else {
    inum = list->inum;
    ilist = list->ilist;
    numneigh = list->numneigh;
    firstneigh = list->firstneigh;
    reaxff_gpu_compute(neighbor->ago, inum, nall, atom->x, atom->type,
                       ilist, numneigh, firstneigh, eflag, vflag, eflag_atom, vflag_atom,
                       host_start, cpu_time, success);
  }
  if (!success)
    error->one(FLERR, "Insufficient memory on accelerator");
}

/* ---------------------------------------------------------------------- */

void PairReaxFFGPU::init_style()
{
  if (force->newton_pair)
    error->all(FLERR, "Cannot use newton pair with reaxff/gpu pair style");

  // Call the base class init_style
  force->newton_pair = 1;
  PairReaxFF::init_style();
  force->newton_pair = 0;

  // initialize GPU
  int maxneigh = neighbor->oneatom;
  int irequest = neighbor->request(this, instance_me);
  
  // Actually ReaxFF relies heavily on ReaxFF API workspace and structures.
  // We will need to pass the ReaxFF API object to the GPU init eventually.
  int success = reaxff_gpu_init(atom->ntypes + 1, atom->nlocal, atom->nlocal + atom->nghost, maxneigh,
                                gpu_mode, screen);
  GPU_EXTRA::check_flag(success, error, world);
}

/* ---------------------------------------------------------------------- */

double PairReaxFFGPU::memory_usage()
{
  double bytes = PairReaxFF::memory_usage();
  return bytes + reaxff_gpu_bytes();
}
