/***************************************************************************
                                 lal_reaxff_ext.cpp
                             -------------------

  Functions for LAMMPS access to ReaxFF acceleration routines.
 ***************************************************************************/

#include <iostream>
#include <cassert>
#include <cmath>

#include "lal_reaxff.h"

using namespace std;
using namespace LAMMPS_AL;

static ReaxFFGPU<PRECISION,ACC_PRECISION> REAXFFMF;

// ---------------------------------------------------------------------------
// Allocate memory on host and device and copy constants to device
// ---------------------------------------------------------------------------
int reaxff_gpu_init(const int ntypes, const int inum, const int nall, const int max_nbors,
                    int &gpu_mode, FILE *screen) {
  REAXFFMF.clear();
  gpu_mode=REAXFFMF.device->gpu_mode();
  double gpu_split=REAXFFMF.device->particle_split();
  int first_gpu=REAXFFMF.device->first_device();
  int last_gpu=REAXFFMF.device->last_device();
  int world_me=REAXFFMF.device->world_me();
  int gpu_rank=REAXFFMF.device->gpu_rank();
  int procs_per_gpu=REAXFFMF.device->procs_per_gpu();

  REAXFFMF.device->init_message(screen,"reaxff",first_gpu,last_gpu);

  bool message=false;
  if (REAXFFMF.device->replica_me()==0 && screen)
    message=true;

  if (message) {
    fprintf(screen,"Initializing Device and compiling on process 0...");
    fflush(screen);
  }

  int init_ok=0;
  if (world_me==0)
    init_ok=REAXFFMF.init(ntypes, inum, nall, max_nbors, gpu_split, screen);

  REAXFFMF.device->world_barrier();
  if (message)
    fprintf(screen,"Done.\n");

  for (int i=0; i<procs_per_gpu; i++) {
    if (message) {
      if (last_gpu-first_gpu==0)
        fprintf(screen,"Initializing Device %d on core %d...",first_gpu,i);
      else
        fprintf(screen,"Initializing Devices %d-%d on core %d...",first_gpu,
                last_gpu,i);
      fflush(screen);
    }
    if (gpu_rank==i && world_me!=0)
      init_ok=REAXFFMF.init(ntypes, inum, nall, max_nbors, gpu_split, screen);

    REAXFFMF.device->serialize_init();
    if (message)
      fprintf(screen,"Done.\n");
  }
  if (message)
    fprintf(screen,"\n");

  if (init_ok==0)
    REAXFFMF.estimate_gpu_overhead();
  return init_ok;
}

void reaxff_gpu_clear() {
  REAXFFMF.clear();
}

void reaxff_qeq_matvec(int nn, int *ilist, int *mask, double *eta, int *type,
                       int *firstnbr, int *numnbrs, int *jlist, double *val,
                       double *x, double *b, int ntypes, int nall, int m_fill, int groupbit) {
  REAXFFMF.qeq_matvec(nn, ilist, mask, eta, type, firstnbr, numnbrs, jlist, val, x, b, ntypes, nall, m_fill, groupbit);
}

void reaxff_gpu_compute(const int ago, const int inum, const int nall, double **host_x,
                        int *host_type, int *ilist, int *numj, int **firstneigh,
                        const bool eflag, const bool vflag, const bool eatom, const bool vatom,
                        int &host_start, const double cpu_time, bool &success) {
  success = true;
  // TODO: implement actual compute execution later
}

double reaxff_gpu_bytes() {
  return REAXFFMF.host_memory_usage();
}
