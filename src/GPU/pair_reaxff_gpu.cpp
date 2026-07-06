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

#include "reaxff_api.h"
#include <cstdlib>
#include <vector>

// External functions from GPU library for ReaxFF
int reaxff_gpu_init(const int ntypes, const int nlocal, const int nall, const int max_nbors,
                    int &gpu_mode, FILE *screen);
void reaxff_gpu_clear();
void reaxff_gpu_nonbonded_energy(int npairs, const float *qiqj, const float *rij,
                                 const int *mtype, const float *Tap, int nt2,
                                 const float *p_D, const float *p_alpha, const float *p_rvdW,
                                 const float *p_gammaw, const float *p_ecore, const float *p_acore,
                                 const float *p_rcore, const float *p_gamma, const float *p_lgcij,
                                 const float *p_lgre, float c_ele, float p_vdW1, int vdw_type,
                                 int lgflag, double *e_vdW_out, double *e_ele_out);
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
  // The ReaxFF force kernels are not ported to the GPU yet, so compute the forces
  // on the CPU via the parent PairReaxFF. The GPU device set up in init_style() is
  // used to offload the QEq charge-equilibration matvec (fix qeq/reaxff/gpu). This
  // makes `pair reaxff/gpu` real ReaxFF with a GPU-accelerated QEq, rather than a
  // zero-force stub, until the force kernels land.
  PairReaxFF::compute(eflag, vflag);

  // Nonbonded (Coulomb) energy validation: recompute the ReaxFF electrostatic
  // energy on the GPU from the far-neighbor list ReaxFF just built, and compare to
  // the CPU value (my_en.e_ele). Enable with REAXFF_GPU_VALIDATE_NB=1. This
  // validates the GPU nonbonded pipeline before the term is actually offloaded;
  // see REAXFF_FORCES_PLAN.md.
  if (std::getenv("REAXFF_GPU_VALIDATE_NB")) {
    using namespace ReaxFF;
    reax_system  *sys  = api->system;
    reax_list    *fnb  = api->lists + FAR_NBRS;
    storage      *ws   = api->workspace;
    const double  nonb_cut = api->control->nonb_cut;
    const double  SMALL = 0.0001;
    const int     natoms = sys->n;
    const int     NT = sys->reax_param.num_atom_types;

    // Flatten the two-body parameters into per-type-pair tables (NT*NT).
    const int nt2 = NT * NT;
    std::vector<float> pD(nt2), palpha(nt2), prvdW(nt2), pgammaw(nt2), pecore(nt2),
                       pacore(nt2), prcore(nt2), pgamma(nt2), plgcij(nt2), plgre(nt2);
    for (int a = 0; a < NT; ++a)
      for (int b = 0; b < NT; ++b) {
        const two_body_parameters &t = sys->reax_param.tbp[a][b];
        const int m = a * NT + b;
        pD[m]=t.D; palpha[m]=t.alpha; prvdW[m]=t.r_vdW; pgammaw[m]=t.gamma_w;
        pecore[m]=t.ecore; pacore[m]=t.acore; prcore[m]=t.rcore; pgamma[m]=t.gamma;
        plgcij[m]=t.lgcij; plgre[m]=t.lgre;
      }

    std::vector<float> qiqj, rij;
    std::vector<int> mtype;
    qiqj.reserve(natoms * 64); rij.reserve(natoms * 64); mtype.reserve(natoms * 64);
    for (int i = 0; i < natoms; ++i) {
      const int ti = sys->my_atoms[i].type;
      if (ti < 0) continue;
      const rc_tagint orig_i = sys->my_atoms[i].orig_id;
      const double qi = sys->my_atoms[i].q;
      const int start = Start_Index(i, fnb), end = End_Index(i, fnb);
      for (int pj = start; pj < end; ++pj) {
        far_neighbor_data *nbr = &fnb->select.far_nbr_list[pj];
        const int j = nbr->nbr;
        const int tj = sys->my_atoms[j].type;
        if (tj < 0) continue;
        const rc_tagint orig_j = sys->my_atoms[j].orig_id;
        int flag = 0;
        if (nbr->d <= nonb_cut) {
          if (j < natoms) flag = 1;
          else if (orig_i < orig_j) flag = 1;
          else if (orig_i == orig_j) {
            if (nbr->dvec[2] > SMALL) flag = 1;
            else if (fabs(nbr->dvec[2]) < SMALL) {
              if (nbr->dvec[1] > SMALL) flag = 1;
              else if (fabs(nbr->dvec[1]) < SMALL && nbr->dvec[0] > SMALL) flag = 1;
            }
          }
        }
        if (flag) {
          qiqj.push_back((float)(qi * sys->my_atoms[j].q));
          rij.push_back((float)nbr->d);
          mtype.push_back(ti * NT + tj);
        }
      }
    }
    float Tap[8];
    for (int k = 0; k < 8; ++k) Tap[k] = (float)ws->Tap[k];
    double e_vdW_gpu = 0.0, e_ele_gpu = 0.0;
    reaxff_gpu_nonbonded_energy((int)qiqj.size(), qiqj.data(), rij.data(), mtype.data(),
                                Tap, nt2, pD.data(), palpha.data(), prvdW.data(),
                                pgammaw.data(), pecore.data(), pacore.data(), prcore.data(),
                                pgamma.data(), plgcij.data(), plgre.data(), (float)C_ele,
                                (float)sys->reax_param.gp.l[28], sys->reax_param.gp.vdw_type,
                                api->control->lgflag, &e_vdW_gpu, &e_ele_gpu);
    const double e_vdW_cpu = api->data->my_en.e_vdW;
    const double e_ele_cpu = api->data->my_en.e_ele;
    if (screen) {
      fprintf(screen, "[NB-VALIDATE] npairs=%zu vdw_type=%d\n", qiqj.size(),
              sys->reax_param.gp.vdw_type);
      fprintf(screen, "[NB-VALIDATE] e_vdW CPU=%.6f GPU=%.6f rel=%.3e\n", e_vdW_cpu,
              e_vdW_gpu, fabs(e_vdW_cpu - e_vdW_gpu) / (fabs(e_vdW_cpu) + 1e-30));
      fprintf(screen, "[NB-VALIDATE] e_ele CPU=%.6f GPU=%.6f rel=%.3e\n", e_ele_cpu,
              e_ele_gpu, fabs(e_ele_cpu - e_ele_gpu) / (fabs(e_ele_cpu) + 1e-30));
    }
  }
  return;

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
  // Forces are computed on the CPU by the parent (see compute()); use its normal
  // ReaxFF neighbor request and newton setting. The GPU device initialized here is
  // used only to offload the QEq matvec (fix qeq/reaxff/gpu), so no `newton off`
  // or `neigh no` is required from the user.
  PairReaxFF::init_style();

  int maxneigh = neighbor->oneatom;
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
