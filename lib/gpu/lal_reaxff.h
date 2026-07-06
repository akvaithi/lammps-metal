/***************************************************************************
                                  lal_reaxff.h
                             -------------------

  Class for acceleration of the reaxff pair style.
 ***************************************************************************/

#ifndef LAL_REAXFF_H
#define LAL_REAXFF_H

#include "lal_base_charge.h"

namespace LAMMPS_AL {

template <class numtyp, class acctyp>
class ReaxFFGPU : public BaseCharge<numtyp, acctyp> {
 public:
  ReaxFFGPU();
  ~ReaxFFGPU();

  int init(const int ntypes, const int inum, const int nall, const int max_nbors,
           const double gpu_split, FILE *screen);

  void clear();
  
  double host_memory_usage() const;

 private:
  bool _allocated;
  int loop(const int eflag, const int vflag);
  
 public:
  UCL_Program *reaxff_program;
  UCL_Kernel k_qeq_matvec;
  UCL_D_Vec<int> d_ilist;
  UCL_D_Vec<int> d_mask;
  UCL_D_Vec<float> d_eta;
  UCL_D_Vec<int> d_type;
  UCL_D_Vec<int> d_firstnbr;
  UCL_D_Vec<int> d_numnbrs;
  UCL_D_Vec<int> d_jlist;
  UCL_D_Vec<float> d_val;
  UCL_D_Vec<float> d_x;
  UCL_D_Vec<float> d_b;

  void qeq_matvec(int nn, int *ilist, int *mask, double *eta, int *type,
                  int *firstnbr, int *numnbrs, int *jlist, double *val,
                  double *x, double *b, int ntypes, int nall, int m_fill, int groupbit);

  // Nonbonded (vdW + Coulomb energy) validation step for the ReaxFF force port.
  UCL_Kernel k_reaxff_nonbonded;
  UCL_D_Vec<float> d_qiqj, d_rij, d_tap, d_evdw_out, d_eele_out;
  UCL_D_Vec<int> d_mtype;
  UCL_D_Vec<float> d_pD, d_palpha, d_prvdW, d_pgammaw, d_pecore, d_pacore,
                   d_prcore, d_pgamma, d_plgcij, d_plgre;
  bool _nb_alloc = false;
  int _nb_cap = 0, _nt2_cap = 0;
  // Sum the ReaxFF nonbonded energy over `npairs` counted pairs; params are flat
  // per-type-pair tables of length nt2 = NT*NT. Returns e_vdW and e_ele.
  void nonbonded_energy(int npairs, const float *qiqj, const float *rij,
                        const int *mtype, const float *Tap, int nt2,
                        const float *p_D, const float *p_alpha, const float *p_rvdW,
                        const float *p_gammaw, const float *p_ecore, const float *p_acore,
                        const float *p_rcore, const float *p_gamma, const float *p_lgcij,
                        const float *p_lgre, float c_ele, float p_vdW1, int vdw_type,
                        int lgflag, double &e_vdW_out, double &e_ele_out);
};

}

#endif
