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
};

}

#endif
