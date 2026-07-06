/***************************************************************************
                                  lal_reaxff.cpp
                             -------------------

  Class for acceleration of the reaxff pair style.
 ***************************************************************************/

#include <iostream>
#include <vector>
#include "lal_reaxff.h"
#include "reaxff_cubin.h"

using namespace LAMMPS_AL;

#define ReaxFFGPUT ReaxFFGPU<numtyp, acctyp>

template <class numtyp, class acctyp>
ReaxFFGPUT::ReaxFFGPU() : BaseCharge<numtyp,acctyp>(), _allocated(false) {
}

template <class numtyp, class acctyp>
ReaxFFGPUT::~ReaxFFGPU() {
  clear();
}

template <class numtyp, class acctyp>
int ReaxFFGPU<numtyp, acctyp>::init(const int ntypes, const int inum, const int nall, const int max_nbors,
                                    const double gpu_split, FILE *screen) {
  int success = this->init_atomic(inum, nall, max_nbors, 0, 0.0, gpu_split, screen, reaxff, "k_qeq_matvec");
  if (success != 0)
    return success;

  this->reaxff_program = new UCL_Program(*(this->device->gpu));
  success = this->reaxff_program->load_string(reaxff, this->device->compile_string().c_str());
  if (success != UCL_SUCCESS) return success;

  k_qeq_matvec.set_function(*(this->reaxff_program), "k_qeq_matvec");
  k_reaxff_nonbonded.set_function(*(this->reaxff_program), "k_reaxff_nonbonded");

  _allocated = true;
  return 0;
}

// Sum the ReaxFF nonbonded (vdW + Coulomb) energy over a host-provided list of
// counted pairs. Energy-only validation against CPU my_en.e_vdW / my_en.e_ele.
template <class numtyp, class acctyp>
void ReaxFFGPUT::nonbonded_energy(int npairs, const float *qiqj, const float *rij,
                                  const int *mtype, const float *Tap, int nt2,
                                  const float *p_D, const float *p_alpha, const float *p_rvdW,
                                  const float *p_gammaw, const float *p_ecore, const float *p_acore,
                                  const float *p_rcore, const float *p_gamma, const float *p_lgcij,
                                  const float *p_lgre, float c_ele, float p_vdW1, int vdw_type,
                                  int lgflag, double &e_vdW_out, double &e_ele_out) {
  e_vdW_out = e_ele_out = 0.0;
  if (npairs <= 0) return;
  auto *dev = this->device->gpu;
  if (!_nb_alloc || npairs > _nb_cap) {
    if (_nb_alloc) { d_qiqj.clear(); d_rij.clear(); d_mtype.clear(); d_evdw_out.clear(); d_eele_out.clear(); }
    d_qiqj.alloc(npairs, *dev, UCL_READ_ONLY);
    d_rij.alloc(npairs, *dev, UCL_READ_ONLY);
    d_mtype.alloc(npairs, *dev, UCL_READ_ONLY);
    d_evdw_out.alloc(npairs, *dev, UCL_WRITE_ONLY);
    d_eele_out.alloc(npairs, *dev, UCL_WRITE_ONLY);
    _nb_cap = npairs;
  }
  if (!_nb_alloc || nt2 > _nt2_cap) {
    if (_nb_alloc) { d_pD.clear(); d_palpha.clear(); d_prvdW.clear(); d_pgammaw.clear();
                     d_pecore.clear(); d_pacore.clear(); d_prcore.clear(); d_pgamma.clear();
                     d_plgcij.clear(); d_plgre.clear(); }
    d_pD.alloc(nt2,*dev,UCL_READ_ONLY);     d_palpha.alloc(nt2,*dev,UCL_READ_ONLY);
    d_prvdW.alloc(nt2,*dev,UCL_READ_ONLY);  d_pgammaw.alloc(nt2,*dev,UCL_READ_ONLY);
    d_pecore.alloc(nt2,*dev,UCL_READ_ONLY); d_pacore.alloc(nt2,*dev,UCL_READ_ONLY);
    d_prcore.alloc(nt2,*dev,UCL_READ_ONLY); d_pgamma.alloc(nt2,*dev,UCL_READ_ONLY);
    d_plgcij.alloc(nt2,*dev,UCL_READ_ONLY); d_plgre.alloc(nt2,*dev,UCL_READ_ONLY);
    _nt2_cap = nt2;
  }
  if (!_nb_alloc) d_tap.alloc(8, *dev, UCL_READ_ONLY);
  _nb_alloc = true;

  // wrap + copy a host pointer of length n into a device vector
  auto up_f = [&](UCL_D_Vec<float> &d, const float *h, int n) {
    UCL_H_Vec<float> v; v.view(const_cast<float*>(h), n, *dev); ucl_copy(d, v, false); };
  up_f(d_qiqj,qiqj,npairs); up_f(d_rij,rij,npairs); up_f(d_tap,Tap,8);
  up_f(d_pD,p_D,nt2); up_f(d_palpha,p_alpha,nt2); up_f(d_prvdW,p_rvdW,nt2);
  up_f(d_pgammaw,p_gammaw,nt2); up_f(d_pecore,p_ecore,nt2); up_f(d_pacore,p_acore,nt2);
  up_f(d_prcore,p_rcore,nt2); up_f(d_pgamma,p_gamma,nt2); up_f(d_plgcij,p_lgcij,nt2);
  up_f(d_plgre,p_lgre,nt2);
  { UCL_H_Vec<int> v; v.view(const_cast<int*>(mtype), npairs, *dev); ucl_copy(d_mtype, v, false); }

  k_reaxff_nonbonded.clear_args();
  k_reaxff_nonbonded.add_arg(&d_qiqj);   k_reaxff_nonbonded.add_arg(&d_rij);
  k_reaxff_nonbonded.add_arg(&d_mtype);  k_reaxff_nonbonded.add_arg(&d_tap);
  k_reaxff_nonbonded.add_arg(&d_pD);     k_reaxff_nonbonded.add_arg(&d_palpha);
  k_reaxff_nonbonded.add_arg(&d_prvdW);  k_reaxff_nonbonded.add_arg(&d_pgammaw);
  k_reaxff_nonbonded.add_arg(&d_pecore); k_reaxff_nonbonded.add_arg(&d_pacore);
  k_reaxff_nonbonded.add_arg(&d_prcore); k_reaxff_nonbonded.add_arg(&d_pgamma);
  k_reaxff_nonbonded.add_arg(&d_plgcij); k_reaxff_nonbonded.add_arg(&d_plgre);
  k_reaxff_nonbonded.add_arg(&c_ele);    k_reaxff_nonbonded.add_arg(&p_vdW1);
  k_reaxff_nonbonded.add_arg(&vdw_type); k_reaxff_nonbonded.add_arg(&lgflag);
  k_reaxff_nonbonded.add_arg(&npairs);   k_reaxff_nonbonded.add_arg(&d_evdw_out);
  k_reaxff_nonbonded.add_arg(&d_eele_out);

  int block_size = 256;
  int grid_size = (npairs + block_size - 1) / block_size;
  k_reaxff_nonbonded.set_size(grid_size, block_size, dev->cq());
  k_reaxff_nonbonded.run();
  dev->sync();

  std::vector<float> evdw(npairs), eele(npairs);
  UCL_H_Vec<float> hv, he;
  hv.view(evdw.data(), npairs, *dev); ucl_copy(hv, d_evdw_out, false);
  he.view(eele.data(), npairs, *dev); ucl_copy(he, d_eele_out, false);
  double ev = 0.0, ee = 0.0;
  for (int p = 0; p < npairs; ++p) { ev += evdw[p]; ee += eele[p]; }
  e_vdW_out = ev; e_ele_out = ee;
}

template <class numtyp, class acctyp>
void ReaxFFGPU<numtyp, acctyp>::clear() {
  if (!_allocated) return;
  _allocated = false;

  k_qeq_matvec.clear();
  d_ilist.clear();
  d_mask.clear();
  d_eta.clear();
  d_type.clear();
  d_firstnbr.clear();
  d_numnbrs.clear();
  d_jlist.clear();
  d_val.clear();
  d_x.clear();
  d_b.clear();
  
  if (this->reaxff_program) {
    delete this->reaxff_program;
    this->reaxff_program = nullptr;
  }
  
  BaseCharge<numtyp, acctyp>::clear_atomic();
}

template <class numtyp, class acctyp>
double ReaxFFGPU<numtyp, acctyp>::host_memory_usage() const {
  return BaseCharge<numtyp, acctyp>::host_memory_usage_atomic();
}

template <class numtyp, class acctyp>
void ReaxFFGPU<numtyp, acctyp>::qeq_matvec(int nn, int *ilist, int *mask, double *eta, int *type,
                                           int *firstnbr, int *numnbrs, int *jlist, double *val,
                                           double *x, double *b, int ntypes, int nall, int m_fill, int groupbit) {
  // Resize buffers if necessary
  if (nn > d_ilist.numel()) d_ilist.alloc(nn, *(this->device->gpu), UCL_READ_ONLY);
  if (nall > d_mask.numel()) {
    d_mask.alloc(nall, *(this->device->gpu), UCL_READ_ONLY);
    d_type.alloc(nall, *(this->device->gpu), UCL_READ_ONLY);
    d_firstnbr.alloc(nall, *(this->device->gpu), UCL_READ_ONLY);
    d_numnbrs.alloc(nall, *(this->device->gpu), UCL_READ_ONLY);
    d_x.alloc(nall, *(this->device->gpu), UCL_READ_ONLY);
    d_b.alloc(nall, *(this->device->gpu), UCL_READ_WRITE);
  }
  if (ntypes + 1 > d_eta.numel()) d_eta.alloc(ntypes + 1, *(this->device->gpu), UCL_READ_ONLY);
  if (m_fill > d_jlist.numel()) {
    d_jlist.alloc(m_fill, *(this->device->gpu), UCL_READ_ONLY);
    d_val.alloc(m_fill, *(this->device->gpu), UCL_READ_ONLY);
  }
  
  // Create temporary float vectors for double arrays
  std::vector<float> h_eta(ntypes + 1);
  for (int i = 0; i < ntypes + 1; ++i) h_eta[i] = eta[i];
  
  std::vector<float> h_val(m_fill);
  for (int i = 0; i < m_fill; ++i) h_val[i] = val[i];
  
  std::vector<float> h_x(nall);
  for (int i = 0; i < nall; ++i) h_x[i] = x[i];
  
  std::vector<float> h_b(nall, 0.0f);

  // Use UCL_H_Vec to wrap host pointers for ucl_copy
  UCL_H_Vec<int> h_ilist_view, h_mask_view, h_type_view, h_firstnbr_view, h_numnbrs_view, h_jlist_view;
  h_ilist_view.view(ilist, nn, *(this->device->gpu));
  h_mask_view.view(mask, nall, *(this->device->gpu));
  h_type_view.view(type, nall, *(this->device->gpu));
  h_firstnbr_view.view(firstnbr, nall, *(this->device->gpu));
  h_numnbrs_view.view(numnbrs, nall, *(this->device->gpu));
  h_jlist_view.view(jlist, m_fill, *(this->device->gpu));

  UCL_H_Vec<float> h_eta_view, h_val_view, h_x_view, h_b_view;
  h_eta_view.view(h_eta.data(), ntypes + 1, *(this->device->gpu));
  h_val_view.view(h_val.data(), m_fill, *(this->device->gpu));
  h_x_view.view(h_x.data(), nall, *(this->device->gpu));
  h_b_view.view(h_b.data(), nall, *(this->device->gpu));

  // Copy data to GPU
  ucl_copy(d_ilist, h_ilist_view, false);
  ucl_copy(d_mask, h_mask_view, false);
  ucl_copy(d_eta, h_eta_view, false);
  ucl_copy(d_type, h_type_view, false);
  ucl_copy(d_firstnbr, h_firstnbr_view, false);
  ucl_copy(d_numnbrs, h_numnbrs_view, false);
  ucl_copy(d_jlist, h_jlist_view, false);
  ucl_copy(d_val, h_val_view, false);
  ucl_copy(d_x, h_x_view, false);
  ucl_copy(d_b, h_b_view, false);
  
  // Launch kernel
  int block_size = 256;
  int grid_size = (nn + block_size - 1) / block_size;
  
  k_qeq_matvec.clear_args();
  k_qeq_matvec.add_arg(&d_ilist);
  k_qeq_matvec.add_arg(&d_mask);
  k_qeq_matvec.add_arg(&d_eta);
  k_qeq_matvec.add_arg(&d_type);
  k_qeq_matvec.add_arg(&d_firstnbr);
  k_qeq_matvec.add_arg(&d_numnbrs);
  k_qeq_matvec.add_arg(&d_jlist);
  k_qeq_matvec.add_arg(&d_val);
  k_qeq_matvec.add_arg(&d_x);
  k_qeq_matvec.add_arg(&d_b);
  k_qeq_matvec.add_arg(&nn);
  k_qeq_matvec.add_arg(&groupbit);

  // We must set size on the command queue before running
  k_qeq_matvec.set_size(grid_size, block_size, this->device->gpu->cq());
#ifdef UCL_METAL_DEBUG
  { static int c=0; if(c++<1) fprintf(stderr,"[QEQ] k_qeq_matvec running on GPU (inum=%d)\n", nn); }
#endif
  k_qeq_matvec.run();
  this->device->gpu->sync();
  
  // Copy back to CPU
  ucl_copy(h_b_view, d_b, false);
  for (int i = 0; i < nall; ++i) b[i] = h_b[i];
}

template <class numtyp, class acctyp>
int ReaxFFGPUT::loop(const int eflag, const int vflag) {
  // Compute loop over particles
  return 0;
}

template class LAMMPS_AL::ReaxFFGPU<PRECISION,ACC_PRECISION>;
