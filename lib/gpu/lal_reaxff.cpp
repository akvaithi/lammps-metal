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
  
  _allocated = true;
  return 0;
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
