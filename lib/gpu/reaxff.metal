#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void k_reaxff_stub(uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                          uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {
    // Stub kernel so init_atomic doesn't fail
}

// ReaxFF nonbonded (van der Waals + Coulomb) energy per counted pair, matching
// reaxff_nonbonded.cpp::vdW_Coulomb_Energy. Two-body parameters are passed as
// flat per-type-pair tables indexed by mtype = ti*NT + tj; Tap is the 7th-order
// taper (Horner, Tap[7] is the r^7 coeff); `gamma` is the tbp gamma (stored as
// gamma^-3). Host sums the per-pair outputs in double. Energy-only validation
// step for the ReaxFF force port (forces come next).
kernel void k_reaxff_nonbonded(device const float* qiqj    [[buffer(0)]],
                               device const float* rij     [[buffer(1)]],
                               device const int*   mtype   [[buffer(2)]],
                               device const float* Tap     [[buffer(3)]],
                               device const float* p_D     [[buffer(4)]],
                               device const float* p_alpha [[buffer(5)]],
                               device const float* p_rvdW  [[buffer(6)]],
                               device const float* p_gammaw[[buffer(7)]],
                               device const float* p_ecore [[buffer(8)]],
                               device const float* p_acore [[buffer(9)]],
                               device const float* p_rcore [[buffer(10)]],
                               device const float* p_gamma [[buffer(11)]],
                               device const float* p_lgcij [[buffer(12)]],
                               device const float* p_lgre  [[buffer(13)]],
                               constant float& C_ele       [[buffer(14)]],
                               constant float& p_vdW1      [[buffer(15)]],
                               constant int& vdw_type      [[buffer(16)]],
                               constant int& lgflag        [[buffer(17)]],
                               constant int& npairs        [[buffer(18)]],
                               device float* evdw_out      [[buffer(19)]],
                               device float* eele_out      [[buffer(20)]],
                               uint p [[thread_position_in_grid]])
{
    if ((int)p >= npairs) return;
    float r = rij[p];
    int m = mtype[p];

    float Tap_v = Tap[7]*r + Tap[6];
    Tap_v = Tap_v*r + Tap[5];
    Tap_v = Tap_v*r + Tap[4];
    Tap_v = Tap_v*r + Tap[3];
    Tap_v = Tap_v*r + Tap[2];
    Tap_v = Tap_v*r + Tap[1];
    Tap_v = Tap_v*r + Tap[0];

    // van der Waals
    float e_vdW;
    if (vdw_type == 1 || vdw_type == 3) {   // shielding
        float p_vdW1i = 1.0f / p_vdW1;
        float powr  = pow(r, p_vdW1);
        float powgi = pow(1.0f / p_gammaw[m], p_vdW1);
        float fn13  = pow(powr + powgi, p_vdW1i);
        float x     = 1.0f - fn13 / p_rvdW[m];
        e_vdW = p_D[m] * (exp(p_alpha[m]*x) - 2.0f*exp(0.5f*p_alpha[m]*x));
    } else {                                 // no shielding
        float x = 1.0f - r / p_rvdW[m];
        e_vdW = p_D[m] * (exp(p_alpha[m]*x) - 2.0f*exp(0.5f*p_alpha[m]*x));
    }
    float e_nb = Tap_v * e_vdW;
    if (vdw_type == 2 || vdw_type == 3) {    // inner-wall core
        float e_core = p_ecore[m] * exp(p_acore[m] * (1.0f - r/p_rcore[m]));
        e_nb += Tap_v * e_core;
        if (lgflag) {                        // lg dispersion correction
            float r6 = pow(r, 6.0f), re6 = pow(p_lgre[m], 6.0f);
            e_nb += Tap_v * (-(p_lgcij[m] / (r6 + re6)));
        }
    }
    evdw_out[p] = e_nb;

    // Coulomb
    float dr3gamij_3 = pow(r*r*r + p_gamma[m], 0.33333333333333f);
    eele_out[p] = C_ele * qiqj[p] * Tap_v / dr3gamij_3;
}

// ReaxFF nonbonded (vdW + Coulomb) FORCES, one thread per counted pair, matching
// the CEvd/CEclmb derivatives in reaxff_nonbonded.cpp. Forces are accumulated into
// a 3*nall device array with atomics: f[i] += -(CEvd+CEclmb)*dvec, f[j] -= same.
kernel void k_reaxff_nb_force(device const int*   ai      [[buffer(0)]],
                              device const int*   aj      [[buffer(1)]],
                              device const float* dvx     [[buffer(2)]],
                              device const float* dvy     [[buffer(3)]],
                              device const float* dvz     [[buffer(4)]],
                              device const float* rij     [[buffer(5)]],
                              device const float* qiqj    [[buffer(6)]],
                              device const int*   mtype   [[buffer(7)]],
                              device const float* Tap     [[buffer(8)]],
                              device const float* p_D     [[buffer(9)]],
                              device const float* p_alpha [[buffer(10)]],
                              device const float* p_rvdW  [[buffer(11)]],
                              device const float* p_gammaw[[buffer(12)]],
                              device const float* p_ecore [[buffer(13)]],
                              device const float* p_acore [[buffer(14)]],
                              device const float* p_rcore [[buffer(15)]],
                              device const float* p_gamma [[buffer(16)]],
                              device const float* p_lgcij [[buffer(17)]],
                              device const float* p_lgre  [[buffer(18)]],
                              constant float& C_ele       [[buffer(19)]],
                              constant float& p_vdW1      [[buffer(20)]],
                              constant int& vdw_type      [[buffer(21)]],
                              constant int& lgflag        [[buffer(22)]],
                              constant int& npairs        [[buffer(23)]],
                              device atomic_float* f      [[buffer(24)]],
                              uint p [[thread_position_in_grid]])
{
    if ((int)p >= npairs) return;
    float r = rij[p];
    int m = mtype[p];

    float Tap_v = Tap[7]*r + Tap[6];
    Tap_v = Tap_v*r + Tap[5]; Tap_v = Tap_v*r + Tap[4]; Tap_v = Tap_v*r + Tap[3];
    Tap_v = Tap_v*r + Tap[2]; Tap_v = Tap_v*r + Tap[1]; Tap_v = Tap_v*r + Tap[0];

    float dTap = 7.0f*Tap[7]*r + 6.0f*Tap[6];
    dTap = dTap*r + 5.0f*Tap[5]; dTap = dTap*r + 4.0f*Tap[4]; dTap = dTap*r + 3.0f*Tap[3];
    dTap = dTap*r + 2.0f*Tap[2]; dTap += Tap[1]/r;

    // van der Waals CEvd
    float CEvd;
    if (vdw_type == 1 || vdw_type == 3) {
        float p_vdW1i = 1.0f / p_vdW1;
        float powr  = pow(r, p_vdW1);
        float powgi = pow(1.0f / p_gammaw[m], p_vdW1);
        float fn13  = pow(powr + powgi, p_vdW1i);
        float x     = 1.0f - fn13 / p_rvdW[m];
        float exp1  = exp(p_alpha[m]*x), exp2 = exp(0.5f*p_alpha[m]*x);
        float e_vdW = p_D[m]*(exp1 - 2.0f*exp2);
        float dfn13 = pow(powr + powgi, p_vdW1i - 1.0f) * pow(r, p_vdW1 - 2.0f);
        CEvd = dTap*e_vdW - Tap_v*p_D[m]*(p_alpha[m]/p_rvdW[m])*(exp1 - exp2)*dfn13;
    } else {
        float x     = 1.0f - r / p_rvdW[m];
        float exp1  = exp(p_alpha[m]*x), exp2 = exp(0.5f*p_alpha[m]*x);
        float e_vdW = p_D[m]*(exp1 - 2.0f*exp2);
        CEvd = dTap*e_vdW - Tap_v*p_D[m]*(p_alpha[m]/p_rvdW[m])*(exp1 - exp2)/r;
    }
    if (vdw_type == 2 || vdw_type == 3) {
        float e_core  = p_ecore[m]*exp(p_acore[m]*(1.0f - r/p_rcore[m]));
        float de_core = -(p_acore[m]/p_rcore[m])*e_core;
        CEvd += dTap*e_core + Tap_v*de_core/r;
        if (lgflag) {
            float r5 = pow(r,5.0f), r6 = pow(r,6.0f), re6 = pow(p_lgre[m],6.0f);
            float e_lg  = -(p_lgcij[m]/(r6 + re6));
            float de_lg = -6.0f*e_lg*r5/(r6 + re6);
            CEvd += dTap*e_lg + Tap_v*de_lg/r;
        }
    }

    // Coulomb CEclmb
    float dr3gamij_1 = r*r*r + p_gamma[m];
    float dr3gamij_3 = pow(dr3gamij_1, 0.33333333333333f);
    float CEclmb = C_ele*qiqj[p]*(dTap - Tap_v*r/dr3gamij_1)/dr3gamij_3;

    float fc = -(CEvd + CEclmb);
    int i = ai[p], j = aj[p];
    atomic_fetch_add_explicit(&f[3*i+0],  fc*dvx[p], memory_order_relaxed);
    atomic_fetch_add_explicit(&f[3*i+1],  fc*dvy[p], memory_order_relaxed);
    atomic_fetch_add_explicit(&f[3*i+2],  fc*dvz[p], memory_order_relaxed);
    atomic_fetch_add_explicit(&f[3*j+0], -fc*dvx[p], memory_order_relaxed);
    atomic_fetch_add_explicit(&f[3*j+1], -fc*dvy[p], memory_order_relaxed);
    atomic_fetch_add_explicit(&f[3*j+2], -fc*dvz[p], memory_order_relaxed);
}

kernel void k_qeq_matvec(device const int* ilist [[buffer(0)]],
                         device const int* mask [[buffer(1)]],
                         device const float* eta [[buffer(2)]],
                         device const int* type [[buffer(3)]],
                         device const int* firstnbr [[buffer(4)]],
                         device const int* numnbrs [[buffer(5)]],
                         device const int* jlist [[buffer(6)]],
                         device const float* val [[buffer(7)]],
                         device const float* x [[buffer(8)]],
                         device atomic_float* b [[buffer(9)]],
                         constant int& inum [[buffer(10)]],
                         constant int& groupbit [[buffer(11)]],
                         uint ii [[thread_position_in_grid]])
{
    if (ii >= inum) return;
    
    int i = ilist[ii];
    if (mask[i] & groupbit) {
        float b_i = eta[type[i]] * x[i];
        
        int start = firstnbr[i];
        int n_nbrs = numnbrs[i];
        for (int itr_j = start; itr_j < start + n_nbrs; ++itr_j) {
            int j = jlist[itr_j];
            float v = val[itr_j];
            
            b_i += v * x[j];
            
            // atomic add to b[j]
            atomic_fetch_add_explicit(&b[j], v * x[i], memory_order_relaxed);
        }
        
        // atomic add to b[i]
        atomic_fetch_add_explicit(&b[i], b_i, memory_order_relaxed);
    }
}

