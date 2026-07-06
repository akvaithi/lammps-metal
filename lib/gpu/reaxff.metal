#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void k_reaxff_stub(uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                          uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {
    // Stub kernel so init_atomic doesn't fail
}

// ReaxFF electrostatic (Coulomb) energy per counted pair, matching
// reaxff_nonbonded.cpp: e_ele = C_ele * qi*qj * Tap(r) / (r^3 + gamma)^(1/3),
// where Tap is the 7th-order taper (Horner form, Tap[7] is the r^7 coeff) and
// `gamma` is the tbp gamma parameter (already stored as gamma^-3). Host sums the
// per-pair output in double. Energy-only validation step for the ReaxFF force port.
kernel void k_reaxff_coul(device const float* qiqj     [[buffer(0)]],
                          device const float* rij      [[buffer(1)]],
                          device const float* gamma_ij [[buffer(2)]],
                          device const float* Tap      [[buffer(3)]],
                          constant float& C_ele        [[buffer(4)]],
                          constant int& npairs         [[buffer(5)]],
                          device float* e_out          [[buffer(6)]],
                          uint p [[thread_position_in_grid]])
{
    if ((int)p >= npairs) return;
    float r = rij[p];
    float Tap_v = Tap[7]*r + Tap[6];
    Tap_v = Tap_v*r + Tap[5];
    Tap_v = Tap_v*r + Tap[4];
    Tap_v = Tap_v*r + Tap[3];
    Tap_v = Tap_v*r + Tap[2];
    Tap_v = Tap_v*r + Tap[1];
    Tap_v = Tap_v*r + Tap[0];
    float dr3gamij_1 = r*r*r + gamma_ij[p];
    float dr3gamij_3 = pow(dr3gamij_1, 0.33333333333333f);
    e_out[p] = C_ele * qiqj[p] * Tap_v / dr3gamij_3;
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

