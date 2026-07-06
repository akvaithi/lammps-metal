#include <metal_stdlib>
#include <metal_atomic>
using namespace metal;

kernel void k_reaxff_stub(uint threadgroup_position_in_grid [[threadgroup_position_in_grid]],
                          uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]) {
    // Stub kernel so init_atomic doesn't fail
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

