#include <metal_stdlib>
using namespace metal;

#if defined(_SINGLE_SINGLE)
typedef float numtyp;
typedef float2 numtyp2;
typedef float4 numtyp4;
typedef float acctyp;
typedef float2 acctyp2;
typedef float4 acctyp4;
#define acctyp3 float3
#elif defined(_DOUBLE_DOUBLE)
// Metal doesn't support double precision. If we reach here, compilation will fail (as expected).
typedef float numtyp;
typedef float4 numtyp4;
typedef float acctyp;
#define acctyp3 float3
#else // MIXED
typedef float numtyp;
typedef float2 numtyp2;
typedef float4 numtyp4;
typedef float acctyp;
typedef float2 acctyp2;
typedef float4 acctyp4;
#define acctyp3 float3
#endif

#define SBBITS 30
#define NEIGHMASK 0x3FFFFFFF

kernel void k_lj(
    device const numtyp4 *x_ [[buffer(0)]],
    device const numtyp4 *lj1 [[buffer(1)]],
    device const numtyp4 *lj3 [[buffer(2)]],
    constant int &lj_types [[buffer(3)]],
    device const numtyp *sp_lj [[buffer(4)]],
    device const int *dev_nbor [[buffer(5)]],
    device const int *dev_packed [[buffer(6)]],
    device packed_float3 *ans [[buffer(7)]],
    device acctyp *engv [[buffer(8)]],
    constant int &eflag [[buffer(9)]],
    constant int &vflag [[buffer(10)]],
    constant int &inum [[buffer(11)]],
    constant int &nbor_pitch [[buffer(12)]],
    constant int &t_per_atom [[buffer(13)]],
    uint tid [[thread_position_in_grid]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]])
{
    int ii = tid;

    acctyp4 f = {0.0, 0.0, 0.0, 0.0};
    acctyp energy = 0.0;
    acctyp virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // The grid is rounded up to full threadgroups, so some threads have ii >= inum.
    // They must NOT read the neighbor list, but they DO fall through to the
    // threadgroup reduction below with zero contribution (returning early here would
    // leave their shared-memory slot uninitialized and corrupt the reduced sum).
    if (ii < inum) {
    int i = dev_nbor[ii];
    numtyp4 ix = x_[i];
    int itype = (int)ix.w;

    int nbor_begin = ii + nbor_pitch;
    int numj = dev_nbor[nbor_begin];

    int n_stride;
    int nbor_end;
    if (dev_nbor == dev_packed) {
        nbor_begin += nbor_pitch + ii * (t_per_atom - 1);
        n_stride = t_per_atom * nbor_pitch;
        nbor_end = nbor_begin + (numj / t_per_atom) * n_stride + (numj & (t_per_atom - 1));
        // offset is 0 for now since we don't use SIMD thread_index_in_threadgroup offsets
    } else {
        nbor_begin += nbor_pitch;
        nbor_begin = dev_nbor[nbor_begin];
        nbor_end = nbor_begin + numj;
        n_stride = t_per_atom;
    }

    for (int nbor = nbor_begin; nbor < nbor_end; nbor += n_stride) {
        int j = dev_packed[nbor];
        numtyp factor_lj = sp_lj[(j >> SBBITS) & 3];
        j &= NEIGHMASK;

        numtyp4 jx = x_[j];
        int jtype = (int)jx.w;

        numtyp delx = ix.x - jx.x;
        numtyp dely = ix.y - jx.y;
        numtyp delz = ix.z - jx.z;
        numtyp r2inv = delx*delx + dely*dely + delz*delz;

        int mtype = itype*lj_types + jtype;
        if (r2inv < lj1[mtype].z) {
            r2inv = 1.0f/r2inv;
            numtyp r6inv = r2inv*r2inv*r2inv;
            numtyp force = r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
            force *= factor_lj;
            
            f.x += delx*force;
            f.y += dely*force;
            f.z += delz*force;

            if (eflag) {
                numtyp e = r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
                energy += factor_lj*(e-lj3[mtype].z);
            }

            if (vflag) {
                virial[0] += delx*delx*force;
                virial[1] += dely*dely*force;
                virial[2] += delz*delz*force;
                virial[3] += delx*dely*force;
                virial[4] += delx*delz*force;
                virial[5] += dely*delz*force;
            }
        }
    }
    } // end if (ii < inum) — reduction below runs for all threads

    if (eflag == 2) {
        if (ii < inum) engv[ii] = energy * 0.5f;
    } else if (eflag == 1) {
        threadgroup acctyp local_eng[1024];
        local_eng[thread_index_in_threadgroup] = energy;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
            if (thread_index_in_threadgroup < s) local_eng[thread_index_in_threadgroup] += local_eng[thread_index_in_threadgroup + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index_in_threadgroup == 0) engv[threadgroup_id] = local_eng[0] * 0.5f;
    }
    
    if (vflag == 2) {
        if (ii < inum) {
            int v_offset = inum;
            for(int v=0; v<6; ++v) {
                engv[v_offset + v * inum + ii] = virial[v] * 0.5f;
            }
        }
    } else if (vflag == 1) {
        threadgroup acctyp local_vir[6][1024];
        for(int v=0; v<6; ++v) local_vir[v][thread_index_in_threadgroup] = virial[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
            if (thread_index_in_threadgroup < s) {
                for(int v=0; v<6; ++v) local_vir[v][thread_index_in_threadgroup] += local_vir[v][thread_index_in_threadgroup + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index_in_threadgroup == 0) {
            int v_offset = threadgroups_per_grid;
            for(int v=0; v<6; ++v) {
                engv[v_offset + v * threadgroups_per_grid + threadgroup_id] = local_vir[v][0] * 0.5f;
            }
        }
    }
    if (ii < inum) ans[ii] = packed_float3(f.x, f.y, f.z);
}

kernel void k_lj_fast(
    device const numtyp4 *x_ [[buffer(0)]],
    device const numtyp4 *lj1 [[buffer(1)]],
    device const numtyp4 *lj3 [[buffer(2)]],
    device const numtyp *sp_lj [[buffer(3)]],
    device const int *dev_nbor [[buffer(4)]],
    device const int *dev_packed [[buffer(5)]],
    device packed_float3 *ans [[buffer(6)]],
    device acctyp *engv [[buffer(7)]],
    constant int &eflag [[buffer(8)]],
    constant int &vflag [[buffer(9)]],
    constant int &inum [[buffer(10)]],
    constant int &nbor_pitch [[buffer(11)]],
    constant int &t_per_atom [[buffer(12)]],
    uint tid [[thread_position_in_grid]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]]
) {
    int ii = tid;

    acctyp4 f = {0.0, 0.0, 0.0, 0.0};
    acctyp energy = 0.0;
    acctyp virial[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // The grid is rounded up to full threadgroups, so some threads have ii >= inum.
    // They must NOT read the neighbor list, but they DO fall through to the
    // threadgroup reduction below with zero contribution (returning early here would
    // leave their shared-memory slot uninitialized and corrupt the reduced sum).
    if (ii < inum) {
    int i = dev_nbor[ii];
    numtyp4 ix = x_[i];
    int itype = (int)ix.w;

    int nbor_begin = ii + nbor_pitch;
    int numj = dev_nbor[nbor_begin];

    int n_stride;
    int nbor_end;
    if (dev_nbor == dev_packed) {
        nbor_begin += nbor_pitch + ii * (t_per_atom - 1);
        n_stride = t_per_atom * nbor_pitch;
        nbor_end = nbor_begin + (numj / t_per_atom) * n_stride + (numj & (t_per_atom - 1));
        // offset is 0 for non-simd CPU-based dispatch without thread_index_in_threadgroup offsets
    } else {
        nbor_begin += nbor_pitch;
        nbor_begin = dev_nbor[nbor_begin];
        nbor_end = nbor_begin + numj;
        n_stride = t_per_atom;
    }

    for (int nbor = nbor_begin; nbor < nbor_end; nbor += n_stride) {
        int j = dev_packed[nbor];
        numtyp factor_lj = sp_lj[(j >> SBBITS) & 3];
        j &= NEIGHMASK;

        numtyp4 jx = x_[j];
        int jtype = (int)jx.w;

        numtyp delx = ix.x - jx.x;
        numtyp dely = ix.y - jx.y;
        numtyp delz = ix.z - jx.z;
        numtyp r2inv = delx*delx + dely*dely + delz*delz;

        int mtype = itype*11 + jtype; // MAX_SHARED_TYPES = 11
        if (r2inv < lj1[mtype].z) {
            r2inv = 1.0f/r2inv;
            numtyp r6inv = r2inv*r2inv*r2inv;
            numtyp force = r2inv*r6inv*(lj1[mtype].x*r6inv-lj1[mtype].y);
            force *= factor_lj;
            
            f.x += delx*force;
            f.y += dely*force;
            f.z += delz*force;

            if (eflag) {
                numtyp e = r6inv*(lj3[mtype].x*r6inv-lj3[mtype].y);
                energy += factor_lj*(e-lj3[mtype].z);
            }

            if (vflag) {
                virial[0] += delx*delx*force;
                virial[1] += dely*dely*force;
                virial[2] += delz*delz*force;
                virial[3] += delx*dely*force;
                virial[4] += delx*delz*force;
                virial[5] += dely*delz*force;
            }
        }
    }
    } // end if (ii < inum) — reduction below runs for all threads

    if (eflag == 2) {
        if (ii < inum) engv[ii] = energy * 0.5f;
    } else if (eflag == 1) {
        threadgroup acctyp local_eng[1024];
        local_eng[thread_index_in_threadgroup] = energy;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
            if (thread_index_in_threadgroup < s) local_eng[thread_index_in_threadgroup] += local_eng[thread_index_in_threadgroup + s];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index_in_threadgroup == 0) engv[threadgroup_id] = local_eng[0] * 0.5f;
    }
    
    if (vflag == 2) {
        if (ii < inum) {
            int v_offset = inum;
            for(int v=0; v<6; ++v) {
                engv[v_offset + v * inum + ii] = virial[v] * 0.5f;
            }
        }
    } else if (vflag == 1) {
        threadgroup acctyp local_vir[6][1024];
        for(int v=0; v<6; ++v) local_vir[v][thread_index_in_threadgroup] = virial[v];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = threads_per_threadgroup / 2; s > 0; s >>= 1) {
            if (thread_index_in_threadgroup < s) {
                for(int v=0; v<6; ++v) local_vir[v][thread_index_in_threadgroup] += local_vir[v][thread_index_in_threadgroup + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (thread_index_in_threadgroup == 0) {
            int v_offset = threadgroups_per_grid;
            for(int v=0; v<6; ++v) {
                engv[v_offset + v * threadgroups_per_grid + threadgroup_id] = local_vir[v][0] * 0.5f;
            }
        }
    }
    if (ii < inum) ans[ii] = packed_float3(f.x, f.y, f.z);
}
