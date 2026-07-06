import os

atom_metal = """
#include <metal_stdlib>
using namespace metal;

struct numtyp4 { float x, y, z, w; };

kernel void kernel_cast_x(
    device numtyp4 *x_type [[buffer(0)]],
    device const uint2 *x [[buffer(1)]], // double is uint2
    device const int *type [[buffer(2)]],
    constant int &nall [[buffer(3)]],
    uint ii [[thread_position_in_grid]]
) {
  if (ii < nall) {
    numtyp4 xt;
    xt.w = type[ii];
    
    // unpack double to float
    int i = ii * 3;
    
    uint2 dx = x[i];
    float fx = as_type<float>((((dx.y >> 31) & 1) << 31) | (((((dx.y >> 20) & 0x7FF) - 1023) + 127) << 23) | ((dx.y & 0xFFFFF) << 3 | (dx.x >> 29)));
    if (((dx.y >> 20) & 0x7FF) == 0) fx = 0.0f;
    xt.x = fx;
    
    uint2 dy = x[i+1];
    float fy = as_type<float>((((dy.y >> 31) & 1) << 31) | (((((dy.y >> 20) & 0x7FF) - 1023) + 127) << 23) | ((dy.y & 0xFFFFF) << 3 | (dy.x >> 29)));
    if (((dy.y >> 20) & 0x7FF) == 0) fy = 0.0f;
    xt.y = fy;
    
    uint2 dz = x[i+2];
    float fz = as_type<float>((((dz.y >> 31) & 1) << 31) | (((((dz.y >> 20) & 0x7FF) - 1023) + 127) << 23) | ((dz.y & 0xFFFFF) << 3 | (dz.x >> 29)));
    if (((dz.y >> 20) & 0x7FF) == 0) fz = 0.0f;
    xt.z = fz;
    
    x_type[ii] = xt;
  }
}
"""

neighbor_cpu_metal = """
#include <metal_stdlib>
using namespace metal;

kernel void kernel_unpack(
    device int *dev_nbor [[buffer(0)]],
    device const int *dev_ij [[buffer(1)]],
    device const int *dev_ij_begin [[buffer(2)]],
    constant int &inum [[buffer(3)]],
    constant int &t_per_atom [[buffer(4)]],
    uint tid [[thread_position_in_grid]],
    uint threadgroup_id [[threadgroup_position_in_grid]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]
) {
  int offset = tid & (t_per_atom - 1);
  int ii = (threadgroup_id * (threads_per_threadgroup / t_per_atom)) + (tid / t_per_atom);

  if (ii < inum) {
    int nbor = ii + inum;
    int numj = dev_nbor[nbor];
    nbor += inum;
    int list = dev_ij_begin[ii];
    int list_end = list + numj;
    list += offset;
    nbor += (ii * (t_per_atom - 1)) + offset;
    int stride = t_per_atom * inum;

    for ( ; list < list_end; list++) {
      dev_nbor[nbor] = dev_ij[list];
      nbor += stride;
    }
  }
}
"""

neighbor_gpu_metal = """
#include <metal_stdlib>
using namespace metal;

// BLOCK_NBOR_BUILD from device.metal kernel_info (info[13]); the Metal runtime
// compile does not receive -D defines, so it is hardcoded here.
#define BLOCK_NBOR_BUILD 256

// Cell-list neighbor build (the LAL_USE_OLD_NEIGHBOR / __APPLE__ path, i.e. the
// simple shared-memory version of lal_neighbor_gpu.cu's calc_neigh_list_cell).
// On Metal the binning/cell-count/sort steps run on the host (gpu_nbor==2 hybrid,
// forced because Metal has no device radix sort), so only this kernel is needed;
// it writes the packed neighbor list the pair kernels read when dev_nbor==dev_packed.
kernel void calc_neigh_list_cell(
    device const float4 *x_               [[buffer(0)]],
    device const int *cell_particle_id    [[buffer(1)]],
    device const int *cell_counts         [[buffer(2)]],
    device int *nbor_list                 [[buffer(3)]],
    device int *host_nbor_list            [[buffer(4)]],
    device int *host_numj                 [[buffer(5)]],
    constant int &_max_nbors              [[buffer(6)]],
    constant float &cutoff_neigh          [[buffer(7)]],
    constant int &ncellx                  [[buffer(8)]],
    constant int &ncelly                  [[buffer(9)]],
    constant int &ncellz                  [[buffer(10)]],
    constant int &inum                    [[buffer(11)]],
    constant int &nt                      [[buffer(12)]],
    constant int &nall                    [[buffer(13)]],
    constant int &t_per_atom              [[buffer(14)]],
    constant int &cells_in_cutoff         [[buffer(15)]],
    uint3 tgid  [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 tptg  [[threads_per_threadgroup]])
{
    int tid = (int)tpitg.x;
    int ix = (int)tgid.x + cells_in_cutoff;
    int iy = (int)tgid.y % (ncelly - cells_in_cutoff*2) + cells_in_cutoff;
    int iz = (int)tgid.y / (ncelly - cells_in_cutoff*2) + cells_in_cutoff;
    int bsx = (int)tptg.x;

    int icell = ix + iy*ncellx + iz*ncellx*ncelly;

    threadgroup int cell_list_sh[BLOCK_NBOR_BUILD];
    threadgroup float4 pos_sh[BLOCK_NBOR_BUILD];

    int icell_begin = cell_counts[icell];
    int icell_end = cell_counts[icell+1];

    int nborz0 = iz-cells_in_cutoff, nborz1 = iz+cells_in_cutoff,
        nbory0 = iy-cells_in_cutoff, nbory1 = iy+cells_in_cutoff,
        nborx0 = ix-cells_in_cutoff, nborx1 = ix+cells_in_cutoff;

    float4 diff;
    float r2;
    int cap = (int)ceil((float)(icell_end - icell_begin)/(float)bsx);
    for (int ii = 0; ii < cap; ii++) {
        int i = icell_begin + tid + ii*bsx;
        int pid_i = nall, stride;
        float4 atom_i, atom_j;
        int cnt = 0;
        device int *neigh_counts;
        device int *neigh_list;

        if (i < icell_end)
            pid_i = cell_particle_id[i];

        if (pid_i < nt)
            atom_i = x_[pid_i];

        if (pid_i < inum) {
            stride=inum;
            neigh_counts=nbor_list+stride+pid_i;
            neigh_list=neigh_counts+stride+pid_i*(t_per_atom-1);
            stride=stride*t_per_atom-t_per_atom;
            nbor_list[pid_i]=pid_i;
        } else {
            stride=0;
            neigh_counts=host_numj+pid_i-inum;
            neigh_list=host_nbor_list+(pid_i-inum)*_max_nbors;
        }

        for (int nborz = nborz0; nborz <= nborz1; nborz++) {
          for (int nbory = nbory0; nbory <= nbory1; nbory++) {
            for (int nborx = nborx0; nborx <= nborx1; nborx++) {
              int jcell = nborx + nbory*ncellx + nborz*ncellx*ncelly;
              int jcell_begin = cell_counts[jcell];
              int jcell_end = cell_counts[jcell+1];
              int num_atom_cell = jcell_end - jcell_begin;
              int num_iter = (int)ceil((float)num_atom_cell/(float)bsx);

              for (int k = 0; k < num_iter; k++) {
                int end_idx = min(bsx, num_atom_cell-k*bsx);
                if (tid < end_idx) {
                    int pid_j = cell_particle_id[tid+k*bsx+jcell_begin];
                    cell_list_sh[tid] = pid_j;
                    atom_j = x_[pid_j];
                    pos_sh[tid].x = atom_j.x;
                    pos_sh[tid].y = atom_j.y;
                    pos_sh[tid].z = atom_j.z;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (pid_i < nt) {
                    for (int j = 0; j < end_idx; j++) {
                        int pid_j = cell_list_sh[j];
                        diff.x = atom_i.x - pos_sh[j].x;
                        diff.y = atom_i.y - pos_sh[j].y;
                        diff.z = atom_i.z - pos_sh[j].z;
                        r2 = diff.x*diff.x + diff.y*diff.y + diff.z*diff.z;
                        if (r2 < cutoff_neigh*cutoff_neigh && pid_j != pid_i) {
                            cnt++;
                            if (cnt <= _max_nbors) {
                                *neigh_list = pid_j;
                                neigh_list++;
                                if ((cnt & (t_per_atom-1))==0)
                                    neigh_list=neigh_list+stride;
                            }
                        }
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
              } // for k
            }
          }
        }
        if (pid_i < nt)
            *neigh_counts = cnt;
    } // for ii
}

// The remaining GPU-neighbor kernels are only used for the full-GPU-binning path
// (calc_cell_id/kernel_calc_cell_counts) or bonded systems with special bonds
// (transpose/kernel_special). Neither is exercised by the Metal hybrid path yet,
// but they must exist as valid functions for set_function(). TODO: implement
// kernel_special for correct handling of molecular (special-bond) systems.
kernel void calc_cell_id() {}
kernel void kernel_calc_cell_counts() {}
kernel void transpose() {}
kernel void kernel_special() {}
"""

with open("atom_cubin.h", "w") as f:
    f.write('const char *atom = R"METAL(' + atom_metal + ')METAL";\n')

with open("neighbor_cpu_cubin.h", "w") as f:
    f.write('const char *neighbor_cpu = R"METAL(' + neighbor_cpu_metal + ')METAL";\n')

with open("neighbor_gpu_cubin.h", "w") as f:
    f.write('const char *neighbor_gpu = R"METAL(' + neighbor_gpu_metal + ')METAL";\n')

# device_cubin.h, lj_cubin.h and reaxff_cubin.h wrap the standalone .metal shader
# files (rather than an inline string) into the `const char *` symbol their .cpp
# expects. reaxff.metal only exists in the ReaxFF-enabled tree, so skip it if absent.
_metal_dir = os.path.dirname(os.path.abspath(__file__))
for _var, _src in (("device", "device.metal"), ("lj", "lj.metal"), ("reaxff", "reaxff.metal")):
    _path = os.path.join(_metal_dir, _src)
    if not os.path.exists(_path):
        continue
    with open(_path) as f:
        _body = f.read()
    with open(os.path.join(_metal_dir, _var + "_cubin.h"), "w") as f:
        f.write('const char *%s = R"METAL(%s)METAL";\n' % (_var, _body))

