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

kernel void calc_cell_id() {}
kernel void kernel_calc_cell_counts() {}
kernel void calc_neigh_list_cell() {}
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

