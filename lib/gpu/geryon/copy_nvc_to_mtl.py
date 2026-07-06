import glob
import os

# First, recreate the mtl_* files from nvd_* files
files = glob.glob('/Users/akvaithi/Downloads/test/lammps/lib/gpu/geryon/nvd_*.h')
for f in files:
    base = os.path.basename(f)
    new_base = base.replace('nvd_', 'mtl_')
    new_path = os.path.join(os.path.dirname(f), new_base)
    
    with open(f, 'r') as fp:
        content = fp.read()
    
    content = content.replace('nvd_', 'mtl_')
    content = content.replace('NVD_', 'MTL_')
    content = content.replace('ucl_cudadr', 'ucl_metal')
    content = content.replace('#include <cuda.h>', '//#include <cuda.h>')
    
    # Prepend our stubs include
    content = '#include "mtl_cuda_stubs.h"\n' + content
    
    with open(new_path, 'w') as fp:
        fp.write(content)
    print(f"Created {new_path}")
