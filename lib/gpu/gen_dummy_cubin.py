import glob
import os

for f in glob.glob('/Users/akvaithi/Downloads/test/lammps/lib/gpu/*.cu'):
    base = os.path.basename(f)
    if base.startswith('lal_'):
        name = base[4:-3]
        cubin_name = f"/Users/akvaithi/Downloads/test/lammps/lib/gpu/{name}_cubin.h"
        with open(cubin_name, 'w') as out:
            out.write(f"const unsigned char {name}[] = {{0}};\n")
            out.write(f"const unsigned int {name}_size = 0;\n")
