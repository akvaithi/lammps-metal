import glob
import re
import os

for f in glob.glob('/Users/akvaithi/Downloads/test/lammps/lib/gpu/*.h') + glob.glob('/Users/akvaithi/Downloads/test/lammps/lib/gpu/*.cpp'):
    with open(f, 'r') as fp:
        content = fp.read()
    
    def replacer(match):
        hip_block = match.group(1)
        metal_block = hip_block.replace('USE_HIP', 'USE_METAL').replace('hip_', 'mtl_').replace('ucl_hip', 'ucl_metal')
        return hip_block + metal_block + '#else'
        
    pattern = re.compile(r'(#elif defined\(USE_HIP\).*?\n)#else', re.DOTALL)
    new_content = pattern.sub(replacer, content)
    
    if new_content != content:
        with open(f, 'w') as fp:
            fp.write(new_content)
        print(f"Updated {f}")
