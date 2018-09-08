# A simple utility script to replace label names in the Annotation xml
# files.

import os
import glob

files = glob.glob('Annotations/*.xml')

oldkey = 'helmet'
newkey = 'obj'

for file in files:
    lines = ''
    with open(file, 'r') as f:
        lines = f.read().replace(oldkey, newkey)
    with open(file, 'w') as f:
        f.write(lines)
