# 当前目录是 ~/Documents/fstr_psd/trust-region-newton/build
# python3 ../experiments_icgip26/rebuild.py

import sys, os
# export CXXFLAGS="-I/home/zjtest/Documents/cubic_fstr_psd/trust-region-newton/build/_deps/libigl-src/include"
commands = ['cd ..', 'pwd','rm -rfi build', 'mkdir build', 'cd build','cmake .. -DCMAKE_CXX_FLAGS="-I/usr/local/include"', 'make -j16']
for com in commands:
    os.system(com)

 