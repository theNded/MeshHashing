# Linux Version of Voxel Hashing

## Rules for CUDA - C++ integration

### Function
0. In general, functions outside classes should only be
 related to kernel calls
1. *\_\_global__* functions and the necessary *\_\_device__*
  helpers should stay in .cu
2. The caller of *\_\_global__* functions, 
  i.e., *\_\_host__* functions should be 
  declared in .h and defined in .cu

### Class
0. In general, a class should only contain 
  *\_\_device__* and *\_\_host__* functions.
  Their implementations should stay in one .h to avoid being 
  written twice in .cc and .cu
1. If a class calls any kernels, call a non-class function
  refer to **Function** part
2. If a .h is not included by any other .cu, write an empty
  .cu to include it, guarantee that it is compiled by nvcc
4. When refer to external symbols, ensure that 
  'separate compilation' is enabled. The .cu with external 
  symbols **MUST BE** compiled along with the file including
  the origin symbol.