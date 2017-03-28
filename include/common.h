//
// Created by wei on 17-3-12.
//
/// Header for .h that are both included by .cc and .cu
#ifndef VH_COMMON_H
#define VH_COMMON_H

/// Type redefinitions
#ifndef sint
typedef signed int sint;
#endif

#ifndef uint
typedef unsigned int uint;
#endif

#ifndef slong
typedef signed long slong;
#endif

#ifndef ulong
typedef unsigned long ulong;
#endif

#ifndef uchar
typedef unsigned char uchar;
#endif

#ifndef schar
typedef signed char schar;
#endif

/// Useful macros
#if defined(__CUDACC__)
#define __ALIGN__(n)  __align__(n)
#else
#define __ALIGN__(n) __attribute__((aligned(n)))
#include <cuda_runtime.h>
#endif


#endif //_VH_COMMON_H_
