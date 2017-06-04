//
// Created by wei on 17-3-12.
//
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

/// Enable linked list in the hash table
#define HANDLE_COLLISIONS

/// Block size in voxel unit
#define BLOCK_SIDE_LENGTH  8
#define BLOCK_SIZE         512 // 8x8x8
#define MEMORY_LIMIT       100

/// Entry state
#define LOCK_ENTRY -1
#define FREE_ENTRY -2
#define NO_OFFSET   0

#endif //_VH_COMMON_H_
