//
// Created by wei on 17-10-21.
//

#ifndef CORE_HASH_ENTRY_H
#define CORE_HASH_ENTRY_H

#include "core/common.h"
#include "helper_math.h"

struct __ALIGN__(8) HashEntry {
  int3	pos;		   // block position (lower left corner of SDFBlock))
  int		ptr;	     // pointer into heap to SDFBlock
  uint	offset;		 // offset for linked lists

  __device__
  void operator=(const struct HashEntry& e) {
    ((long long*)this)[0] = ((const long long*)&e)[0];
    ((long long*)this)[1] = ((const long long*)&e)[1];
    ((int*)this)[4]       = ((const int*)&e)[4];
  }

  __device__
  void Clear() {
    pos    = make_int3(0);
    ptr    = FREE_ENTRY;
    offset = 0;
  }
};

#endif //MESH_HASHING_HASH_ENTRY_H
