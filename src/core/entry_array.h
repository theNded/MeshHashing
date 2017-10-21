//
// Created by wei on 17-10-21.
//

#ifndef MESH_HASHING_ENTRY_ARRAY_H
#define MESH_HASHING_ENTRY_ARRAY_H

#include "hash_entry.h"

class EntryArray {
private:
  HashEntry *entries;       /// allocated for parallel computation
  uint       entry_count_;

public:
  int       *candidate_entry_counter; /// atomic counter to add compacted entries atomically
  int       *entry_recycle_flags;     /// used in garbage collection

  __host__ EntryArray();
  __host__ void Alloc(uint entry_count);
  __host__ void Free();

//  ~EntryArray();

  uint entry_count();
  void reset_entry_count();

  void Resize(uint entry_count);
  void Reset();

  __host__ __device__
  HashEntry& operator [] (int i) {
    return entries[i];
  }
};

#endif //MESH_HASHING_ENTRY_ARRAY_H
