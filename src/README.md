# Project Structure

## app
Target executables

## Framework
### core
Basic data structures, including
- parameters and settings
- **hash_table**
- **block**, holding its minimum unit, **voxel**
- **mesh**, holding **vertices** and **triangles**

These structs are designed to be usable on both CPU and GPU. The protocols for a `struct` are:
- Mass data should be stored by *ptr, so that it can be correctly hold on memory either on CPU or GPU;
- Non-const units should behave the same as mass data, so that when changed on GPU, they can be accesed by CPU via memcpy;
- Const small units such as `int` should be stored as they are, so that they can be directly passed from device to host, vice versa.

```cpp
struct DSMemory {
  const int value;
  int* value;
  T* array;
}

class DS {
  DSMemory ds_;
  /// Manage memory accordingly (CPU or GPU)
  DS() { ... }
  void Resize() { ... }
}
``` 

### engine
Higher level wrappers for the data structures in `core`, basically
- **map_engine**, managing **hash_table**, **blocks**, and **mesh**
- **sensor_engine**, managing data from sensors

## Function
### fusion
Using naive/modified algorithms to fuse data into map.

### meshing
Generate **mesh** from **blocks** using modified Marching Cubes.

### geometry
- Conversion of coordinates between *world*, *block*, and *voxel*.
- Conversion of systems between *camera* and *world*
- Projections

### visualization
- **ray_caster**
- **mesh_renderer**

### io
Read and write images/data/map/mesh

## Util
Backup garbages here.