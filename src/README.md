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
- Array should be stored by *ptr, so that it can be correctly allocated on memory either on CPU or GPU;
- Non-const elements should behave the same as array, so that when changed on GPU, they can be accesed by CPU via memcpy;
- Const elements such as `int` should be stored as they are, so that they can be directly passed from device to host, vice versa.

### engine
Higher level wrappers for the data structures in `core`, basically
- **main_engine**, managing **hash_table**, **blocks**, and **mesh**
- **visualizing_engine**, manage **compact_mesh**, **bounding_box**, and **trajectory**
- **logging_engine**, record profiles, etc

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

## TODO
- Separate .cu && .cc (functions and methods)
- Integrate statistics
- Involve time profile
- Minor argument configurations