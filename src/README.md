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