cmake_minimum_required(VERSION 3.6)
PROJECT(voxel-hashing)

#----------
# C++ version setting
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -Wall")

#----------
# Project variable configurations
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

#----------
# Find packages
FIND_PACKAGE(PkgConfig)
LIST(APPEND
        CMAKE_MODULE_PATH
        ${PROJECT_SOURCE_DIR}/cmake_modules)

# Graphics dependent
FIND_PACKAGE(GLFW3  REQUIRED)
FIND_PACKAGE(GLEW   REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
# CV utilities
# Linear Algebra, optional
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
# Parallel computation
FIND_PACKAGE(CUDA REQUIRED)

# Log utilities
FIND_PACKAGE(glog REQUIRED)

#----------
## Found Headers
GET_PROPERTY(included_dirs
        DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        PROPERTY INCLUDE_DIRECTORIES)
MESSAGE(STATUS "Found include paths:")
foreach(included_dir ${included_dirs})
    MESSAGE(STATUS ${included_dir})
endforeach()

#----------
## Found Libs
SET(OPENGL_DEPENDENCIES "")
LIST(APPEND
        OPENGL_DEPENDENCIES
        ${OPENGL_LIBRARY}
        ${GLEW_LIBRARY})
if (APPLE)
    LIST(APPEND OPENGL_DEPENDENCIES
            ${GLFW3_LIBRARIES})
else ()
    LIST(APPEND OPENGL_DEPENDENCIES
            ${GLFW3_STATIC_LIBRARIES})
endif()

SET(CUDA_DEPENDENCIES "")
LIST(APPEND
        CUDA_DEPENDENCIES
        ${CUDA_CUDART_LIBRARY}
        ${CUDA_CUDA_LIBRARY})

SET(OTHER_DEPENDENCIES ${OpenCV_LIBRARIES})

MESSAGE(STATUS "Found libraries:")
foreach(library ${OPENGL_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
foreach(library ${CUDA_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
foreach(library ${OTHER_DEPENDENCIES})
    MESSAGE(STATUS ${library})
endforeach()
#---------

#----------
SET(VH ${PROJECT_SOURCE_DIR})
MESSAGE(STATUS ${CUDA_TOOLKIT_INCLUDE})

INCLUDE_DIRECTORIES(
        test/include
        ${VH}/include
        ${VH}/gl_utils
        ${VH}/cuda_utils
        ${MRF}/include
        ${CUDA_TOOLKIT_INCLUDE})

#----------
## Building Libraries
### 1. OpenGL utils
ADD_LIBRARY(GL_UTILS
        ${VH}/gl_utils/context.cc
        ${VH}/gl_utils/control.cc
        ${VH}/gl_utils/shader.cc)
TARGET_LINK_LIBRARIES(GL_UTILS glog ${OPENGL_DEPENDENCIES})

### 2. VoxelHashing CUDA
SET(CUDA_SEPARABLE_COMPILATION ON)
# sm MUST >= 50, otherwise separable compilation will be disabled...
# compute_35, sm_50 is ok
LIST(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_52)
CUDA_ADD_LIBRARY(CUDAVoxelHashing
        ${VH}/src/constant.cu
        ${VH}/src/sensor.cu
        ${VH}/src/mapper.cu
        ${VH}/src/ray_caster.cu)

### 3. VoxelHashing C++
ADD_LIBRARY(VoxelHashing
        ${VH}/src/renderer.cc
        ${VH}/src/sensor.cc
        ${VH}/src/mapper.cc
        ${VH}/src/ray_caster.cc)
TARGET_LINK_LIBRARIES(VoxelHashing GL_UTILS)

#----------
## Unit tests
### 1. Core (SDF, HashEntry)
CUDA_ADD_LIBRARY(cu_test_core test/test_core/test_core.cu)
ADD_EXECUTABLE(test_core test/test_core/test_core.cc)
TARGET_LINK_LIBRARIES(test_core cu_test_core glog)

### 2. HashTable (Hash)
### shared variables should compile together by nvcc
CUDA_ADD_LIBRARY(cu_test_hash_table
        test/test_hash_table/test_hash_table.cu
        ${VH}/src/constant.cu)
ADD_EXECUTABLE(test_hash_table test/test_hash_table/test_hash_table.cc)
TARGET_LINK_LIBRARIES(test_hash_table cu_test_hash_table VoxelHashing glog)

### 3. CUDA - OpenGL intergration
ADD_EXECUTABLE(test_cuda_gl test/test_cuda_gl.cc)
TARGET_LINK_LIBRARIES(test_cuda_gl GL_UTILS
        VoxelHashing
        glog ${CUDA_DEPENDENCIES} ${OTHER_DEPENDENCIES})

### 4. Mapper and RayCaster
ADD_EXECUTABLE(test_voxel_hashing test/test_voxel_hashing.cc)
TARGET_LINK_LIBRARIES(test_voxel_hashing GL_UTILS
        VoxelHashing CUDAVoxelHashing
        glog ${CUDA_DEPENDENCIES} ${OTHER_DEPENDENCIES})

### 5. Loop over
ADD_EXECUTABLE(test_voxel_hashing_loop test/test_voxel_hashing_loop.cc)
TARGET_LINK_LIBRARIES(test_voxel_hashing_loop GL_UTILS
        VoxelHashing CUDAVoxelHashing
        glog ${CUDA_DEPENDENCIES} ${OTHER_DEPENDENCIES})

