cmake_minimum_required(VERSION 3.5)
PROJECT(mesh-hashing)
cmake_policy(SET CMP0015 NEW)

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
FIND_PACKAGE(OpenCV REQUIRED)
# Parallel computation
FIND_PACKAGE(CUDA REQUIRED)

# Log utilities
FIND_PACKAGE(Glog REQUIRED)

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
SET(VH ${PROJECT_SOURCE_DIR}/src)
SET(GL_WRAPPER ${VH}/extern/opengl-wrapper)

INCLUDE_DIRECTORIES(
        ${VH}
        ${VH}/extern/cuda
        ${GL_WRAPPER}/include
        ${GLOG_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_INCLUDE})

#----------
## Building Libraries
### 1. OpenGL util
ADD_LIBRARY(gl-util
        ${GL_WRAPPER}/src/args.cc
        ${GL_WRAPPER}/src/program.cc
        ${GL_WRAPPER}/src/window.cc
        ${GL_WRAPPER}/src//uniforms.cc
        ${GL_WRAPPER}/src/camera.cc)
SET_TARGET_PROPERTIES(gl-util PROPERTIES COMPILE_DEFINITIONS USE_CUDA)
TARGET_LINK_LIBRARIES(gl-util
        ${OPENGL_DEPENDENCIES}
        ${GLOG_LIBRARIES})

### 2. CUDA
# Don't know exactly how it should be configured
LIST(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_50,code=sm_60)
CUDA_ADD_LIBRARY(mesh-hashing-cuda
        ${VH}/core/hash_table.cu
        ${VH}/core/entry_array.cu
        ${VH}/core/block_array.cu
        ${VH}/core/mesh.cu
        ${VH}/core/collect.cu

        ${VH}/sensor/rgbd_sensor.cu

        ${VH}/mapping/recycle.cu
        ${VH}/mapping/fusion.cu

        ${VH}/meshing/marching_cubes.cu

        #${VH}/src/map_plane.cu
        ${VH}/util/map_debug.cu

        ${VH}/visualization/compact_mesh.cu
        ${VH}/visualization/compress_mesh.cu
        ${VH}/visualization/bounding_box.cu
        ${VH}/visualization/ray_caster.cu)
TARGET_LINK_LIBRARIES(mesh-hashing-cuda
        ${CUDA_DEPENDENCIES}
        -lopencv_core -lopencv_highgui -lopencv_imgproc
        ${GLOG_LIBRARIES})

### 3. C++
ADD_LIBRARY(mesh-hashing
        ${VH}/engine/mapping_engine.cc
        ${VH}/engine/visualizing_engine.cc
        ${VH}/io/config_manager.cc
        ${VH}/sensor/rgbd_local_sequence.cc)
        #${VH}/tool/cpp/debugger.cc)
TARGET_LINK_LIBRARIES(mesh-hashing
        mesh-hashing-cuda
        gl-util)

#----------
## Unit tests
### CUDA - OpenGL intergration
ADD_EXECUTABLE(cuda_gl_interp src/util/cuda_gl_interp.cc)
TARGET_LINK_LIBRARIES(cuda_gl_interp
        mesh-hashing
        -lopencv_core -lopencv_highgui
        ${GLOG_LIBRARIES})

ADD_EXECUTABLE(block_visualizer src/util/block_visualizer.cc
        src/util/timer.h
        src/util/video_clip.cc
        src/core/block_array.h src/core/hash_entry.h src/core/entry_array.cu src/core/entry_array.h src/core/vertex.h src/core/triangle.h)
TARGET_LINK_LIBRARIES(block_visualizer
        mesh-hashing
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES})

ADD_EXECUTABLE(video_clip src/util/video_clip.cc)
TARGET_LINK_LIBRARIES(video_clip
        -lopencv_core -lopencv_highgui)

### Loop over
MESSAGE(STATUS ${GLOG_LIBRARIES})
ADD_EXECUTABLE(mesh_hashing
        src/app/mesh_hashing.cc)
TARGET_LINK_LIBRARIES(mesh_hashing
        mesh-hashing
        -lopencv_core -lopencv_highgui
        ${GLOG_LIBRARIES})

### An ORB app
OPTION(WITH_ORBSLAM2 "Build with orb slam" ON)
if (WITH_ORBSLAM2)
    SET(ORB_SLAM2_PATH ../../opensource/orb_slam2)
    MESSAGE(STATUS "Build with ORB-SLAM2")

    FIND_PACKAGE(Pangolin REQUIRED)
    FIND_PACKAGE(Eigen3 REQUIRED)
    INCLUDE_DIRECTORIES(
            ${ORB_SLAM2_PATH}
            ${ORB_SLAM2_PATH}/include
            ${EIGEN3_INCLUDE_DIR})
    LINK_DIRECTORIES(${ORB_SLAM2_PATH}/lib)
    ADD_EXECUTABLE(rgbd_orb src/app/rgbd_orb.cc)
    TARGET_LINK_LIBRARIES(rgbd_orb
            mesh-hashing
            ORB_SLAM2
            ${Pangolin_LIBRARIES}
            -lopencv_core -lopencv_highgui
            ${GLOG_LIBRARIES})
endif(WITH_ORBSLAM2)
