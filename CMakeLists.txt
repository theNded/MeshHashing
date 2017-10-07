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
SET(VH ${PROJECT_SOURCE_DIR})
MESSAGE(STATUS ${CUDA_TOOLKIT_INCLUDE})

INCLUDE_DIRECTORIES(
        ${VH}/include
        ${VH}/include/opengl
        ${VH}/include/cuda
        ${GLOG_INCLUDE_DIRS}
        ${CUDA_TOOLKIT_INCLUDE})

#----------
## Building Libraries
### 1. OpenGL util
ADD_LIBRARY(gl-util
        ${VH}/include/opengl/args.cc
        ${VH}/include/opengl/program.cc
        ${VH}/include/opengl/window.cc
        ${VH}/include/opengl/uniforms.cc
        ${VH}/include/opengl/camera.cc)
TARGET_LINK_LIBRARIES(gl-util
        ${OPENGL_DEPENDENCIES}
        ${GLOG_LIBRARIES})

### 2. CUDA
SET(CUDA_SEPARABLE_COMPILATION ON)
# sm MUST >= 50, otherwise separable compilation will be disabled...
# compute_35, sm_50 is ok
LIST(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_52,code=sm_61)
CUDA_ADD_LIBRARY(mesh-hashing-cuda
        ${VH}/src/constant.cu
        ${VH}/src/hash_table.cu
        ${VH}/src/block.cu
        ${VH}/src/mesh.cu
        ${VH}/src/sensor.cu
        ${VH}/src/map_core.cu
        ${VH}/src/map_fusion.cu
        ${VH}/src/map_mesh.cu
        #${VH}/src/map_plane.cu
        ${VH}/src/map_debug.cu
        ${VH}/src/ray_caster.cu)
TARGET_LINK_LIBRARIES(mesh-hashing-cuda
        ${CUDA_DEPENDENCIES}
        -lopencv_core -lopencv_highgui -lopencv_imgproc# cv::Mat in sensor
        ${GLOG_LIBRARIES})

### 3. C++
ADD_LIBRARY(mesh-hashing
        ${VH}/src/dataset_manager.cc
        ${VH}/tool/cpp/debugger.cc)
TARGET_LINK_LIBRARIES(mesh-hashing
        mesh-hashing-cuda
        gl-util)

#----------
## Unit tests
### CUDA - OpenGL intergration
ADD_EXECUTABLE(cuda_gl_interp src/unit_test/cuda_gl_interp.cc)
TARGET_LINK_LIBRARIES(cuda_gl_interp
        mesh-hashing
        -lopencv_core -lopencv_highgui
        ${GLOG_LIBRARIES})

ADD_EXECUTABLE(block_visualizer src/unit_test/block_visualizer.cc include/timer.h src/unit_test/video_clip.cc)
TARGET_LINK_LIBRARIES(block_visualizer
        mesh-hashing
        ${OpenCV_LIBRARIES}
        ${GLOG_LIBRARIES})

ADD_EXECUTABLE(video_clip src/unit_test/video_clip.cc)
TARGET_LINK_LIBRARIES(video_clip
        -lopencv_core -lopencv_highgui)

### Loop over
MESSAGE(STATUS ${GLOG_LIBRARIES})
ADD_EXECUTABLE(mesh_hashing
        src/example/mesh_hashing.cc)
TARGET_LINK_LIBRARIES(mesh_hashing
        mesh-hashing
        -lopencv_core -lopencv_highgui
        ${GLOG_LIBRARIES})

### An ORB example
OPTION(WITH_ORBSLAM2 "Build with orb slam" OFF)
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
    ADD_EXECUTABLE(rgbd_orb src/example/rgbd_orb.cc)
    TARGET_LINK_LIBRARIES(rgbd_orb
            mesh-hashing
            ORB_SLAM2
            ${Pangolin_LIBRARIES}
            -lopencv_core -lopencv_highgui
            ${GLOG_LIBRARIES})
endif(WITH_ORBSLAM2)
