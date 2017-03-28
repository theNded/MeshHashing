# - Try to find GLEW
# Once done, this will define
#
#  GLEW_FOUND - system has GLEW
#  GLEW_INCLUDE_DIR - the GLEW include directory
#  GLEW_LIBRARY - the GLEW library


# Use PkgConfig if available to get hints about glew location
find_package(PkgConfig)
if (PKG_CONFIG_FOUND)
    pkg_check_modules(PKG_GLEW QUIET glew)
endif()

# Search for library objects in known paths and optional hint from pkg-config
find_library(GLEW_LIBRARY
    NAMES GLEW
    HINTS ${PKG_GLEW_LIBRARY_DIRS} ${PKG_GLEW_LIBDIR}
)

# Search for header files in known paths and optional hint from pkg-config
find_path(GLEW_INCLUDE_DIR
    GL/glew.h
    HINTS ${PKG_GLEW_INCLUDE_DIRS} ${PKG_GLEW_INCLUDEDIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GLEW REQUIRED_VARS GLEW_LIBRARY GLEW_INCLUDE_DIR)
mark_as_advanced(GLEW_LIBRARY GLEW_INCLUDE_DIR)