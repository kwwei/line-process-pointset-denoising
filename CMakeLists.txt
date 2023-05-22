cmake_minimum_required(VERSION 3.1)
project(Denoising)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpic -fopenmp -ffast-math")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Boost REQUIRED COMPONENTS filesystem)
if(Boost_FOUND)
  message("${Boost_LIBRARIES}")
  include_directories(${Boost_INCLUDE_DIRS})
endif()

find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

include_directories(external/amgcl)
include_directories(external/CGAL)
include_directories(external/eigen3)
include_directories(external/eigen3/Eigen)

# libigl
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)
option(LIBIGL_IMGUI                  "Use IMGUI"          ON)
option(LIBIGL_GLFW                   "Use GLFW"           ON)
add_subdirectory(libigl)

add_subdirectory(src)
add_subdirectory(examples)


