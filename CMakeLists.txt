cmake_minimum_required(VERSION 3.24 FATAL_ERROR) # for madrona

include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")

project(Madrona3DExample LANGUAGES C CXX)

include(setup)
include(dependencies)

add_subdirectory(external)
enable_testing()  # Enable testing capabilities

add_subdirectory(src)
add_subdirectory(tests)  # Add your tests directory
