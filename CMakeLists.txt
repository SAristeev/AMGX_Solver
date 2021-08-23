cmake_minimum_required (VERSION 3.8)

project ("AMGX_Solver")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)
file(COPY "${AMGX_LIBRARIES_DIR}/amgxsh.dll" DESTINATION ${CMAKE_SOURCE_DIR}/bld)

add_executable (AMGX_Solver "AMGX_Solver/AMGX_Solver.cpp" )

# Include sub-projects.
add_subdirectory ("AMGX_Solver")
