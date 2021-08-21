cmake_minimum_required (VERSION 3.8)

project ("AMGX_Solver")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)
file(COPY "${AMGX_LIBRARIES_DIR}/amgxsh.dll" DESTINATION ${CMAKE_SOURCE_DIR}/bld)

add_executable (AMGX_Solver "AMGX_Solver/AMGX_Solver.cpp" )

target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgx.lib")
target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgxsh.lib")

target_include_directories(${PROJECT_NAME} PRIVATE ${AMGX_INCLUDE_CORE_DIR} ${AMGX_INCLUDE_BASE_DIR})
