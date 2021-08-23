cmake_minimum_required (VERSION 3.8)

project ("AMGX_Solver")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)

add_executable (AMGX_Solver "src/AMGX_main.cpp" "src/AMGX_helper.h" "src/AMGX_helper.cpp")

find_package(CUDAToolkit REQUIRED)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cusparse)

target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgx.lib")
target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgxsh.lib")

target_include_directories(${PROJECT_NAME} PRIVATE ${AMGX_INCLUDE_CORE_DIR} ${AMGX_INCLUDE_BASE_DIR})