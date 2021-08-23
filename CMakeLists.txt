cmake_minimum_required (VERSION 3.8)

project ("AMGX_Solver")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bld)
file(COPY "${AMGX_LIBRARIES_DIR}/amgxsh.dll" DESTINATION ${CMAKE_SOURCE_DIR}/bld)

find_package(MPI)

if (MPI_FOUND)
	message("MPI build")
	set(SRC "AMGX_Solver/mpi_test.cpp" "AMGX_Solver/mpi_test.h")
else (MPI_FOUND)
	message("Non-MPI build")
	set(SRC "AMGX_Solver/simple_test.cpp")
endif (MPI_FOUND)

add_executable (AMGX_Solver ${SRC}) 

if (MPI_FOUND)
	include_directories(SYSTEM ${MPI_INCLUDE_PATH})
	target_link_libraries(${PROJECT_NAME} PRIVATE ${MPI_C_LIBRARIES})
endif (MPI_FOUND)

find_package(CUDAToolkit REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgx.lib")
target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgxsh.lib")

target_include_directories(${PROJECT_NAME} PRIVATE ${AMGX_INCLUDE_CORE_DIR} ${AMGX_INCLUDE_BASE_DIR})
