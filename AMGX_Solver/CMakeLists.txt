
cmake_minimum_required (VERSION 3.8)

add_executable (AMGX_Solver "AMGX_Solver.cpp" )

target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgx.lib")
target_link_libraries(${PROJECT_NAME} PRIVATE "${AMGX_LIBRARIES_DIR}/amgxsh.lib")

target_include_directories(${PROJECT_NAME} PRIVATE ${AMGX_INCLUDE_CORE_DIR} ${AMGX_INCLUDE_BASE_DIR})
