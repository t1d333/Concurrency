cmake_minimum_required(VERSION 3.12)


find_package(OpenMP)
add_library(Solver STATIC solver.cpp)
if(OpenMP_CXX_FOUND)
	target_link_libraries(Solver PUBLIC OpenMP::OpenMP_CXX)
endif()

