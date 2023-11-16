cmake_minimum_required(VERSION 3.12)

add_subdirectory(./solver)

find_package(OpenMP)

add_executable(lab3 main.cpp)

target_link_libraries(lab3 PUBLIC Solver)

if(OpenMP_CXX_FOUND)
    target_link_libraries(lab3 PUBLIC OpenMP::OpenMP_CXX)
endif()




