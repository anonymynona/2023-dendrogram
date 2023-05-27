cmake_minimum_required(VERSION 3.16)
project(Dendrogram CXX)

find_package(ArborX 1.4 REQUIRED)
find_package(Boost 1.56.0 REQUIRED COMPONENTS program_options)

add_executable(dendrogram.exe dendrogram.cpp)
target_link_libraries(dendrogram.exe ArborX::ArborX Boost::program_options)
