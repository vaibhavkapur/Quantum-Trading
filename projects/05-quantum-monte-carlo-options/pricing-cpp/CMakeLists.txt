cmake_minimum_required(VERSION 3.16)
project(mc_pricer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
add_executable(mc_pricer mc_pricer.cpp)
target_link_libraries(mc_pricer PRIVATE Threads::Threads)
