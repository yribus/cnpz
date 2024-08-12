cmake_minimum_required(VERSION 3.28)
project(cnpz)

set(CMAKE_CXX_STANDARD 20)

find_package(ZLIB REQUIRED)

add_executable(cnpz
    src/cnpz.cpp
    src/cnpz.h
    main.cpp)

target_link_libraries(cnpz PRIVATE ZLIB::ZLIB)