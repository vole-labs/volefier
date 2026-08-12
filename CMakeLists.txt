cmake_minimum_required (VERSION 3.13)
project (volefier)
set(NAME "volefier")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Put test binaries in build/bin/ (the legacy emp-base.cmake set this; the modern emp does not, so
# without it binaries land in build/test/ and the `run` scripts / setup.sh smoke test miss them).
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# This project's own headers (volefier/...) are read from the SOURCE TREE -- they are not installed.
include_directories(${CMAKE_SOURCE_DIR})

# emp toolkit (>= 1.0, modern CMake): find_package brings the exported target emp-zk::emp-zk, which
# transitively pulls emp-ot::emp-ot, emp-tool::emp-tool, OpenSSL, Threads, C++20 and the AES-NI/AVX
# compile flags via its INTERFACE properties -- so linking emp-zk::emp-zk is all that's needed.
find_package(emp-zk REQUIRED)
set(EMP-ZK_LIBRARIES emp-zk::emp-zk)

ENABLE_TESTING()
ADD_SUBDIRECTORY(test)
