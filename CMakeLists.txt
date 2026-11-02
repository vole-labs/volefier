cmake_minimum_required (VERSION 3.21)
project (volefier VERSION 0.1.0 LANGUAGES CXX)
set(NAME "volefier")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(VOLEFIER_BUILD_TESTS "Build volefier tests"            ${PROJECT_IS_TOP_LEVEL})
option(VOLEFIER_INSTALL     "Generate install / export rules" ${PROJECT_IS_TOP_LEVEL})

# emp toolkit (>= 1.0, modern CMake): find_package brings the exported target emp-zk::emp-zk, which
# transitively pulls emp-ot::emp-ot, emp-tool::emp-tool, OpenSSL, Threads, C++20 and the AES-NI/AVX
# compile flags via its INTERFACE properties -- so linking emp-zk::emp-zk is all that's needed.
# A parent project that already built emp-zk in-tree (add_subdirectory) provides the target itself.
if(NOT TARGET emp-zk::emp-zk)
  find_package(emp-zk 1.0 REQUIRED)
endif()

# Header-only library: volefier/... included as "volefier/<module>/<header>.h".
add_library(volefier INTERFACE)
add_library(volefier::volefier ALIAS volefier)
target_compile_features(volefier INTERFACE cxx_std_20)
target_include_directories(volefier INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(volefier INTERFACE emp-zk::emp-zk)

if(VOLEFIER_INSTALL)
  set(_cmake_install_dir ${CMAKE_INSTALL_LIBDIR}/cmake/volefier)
  install(TARGETS volefier EXPORT volefierTargets)
  install(DIRECTORY volefier
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
  install(EXPORT volefierTargets
    NAMESPACE   volefier::
    DESTINATION ${_cmake_install_dir}
    FILE        volefierTargets.cmake)
  configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/cmake/volefier-config.cmake.in
    ${PROJECT_BINARY_DIR}/volefier-config.cmake
    INSTALL_DESTINATION ${_cmake_install_dir})
  write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/volefier-config-version.cmake
    VERSION       ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
  install(FILES
    ${PROJECT_BINARY_DIR}/volefier-config.cmake
    ${PROJECT_BINARY_DIR}/volefier-config-version.cmake
    DESTINATION ${_cmake_install_dir})

  # Build-tree export so siblings can consume without `make install` (volefier_DIR=<build>).
  export(EXPORT volefierTargets
    NAMESPACE volefier::
    FILE      ${PROJECT_BINARY_DIR}/volefierTargets.cmake)
endif()

if(VOLEFIER_BUILD_TESTS)
  # Put test binaries in build/bin/ (the legacy emp-base.cmake set this; the modern emp does not, so
  # without it binaries land in build/test/ and the `run` scripts / setup.sh smoke test miss them).
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(test)
endif()
