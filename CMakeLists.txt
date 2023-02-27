cmake_minimum_required(VERSION 3.12)

message(STATUS "Start of top-level CMakeLists.txt")

project(picograd
    VERSION 1.0.0
    DESCRIPTION "A header-only C++ template class for automatic differentiation."
    LANGUAGES CXX
#    HOMEPAGE "https://github.com/schuderer/picograd"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# will provide us with a set of variables containing installation directories for various artifacts
include(GNUInstallDirs)

#set(TRYOUT_TARGET picograd_tryout)

add_library(${PROJECT_NAME} INTERFACE)  # because of header-only library, otherwise STATIC/SHARED (I think)
#add_executable(${TRYOUT_TARGET} main.cpp)

# Specify header locations
# From https://medium.com/swlh/c-project-structure-for-cmake-67d60135f6f5 :
#target_include_directories(${TRYOUT_TARGET} PRIVATE
#    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}>
#    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}>
#)
target_include_directories(${PROJECT_NAME} INTERFACE  # would be PUBLIC for library with cpp files under src
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Compile options and linker flags
# target_compile_options(${PROJECT_NAME} INTERFACE ...)
#target_link_libraries(${TRYOUT_TARGET} ${PROJECT_NAME})

## Installation destination
#install(TARGETS ${TRYOUT_TARGET}
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#)

# Add a subdirectory for the library source files
add_subdirectory(src)  # empty because of header-only library

# If top-level project:
# - enable verbose logging if set to ON
# - build unit tests
option(VERBOSE "Turn on verbose logging for library builds" ON)
message(STATUS "picograd CMAKE_PROJECT_NAME: " ${CMAKE_PROJECT_NAME})
message(STATUS "picograd PROJECT_NAME: " ${PROJECT_NAME})

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    if(VERBOSE)
        add_definitions(-DAJS_VERBOSE)
    endif()
    include(CTest)
    add_subdirectory(test)
endif()




## FROM HERE ON: FOR DISTRIBUTION
## (I don't yet understand things sufficiently to make proper sense of this)
## Might clarify things a bit: https://dominikberner.ch/cmake-interface-lib/

## Install the library header files
#install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/${PROJECT_NAME}
#        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
#)

## Configure the library to be exported to the build tree and the install tree
#install(TARGETS ${PROJECT_NAME}
#    EXPORT ${PROJECT_NAME}_targets
#    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#)

## The write_basic_package_version_file() function generates a version file for the library,
## which is used by downstream projects to ensure compatibility.
#include(CMakePackageConfigHelpers)
#write_basic_package_version_file(
#  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
#  VERSION ${PROJECT_VERSION}
#  COMPATIBILITY SameMajorVersion
#)

## The export() command generates a CMake package file called picograd_targets.cmake,
## which exports the picograd library for use in other CMake projects.
#export(TARGETS ${PROJECT_NAME}
#       NAMESPACE ajs::
#       FILE ${PROJECT_NAME}_targets.cmake
#)

## The install(EXPORT ...) command installs this file along with the header files and the library itself.
#install(EXPORT ${PROJECT_NAME}_targets
#        NAMESPACE ajs::
#        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
#)

#install(FILES
#  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
#  ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
#  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
#)
### End of block for distribution


# Example for static library:
#cmake_minimum_required(VERSION 3.14)

#project(library_template_static LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_library(library_template_static STATIC
#  library_template_static.cpp
#  library_template_static.h
#)

#target_compile_definitions(library_template_static PRIVATE LIBRARY_TEMPLATE_STATIC_LIBRARY)


# Example for dynamic/shared library:
#cmake_minimum_required(VERSION 3.14)

#project(library_template_dynamic LANGUAGES CXX)

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_library(library_template_dynamic SHARED
#  library_template_dynamic_global.h
#  library_template_dynamic.cpp
#  library_template_dynamic.h
#)

#target_compile_definitions(library_template_dynamic PRIVATE LIBRARY_TEMPLATE_DYNAMIC_LIBRARY)
