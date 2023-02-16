cmake_minimum_required(VERSION 3.5)

project(picograd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#add_library(picograd picograd.h picograd.cpp)
add_executable(picograd_tryout main.cpp)
#target_link_libraries(picograd_tryout picograd)

install(TARGETS picograd_tryout
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})


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
