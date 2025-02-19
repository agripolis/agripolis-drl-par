cmake_minimum_required(VERSION 3.26.0) #31.0)

file(GLOB cppfiles *.cpp)
file(GLOB ccfiles ../pb/*.cc)

find_package(protobuf REQUIRED CONFIG)

add_executable(agp24  ${cppfiles} ${ccfiles})

target_include_directories(agp24 PUBLIC ../pb PUBLIC /usr/local/include)

set(CUSTOM_LIBRARY_PATH ${PROJECT_SOURCE_DIR}/libs)

find_library(libprotoc NAMES protobuf)
#	  HINTS "${CUSTOM_LIBRARY_PATH}")

find_library(libzmq NAMES zmq)

find_library(glpk 
	if (MSVC) 
	   NAMES glpk_4_45_x64   #.lib 
        else()
	   NAMES glpk  #libglpk.so
	endif()
   HINTS "${CUSTOM_LIBRARY_PATH}/glpk4.45")

target_link_libraries(agp24 ${glpk} ${libprotoc} ${libzmq} )
#target_compile_options(agp24 -O2)
