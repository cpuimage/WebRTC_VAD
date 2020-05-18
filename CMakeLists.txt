cmake_minimum_required(VERSION 2.4)
project(vad)
if (MSVC)
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ")
    else ()
        set(CMAKE_BUILD_TYPE "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  /fp:fast /Gy /Oi /Oy /O2 /Ot /Zi /EHsc ")
    endif ()
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
else ()
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -g -Wall -Wno-unused-variable")
    else (CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(CMAKE_BUILD_TYPE "Release")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    endif (CMAKE_BUILD_TYPE STREQUAL "Debug")
endif ()

file(GLOB VAD_SRC
        ${CMAKE_CURRENT_LIST_DIR}/vad/*.c
        ${CMAKE_CURRENT_LIST_DIR}/vad/*.cc
        )
set(VADCOMPILE_CODE ${VAD_SRC})

add_executable(vad main.cc ${VADCOMPILE_CODE})