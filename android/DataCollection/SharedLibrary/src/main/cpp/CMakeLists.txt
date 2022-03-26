project(NcnnLib)

cmake_minimum_required(VERSION 3.4.1)

option(NCNN_LIB_OPENMP "Enable OpenMP" OFF)
option(NCNN_LIB_VULKAN "Enable Vulkan" ON)
set(NCNN_LIB_OPENMP OFF)
if (NCNN_LIB_VULKAN)
    set(ncnn_DIR "${CMAKE_SOURCE_DIR}/../ncnn-vulkan/${ANDROID_ABI}/lib/cmake/ncnn")
else ()
    set(ncnn_DIR "${CMAKE_SOURCE_DIR}/../ncnn/${ANDROID_ABI}/lib/cmake/ncnn")
endif ()

find_package(ncnn REQUIRED)

## opencv 库
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/../../sdk/native/jni")
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
    message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "opencv Not Found!")
endif (OpenCV_FOUND)

include_directories(include)
file(GLOB NCNN_SRC src/*.cpp)
set(NCNN_COMPILE_CODE ${NCNN_SRC})

add_library(NcnnLib SHARED ${NCNN_COMPILE_CODE})

if (NCNN_LITE_OPENMP)
    target_compile_definitions(NcnnLib PRIVATE __OPENMP__)
endif ()

target_link_libraries(
        NcnnLib
        ncnn
        ${OpenCV_LIBS}
        jnigraphics)

if (NCNN_LITE_VULKAN)
    target_compile_definitions(NcnnLib PRIVATE __VULKAN__)
endif ()