cmake_minimum_required(VERSION 3.16)

project(opencv2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#set(VCPKG_TARGET_TRIPLET "x64-windows" CACHE STRING "" FORCE)
#set(CMAKE_TOOLCHAIN_FILE D:/sourcecode/vcpkg/scripts/buildsystems/vcpkg.cmake)

# add opencv dependence
set(OpenCV_DIR D:\\SDK_PATH\\OpenCV\\build)
set(eigen D:\\SDK_PATH\\eigen-3.4.0)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${eigen})

message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

# import itpp
set(itpp_path D:/SDK_PATH/itpp_4.3.1)
include_directories("${itpp_path}/include")
set(itpp_lib ${itpp_path}/lib/itpp.lib)
file(GLOB itpp_dll ${itpp_path}/bin/*.dll)

## import fftw
#set(fftw_path D:/SDK_PATH/fftw-3.3.10)
#set(FFT_INCLUDES ${fftw_path}/fftw3.h)
#set(FFT_LIBRARIES ${fftw_path}/fftw3.lib)
#file(GLOB fftw_dll ${fftw_path}/fftw3.dll)
#include_directories(${fftw_path})

include_directories(${CMAKE_SOURCE_DIR})

add_subdirectory(test)

add_executable(opencv2 main.cpp)

TARGET_LINK_LIBRARIES(opencv2 ${OpenCV_LIBS} ${itpp_lib})

install(TARGETS opencv2
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

# copy *.dll to exe path
file(COPY ${fftw_dll} ${itpp_dll} DESTINATION ${PROJECT_BINARY_DIR})
file(COPY ${fftw_dll} ${itpp_dll} DESTINATION ${PROJECT_BINARY_DIR}/test)

#copy dataset directory
file(GLOB dataset ${CMAKE_SOURCE_DIR}/dataset/*.*)
file(COPY ${dataset} DESTINATION ${PROJECT_BINARY_DIR}/dataset)
file(COPY ${dataset} DESTINATION ${PROJECT_BINARY_DIR}/test/dataset)
