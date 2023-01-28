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
include_directories("D:/SDK_PATH/itpp-4.3.1/include")
set(itpp_lib D:/SDK_PATH/itpp-4.3.1/lib/itpp.lib)
file(GLOB itpp_dll D:/SDK_PATH/itpp-4.3.1/bin/itpp.dll)

add_executable(opencv2 main.cpp)
add_executable(eigen_demo eigen_demo.cpp)
add_executable(itpp_demo itpp_demo.cpp)
TARGET_LINK_LIBRARIES(opencv2 ${OpenCV_LIBS} ${itpp_lib})
TARGET_LINK_LIBRARIES(eigen_demo ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(itpp_demo ${OpenCV_LIBS} ${itpp_lib})

install(TARGETS opencv2
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} )

file(COPY ${itpp_dll} DESTINATION ${PROJECT_BINARY_DIR})
