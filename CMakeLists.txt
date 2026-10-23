cmake_minimum_required(VERSION 3.10)
project(light_detector LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV 4.5.4 EXACT REQUIRED COMPONENTS core imgproc highgui videoio)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV config: ${OpenCV_DIR}")

add_executable(light_detector src/main.cpp)
target_include_directories(light_detector PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(light_detector PRIVATE ${OpenCV_LIBS})
