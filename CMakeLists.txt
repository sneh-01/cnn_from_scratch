cmake_minimum_required(VERSION 3.0)
project(OpenCVExample)

# Set OpenCV_DIR if not set in the environment or command line
set(OpenCV_DIR "C:/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories from OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})

# Create an executable
add_executable(OpenCVExample functions.cpp)

# Link OpenCV libraries
target_link_libraries(OpenCVExample ${OpenCV_LIBS})
