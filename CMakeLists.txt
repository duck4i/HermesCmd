cmake_minimum_required(VERSION 3.14)  # Specify the required CMake version
project(HermesCLI)                      # Set the project name

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add subdirectory for ggml
add_subdirectory(ggml)
add_subdirectory(llama/src)

# Add executable target for main.cpp
add_executable(hermes main.cpp)

# Link libraries
target_link_libraries(hermes ggml llama)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)