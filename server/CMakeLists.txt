# Copyright (c) 2018 Intel Corporation

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required (VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(InferenceEngine 1.4 REQUIRED)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs)
add_definitions(-DUSE_OPENCV)

set (SERVER_NAME "server")
add_executable(${SERVER_NAME}
        include/samples/common.hpp
        include/cnn.hpp
        include/detector.hpp
        src/cnn.cpp
        src/detector.cpp
        src/align_transform.cpp
        server.cpp)
target_include_directories(${SERVER_NAME} PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(${SERVER_NAME} PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(${SERVER_NAME}
        ${OpenCV_LIBS}
        IE::ie_cpu_extension
        ${InferenceEngine_LIBRARIES}
        ${InferenceEngine_INCLUDE_DIRS}/../external/omp/lib/libiomp5.so
        event_core event_extra)
if(UNIX)
  target_link_libraries(${SERVER_NAME} dl pthread)
endif()
