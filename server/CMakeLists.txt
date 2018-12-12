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

set (TARGET_NAME "smart_classroom_demo")

set(GFLAGS_IS_SUBPROJECT TRUE)
add_subdirectory(gflags)

find_package(InferenceEngine 1.4 REQUIRED)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
add_definitions(-DUSE_OPENCV)

set(sources include/cnn.hpp
            include/action_detector.hpp
            include/detector.hpp
            include/face_reid.hpp
            include/tracker.hpp
            include/image_grabber.hpp
            include/logger.hpp
            src/cnn.cpp
            src/action_detector.cpp
            src/detector.cpp
            src/tracker.cpp
            src/reid_gallery.cpp
            src/logger.cpp
            src/image_grabber.cpp
            src/align_transform.cpp
            main.cpp)

add_executable(${TARGET_NAME} ${sources})
add_dependencies(${TARGET_NAME} gflags)
set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
  COMPILE_PDB_NAME ${TARGET_NAME})

target_include_directories(${TARGET_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(${TARGET_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${InferenceEngine_INCLUDE_DIRS})
target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} ${OpenCV_LIBS} gflags
    ${InferenceEngine_INCLUDE_DIRS}/../external/omp/lib/libiomp5.so)
if(UNIX)
  target_link_libraries(${TARGET_NAME} dl pthread)
endif()

##########################################################

set (SERVER_NAME "server")
add_executable(${SERVER_NAME}
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
