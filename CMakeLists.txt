cmake_minimum_required(VERSION 3.0.0)
project(yolo_cpp)

option(ONNXRUNTIME_DIR "Path to built ONNX Runtime directory." STRING)
message(STATUS "ONNXRUNTIME_DIR: ${ONNXRUNTIME_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
#查找并加载opencv库
find_package(OpenCV 4.7.0 REQUIRED )

include_directories("include/")

add_executable(yolo_cpp
               src/add_fps.cpp
               src/detector.cpp
               src/lime.cpp
               src/visualizer.cpp
               src/utils.cpp)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(ONNXRUNTIME_DIR ~/onnxruntime-linux-x64-1.12.1)

target_include_directories(yolo_cpp PRIVATE "${ONNXRUNTIME_DIR}/include")
# link_directories("${ONNXRUNTIME_DIR}/lib")
target_compile_features(yolo_cpp PRIVATE cxx_std_14)
# 链接Opencv库
target_link_libraries(yolo_cpp ${OpenCV_LIBS} -pthread)

if (WIN32)
    target_link_libraries(yolo_cpp "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib")
endif(WIN32)

if (UNIX)
    target_link_libraries(yolo_cpp "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif(UNIX)

