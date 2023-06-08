#!/bin/bash
mkdir -p build
cd build
cmake .. -DONNXRUNTIME_DIR=/home/narwal/Downloads/onnxruntime-linux-x64-1.12.1 # 将 /path/to/onnxruntime 替换为 ONNX Runtime 的安装路径
make
