# yolov5-onnxruntime

C++ YOLO v5 ONNX Runtime inference code for object detection.

## Dependecies:
- OpenCV 4.x
- ONNXRuntime 1.12.0
- OS: Tested on Windows 10 and Ubuntu 20.04
- CUDA 11+ [Optional]


## Build
To build the project and run the following commands
```bash
cd yolov5_cpp
./build.sh
cd build

```

or you can

```bash
mkdir build
cd build
cmake
make
```

## Run
Before running the executable you should convert your PyTorch model to ONNX if you haven't done it yet. Check the [official tutorial](https://github.com/ultralytics/yolov5/issues/251).

`On Windows`: to run the executable you should add OpenCV and ONNX Runtime libraries to your environment path `or` put all needed libraries near the executable (onnxruntime.dll and opencv_world.dll).

Run from CLI,the --gpu(option):
```bash
./yolov5_cpp --model_path yolov5.onnx --image bus.jpg --class_names coco.names --gpu
# On Windows ./yolo_ort.exe with arguments as above
```

## Demo

YOLOv5s onnx:

<p align="center">
  <a href="images/bus_result.jpg"><img src="images/bus_result.jpg" style="width:60%; height:60%;"/></a>
</p>
<p align="center">
  <a href="images/zidane_result.jpg"><img src="images/zidane_result.jpg" style="width:60%; height:60%;"/></a>
</p>


## References

- YOLO v5 repo: https://github.com/ultralytics/yolov5
- YOLOv5 Runtime Stack repo: https://github.com/zhiqwang/yolov5-rt-stack
- ONNXRuntime Inference examples: https://github.com/microsoft/onnxruntime-inference-examples
