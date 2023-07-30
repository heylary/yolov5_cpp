#include <iostream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"
#include "visualizer.h"
#include "lime.h"

void processFrame(bool isImage, const cv::Mat& inputImage, YOLODetector& detector, cv::VideoCapture& cap, int& frameCount, const std::vector<std::string>& classNames, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx, const std::string& windowName) {
    for (;;) {
        cv::Mat frame;
        std::vector<Detection> result;
        if (isImage) {
            frame = inputImage.clone();
        } else {
            std::unique_lock<std::mutex> lock(mtx);
            cap >> frame;
            frameCount++;
        }
        if (frame.empty()) {
            break;
        }
        if (!isImage) {
            float fps = calculateFPS(frameCount, start);
            if (fps > 0) {
<<<<<<< HEAD
                printFps(fps, frame);
=======
                printFps(fps, frame, windowName + " ");
>>>>>>> f4ee2578a4644196cb5f22aff8b7cda306351420
            }
        }
        if(windowName == "before") {
            result = detectFrame(detector, frame);
            if (showAndWait(windowName, frame, isImage)) {
                break;
            }
        } else if (windowName == "lime") {
            feature::lime limeProcessor(frame);
            cv::Mat enhancedFrame = limeProcessor.lime_enhance(frame);
            result = detectFrame(detector, enhancedFrame);
            visualizeFrame(enhancedFrame, result, classNames);
            if (showAndWait(windowName, enhancedFrame, isImage)) {
                break;
            }
        } else{
            result = detectFrame(detector, frame);
            visualizeFrame(frame, result, classNames);
            if (showAndWait(windowName, frame, isImage)) {
                break;
            }
        }
        if (isImage && frameCount > 1) {
            break;
        }
    }
}

int main(int argc, char* argv[]) {
    cv::namedWindow("before");
    cv::namedWindow("detect");
    cv::namedWindow("lime");

    cmdline::parser cmd;
    cmd.add<std::string>("model_path", 'm', "Path to onnx model.", true, "yolov5.onnx");
    cmd.add<std::string>("class_names", 'c', "Path to class names file.", true, "coco.names");
    cmd.add<std::string>("input", 'i', "Path to the input image or video file.", true, "");
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string modelPath = cmd.get<std::string>("model_path");
    const std::string inputPath = cmd.get<std::string>("input");

    if (classNames.empty()) {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    std::vector<Detection> result;
    cv::Mat inputImage;
    try {
        YOLODetector detector(modelPath, isGPU, cv::Size(640, 640));
        cv::VideoCapture cap = captureVideo(inputPath, inputImage); // 0代表默认摄像头
        // 创建高精度计时器
        auto start = std::chrono::high_resolution_clock::now();
        // 纪录视频帧数
        int frameCount = 0;
        std::mutex cap_mtx;
        bool isImage = isImageFile(inputPath);

        std::thread beforeThread(processFrame, isImage, std::ref(inputImage), std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx), "before");
        std::thread detectThread(processFrame, isImage, std::ref(inputImage), std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx), "detect");
        std::thread limeThread(processFrame, isImage, std::ref(inputImage), std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx), "lime");

        beforeThread.join();
        detectThread.join();
        limeThread.join();

        cap.release(); // 释放资源
        std::cout << "Total frames: " << frameCount << "\n";
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}
