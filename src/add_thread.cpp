#include <iostream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"
#include "visualizer.h"
#include "lime.h"

void processBeforeFrame(bool isImage, const cv::Mat& inputImage, cv::VideoCapture& cap, int& frameCount, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        if(isImage) {
            frame = inputImage.clone();
        }else {
            std::unique_lock<std::mutex> lock(mtx);
            cap >> frame;
            frameCount++;
        }
        float fpsBefore = calculateFPS(frameCount, start);
        if (fpsBefore > 0) {
            printFps(fpsBefore, frame);
        }
        if (showAndWait("before", frame, isImage)) {
            break;
        }

        if (isImage && frameCount > 1) {
            break;
        }
    }
}

void processDetectFrame(bool isImage, const cv::Mat& inputImage, YOLODetector& detector, cv::VideoCapture& cap, int& frameCount, const std::vector<std::string>& classNames, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        std::vector<Detection> result;
        if(isImage) {
            frame = inputImage.clone();
        }else {
            std::unique_lock<std::mutex> lock(mtx);
            cap >> frame;
            frameCount++;
        }
        result = detectFrame(detector, frame);
        visualizeFrame(frame, result, classNames);
        float fpsDetect = calculateFPS(frameCount, start);
        if (fpsDetect > 0) {
            printFps(fpsDetect, frame);
        }
        if (showAndWait("detect", frame, isImage)) {
            break;
        }

        if (isImage && frameCount > 1) {
            break;
        }
    }
}

void processLimeFrame(bool isImage, const cv::Mat& inputImage, YOLODetector& detector, cv::VideoCapture& cap, int& frameCount, const std::vector<std::string>& classNames, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        std::vector<Detection> result;
        if(isImage) {
            frame = inputImage.clone();
        }else {
            std::unique_lock<std::mutex> lock(mtx);
            cap >> frame;
            frameCount++;
        }
        feature::lime limeProcessor(frame);
        cv::Mat enhancedFrame = limeProcessor.lime_enhance(frame);
        result = detectFrame(detector, enhancedFrame);
        visualizeFrame(enhancedFrame, result, classNames);

        float fpsLime = calculateFPS(frameCount, start);
        if (fpsLime > 0) {
            printFps(fpsLime, enhancedFrame);
        }
        if (showAndWait("lime", frame, isImage)) {
            break;
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

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    std::vector<Detection> result;
    cv::Mat inputImage;
    try
    {
        YOLODetector detector(modelPath, isGPU, cv::Size(640, 640));
        cv::VideoCapture cap = captureVideo(inputPath,inputImage); // 0代表默认摄像头
        //创建高精度计时器
        auto start = std::chrono::high_resolution_clock::now();
        //纪录视频帧数
        int frameCount = 0;
        std::mutex cap_mtx;
        bool isImage = isImageFile(inputPath);

        std::thread beforeThread(processBeforeFrame, isImage, std::ref(inputImage), std::ref(cap), std::ref(frameCount), std::ref(start), std::ref(cap_mtx));
        std::thread detectThread(processDetectFrame, isImage, std::ref(inputImage), std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx));
        std::thread limeThread(processLimeFrame, isImage, std::ref(inputImage), std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx));
        
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