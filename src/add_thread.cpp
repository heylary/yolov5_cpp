#include <iostream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"
#include "visualizer.h"
#include "lime.h"

void processBeforeFrame(cv::VideoCapture& cap, int& frameCount, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        {
            std::unique_lock<std::mutex> lock(mtx);
            cap >> frame;
            frameCount++;
        }
        float fpsBefore = calculateFPS(frameCount, start);
        if (fpsBefore > 0) {
            printFps(fpsBefore, frame);
        }
        cv::imshow("before", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}

void processDetectFrame(YOLODetector& detector, cv::VideoCapture& cap, int& frameCount, const std::vector<std::string>& classNames, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        std::vector<Detection> result;
        {
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
        cv::imshow("detect", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }
}

void processLimeFrame(YOLODetector& detector, cv::VideoCapture& cap, int& frameCount, const std::vector<std::string>& classNames, std::chrono::high_resolution_clock::time_point& start, std::mutex& mtx) {
    for (;;) {
        cv::Mat frame;
        std::vector<Detection> result;
        {
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
        cv::imshow("lime", enhancedFrame);
        if (cv::waitKey(1) == 'q') {
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
    cmd.add("gpu", '\0', "Inference on cuda device.");

    cmd.parse_check(argc, argv);

    bool isGPU = cmd.exist("gpu");
    const std::string classNamesPath = cmd.get<std::string>("class_names");
    const std::vector<std::string> classNames = utils::loadNames(classNamesPath);
    const std::string modelPath = cmd.get<std::string>("model_path");

    if (classNames.empty())
    {
        std::cerr << "Error: Empty class names file." << std::endl;
        return -1;
    }

    std::vector<Detection> result;

    try
    {
        YOLODetector detector(modelPath, isGPU, cv::Size(640, 640));
        cv::VideoCapture cap = captureVideo(); // 0代表默认摄像头
        //创建高精度计时器
        auto start = std::chrono::high_resolution_clock::now();
        //纪录视频帧数
        int frameCount = 0;
        std::mutex cap_mtx;

        std::thread beforeThread(processBeforeFrame, std::ref(cap), std::ref(frameCount), std::ref(start), std::ref(cap_mtx));
        std::thread detectThread(processDetectFrame, std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx));
        std::thread limeThread(processLimeFrame, std::ref(detector), std::ref(cap), std::ref(frameCount), std::ref(classNames), std::ref(start), std::ref(cap_mtx));

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