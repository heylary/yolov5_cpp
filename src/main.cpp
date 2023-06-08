#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"


int main(int argc, char* argv[])
{
    const float confThreshold = 0.3f;
    const float iouThreshold = 0.4f;

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

    YOLODetector detector {nullptr};
    std::vector<Detection> result;

    try
    {
        detector = YOLODetector(modelPath, isGPU, cv::Size(640, 640));
        std::cout << "Model was initialized." << std::endl;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    cv::VideoCapture cap(0); // 0代表默认摄像头，如果有多个摄像头，请更改参数

    if (!cap.isOpened())
    {
        std::cerr << "Error: Unable to open camera." << std::endl;
        return -1;
    }

    while (true)
    {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty())
        {
            std::cerr << "Error: Empty frame." << std::endl;
            break;
        }

        try
        {       
            result = detector.detect(frame, confThreshold, iouThreshold);
        }
        catch(const std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            return -1;
        }

        utils::visualizeDetection(frame, result, classNames);

        cv::imshow("result", frame);

        if (cv::waitKey(1) == 'q') // 按下'q'键退出程序
            break;
    }

    cap.release(); // 释放资源
    cv::destroyAllWindows();
    return 0;
}