#include <iostream>
#include <opencv2/opencv.hpp>

#include "cmdline.h"
#include "utils.h"
#include "detector.h"
#include "visualizer.h"


int main(int argc, char* argv[])
{
    cv::namedWindow("before");
    cv::namedWindow("detect");
    
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
        int total_frames = 0;
        //计算FPS(每秒传输帧数)
        float fps = -1;
        for(;;) {
            cv::Mat frame;
            cap >> frame;
            cv::imshow("before", frame);
            if (frame.empty())
            {
                std::cerr << "Error: Empty frame." << std::endl;
                break;
            }
            result = detectFrame(detector,frame);   //检测帧
            visualizeFrame(frame,result,classNames);    //可视化
            frameCount++;
            total_frames++;
            float fpsLime = calculateFPS(frameCount, start);
            if (fpsLime > 0) {
                printFps(fpsLime, frame);
            }
            cv::imshow("detect", frame);

            if (cv::waitKey(1) == 'q') // 按下'q'键退出程序
                break;
        }
        cap.release(); // 释放资源
        std::cout << "Total frames: " << total_frames << "\n";
        cv::destroyAllWindows();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }
}