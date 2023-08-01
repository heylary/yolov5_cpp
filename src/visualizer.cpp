#include "visualizer.h"
#include "utils.h"

#include <iostream>
#include <exception>

bool showAndWait(const std::string& winName, const cv::Mat& image, bool isImage) {
    cv::imshow(winName, image);
    return cv::waitKey(isImage ? 0 : 1) == 'q';
}

bool isImageFile(const std::string& path) {
    std::string extension = path.substr(path.find_last_of(".") + 1);
    return extension == "jpg" || extension == "png" || extension == "jpeg";
}

//打开摄像头
cv::VideoCapture captureVideo(const std::string& inputPath, cv::Mat& image) {
    cv::VideoCapture cap;

    // 判断输入是否为图像
    if (isImageFile(inputPath)) {
        image = cv::imread(inputPath);

        if (image.empty()) {
            std::cerr << "Error: Unable to open image: " << inputPath << std::endl;
            throw;
        }
    } else {
        // 尝试打开输入路径作为视频文件
        cap.open(inputPath);

        // 如果无法打开输入路径作为视频文件，则尝试打开摄像头
        if (!cap.isOpened()) {
            cap.open(0);
        }

        if (!cap.isOpened()) {
            std::cerr << "Error: Unable to open video or camera." << std::endl;
            throw;
        }
    }

    return cap;
}


//计算帧率
float calculateFPS(int& frameCount, std::chrono::high_resolution_clock::time_point& start) {
    if (frameCount >= FPS_CALC_THRESHOLD) {
        const auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        const float fps = frameCount * 1000.0 / elapsed_time;

        frameCount = 0;
        start = std::chrono::high_resolution_clock::now();
        return fps;
    }
    return -1;
}

//打印帧率
void printFps(float fps, cv::Mat& frame, const std::string& windowName) {
    std::ostringstream fps_label;
    fps_label << std::fixed << std::setprecision(2);
    fps_label << "FPS: " << fps;
    std::string fps_label_str = fps_label.str();

    cv::putText(frame, fps_label_str, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
    cv::putText(frame, windowName, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
}



//目标检测
std::vector<Detection> detectFrame(YOLODetector& detector, cv::Mat& frame) {
    try {
        auto result = detector.detect(frame, CONF_THRESHOLD, IOU_THRESHOLD);
        return result;
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
}

//可视化检测结果
void visualizeFrame(cv::Mat& frame, std::vector<Detection>& result, const std::vector<std::string>& classNames) {
    try {
        utils::visualizeDetection(frame, result, classNames);
    } catch(const std::exception& e) {
        std::cerr << e.what() << std::endl;
        throw;
    }
}