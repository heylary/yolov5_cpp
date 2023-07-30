#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.h"


constexpr float CONF_THRESHOLD = 0.3f; 
constexpr float IOU_THRESHOLD = 0.4f;
constexpr int FPS_CALC_THRESHOLD = 1; // Calculate FPS

cv::VideoCapture captureVideo(const std::string& inputPath, cv::Mat& image);
void printFps(float fps, cv::Mat& frame);
std::vector<Detection> detectFrame(YOLODetector& detector, cv::Mat& frame);
void visualizeFrame(cv::Mat& frame, std::vector<Detection>& result, const std::vector<std::string>& classNames);
float calculateFPS(int& frameCount, std::chrono::high_resolution_clock::time_point& start);
bool isImageFile(const std::string& path);
bool showAndWait(const std::string& winName, const cv::Mat& image, bool isImage);