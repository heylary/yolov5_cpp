#include "lime.h"
#include <vector>
#include <iostream>
#include <cmath>

namespace feature {
    lime::lime(cv::Mat src) : channel(src.channels()) {}

    cv::Mat lime::lime_enhance(cv::Mat &src) {
        cv::Mat img_norm;
        src.convertTo(img_norm, CV_32F, 1 / 255.0, 0);

        cv::Size sz(img_norm.size());
        cv::Mat out(sz, CV_32F, cv::Scalar::all(0.0));

        // 估计光照分量并滤波（调用新的illumination_filter函数）
        cv::Mat gammT;
        Illumination(img_norm, out);
        Illumination_filter(out, gammT);
        

        // 应用非局部均值滤波（调用nonlocal_mean_filter函数）
        cv::Mat out_filtered;
        nonlocal_mean_filter(gammT, out_filtered);

        if (channel == 3) {
            std::vector<cv::Mat> img_norm_rgb;
            cv::split(img_norm, img_norm_rgb);

            cv::Mat one = cv::Mat::ones(sz, CV_32F);
            float nameta = 0.9;

            cv::Mat g = 1 - ((one - img_norm_rgb[0]) - (nameta * (one - out_filtered))) / out_filtered;
            cv::Mat b = 1 - ((one - img_norm_rgb[1]) - (nameta * (one - out_filtered))) / out_filtered;
            cv::Mat r = 1 - ((one - img_norm_rgb[2]) - (nameta * (one - out_filtered))) / out_filtered;

            cv::Mat g1, b1, r1;
            threshold(g, g1, 0.0, 0.0, cv::THRESH_TOZERO);
            threshold(b, b1, 0.0, 0.0, cv::THRESH_TOZERO);
            threshold(r, r1, 0.0, 0.0, cv::THRESH_TOZERO);

            img_norm_rgb[0] = g1;
            img_norm_rgb[1] = b1;
            img_norm_rgb[2] = r1;

            cv::merge(img_norm_rgb, out_lime);
            out_lime.convertTo(out_lime, CV_8U, 255);
        } else if(channel == 1) {
            cv::Mat one = cv::Mat::ones(sz, CV_32F);
            float nameta = 0.9;

            cv::Mat out = 1 - ((one - img_norm) - (nameta * (one - out_filtered))) / out_filtered;
            threshold(out, out_lime, 0.0, 0.0, cv::THRESH_TOZERO);
            out_lime.convertTo(out_lime, CV_8UC1, 255);
        } else {
            std::cout << "There is a problem with the channels" << std::endl;
            exit(-1);
        }
        return out_lime.clone();
    }

// 光照估计滤波器(新增双边滤波)
    void lime::Illumination_filter(cv::Mat& img_in, cv::Mat& img_out) {
        int ksize = 5;
        // 使用双边滤波进行光照估计滤波(add)
        cv::bilateralFilter(img_in, img_out, ksize, 75, 75);

        // 使用高斯模糊进行光照估计滤波
        cv::GaussianBlur(img_in, img_out, cv::Size(ksize, ksize), 0);

        int row = img_out.rows;
        int col = img_out.cols;
        float gamma = 0.8;

        cv::MatIterator_<float> it, end;
        for (it = img_out.begin<float>(), end = img_out.end<float>(); it != end; ++it) {
            float tem = std::pow(*it, gamma);
            tem = std::max(tem, 0.0001f);
            tem = std::min(tem, 1.0f);
            *it = tem;
        }

    }

// 光照估计函数(对于图像的每个像素，计算光照估计值)
    void lime::Illumination(cv::Mat& src, cv::Mat& out) {
        int row = src.rows;
        int col = src.cols;

        for(int i=0; i<row; i++) {
            for(int j=0; j<col; j++) {
                // 采用Poisson光照方程
                out.at<float>(i,j) = lime::compare(src.at<cv::Vec3f>(i,j)[0],
                                                   src.at<cv::Vec3f>(i,j)[1],
                                                   src.at<cv::Vec3f>(i,j)[2]);
            }
        }
    }

// 非局部均值滤波函数实现(new)
    void lime::nonlocal_mean_filter(cv::Mat& img_in, cv::Mat& img_out) {
        int ksize = 7; // 邻域窗口大小
        float h = 10.0; // 决定权重的参数
        // cv::fastNlMeansDenoising(img_in, img_out);
        // cv::fastNlMeansDenoisingColored(img_in, img_out, h, h, ksize, ksize);
        img_out = img_in;
    }

}
