#ifndef FEATURE_EXACTION_LIME_H
#define FEATURE_EXACTION_LIME_H
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
namespace feature
{
class lime
 {
public:
     int channel;
     cv::Mat out_lime;

public:
    lime(cv::Mat src);
    cv::Mat lime_enhance(cv::Mat& src);
    // 计算亮度值（修改为Poisson光照方程）
    static inline float compare(float& B,float& G,float& R)
    {
        return 0.299 * R + 0.587 * G + 0.114 * B;
    }

    void Illumination(cv::Mat& src,cv::Mat& out);

    void Illumination_filter(cv::Mat& img_in,cv::Mat& img_out);

    void nonlocal_mean_filter(cv::Mat& img_in, cv::Mat& img_out);

};

}

#endif //FEATURE_EXACTION_LIME_H
