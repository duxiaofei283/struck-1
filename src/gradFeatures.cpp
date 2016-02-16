#include "gradFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

//#define RAD2DEG 180.0/M_PI
static const int kNumGrad = 16;


GradFeatures::GradFeatures(const Config &conf)
{
    SetCount(kNumGrad);
    std::cout<<"grad histogram bins: " << GetCount() << std::endl;

    m_bin_step = 2*M_PI / kNumGrad;

}

void GradFeatures::compGrad(const cv::Mat &img, cv::Mat& ori, cv::Mat& mag) const
{
    assert(img.channels() == 1 || img.channels() == 3);
    cv::Mat grayImg;
    if(img.channels() == 3)
    {
        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1);
        cv::cvtColor(img, grayImg, CV_RGB2GRAY);
    }
    else if(img.channels() == 1)
    {
        //img.copyTo(grayImg);
        grayImg = img;
    }

    cv::Mat x_sobel, y_sobel;
    cv::Sobel(grayImg, x_sobel, CV_32FC1, 1, 0);
    cv::Sobel(grayImg, y_sobel, CV_32FC1, 0, 1);

    // continuous?
    int height = grayImg.rows;
    int width = grayImg.cols;
    if(ori.isContinuous() && mag.isContinuous() && x_sobel.isContinuous() && y_sobel.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy;
    float *xp, *yp, *op, *mp;
    for(iy=0; iy < height; ++iy)
    {
        xp = x_sobel.ptr<float>(iy);
        yp = y_sobel.ptr<float>(iy);
        op = ori.ptr<float>(iy);
        mp = mag.ptr<float>(iy);

        for(ix=0; ix < width; ++ix)
        {
            float x_grad = xp[ix];
            float y_grad = yp[ix];
            if(x_grad == 0)
               x_grad += FLT_EPSILON;
            op[ix] = std::atan2f(y_grad, x_grad) + M_PI;
            mp[ix] = std::sqrtf(x_grad*x_grad + y_grad*y_grad);
        }
    }
}

//void GradFeatures::compGrad(const cv::Mat &img, cv::Mat& ori, cv::Mat& mag) const
//{
//    assert(img.channels() == 1 || img.channels() == 3);
//    cv::Mat grayImg;
//    if(img.channels() == 3)
//    {
//        grayImg = cv::Mat(img.rows, img.cols, CV_8UC1);
//        cv::cvtColor(img, grayImg, CV_RGB2GRAY);
//    }
//    else if(img.channels() == 1)
//    {
//        //img.copyTo(grayImg);
//        grayImg = img;
//    }

//    cv::Mat x_sobel, y_sobel;
//    cv::Sobel(grayImg, x_sobel, CV_32FC1, 1, 0);
//    cv::Sobel(grayImg, y_sobel, CV_32FC1, 0, 1);

//    int ix, iy;

//    for(iy=0; iy < grayImg.rows; ++iy)
//    {
//        for(ix=0; ix < grayImg.cols; ++ix)
//        {
//            float x_grad = x_sobel.at<float>(iy, ix);
//            float y_grad = y_sobel.at<float>(iy, ix);
//            if(x_grad == 0)
//               x_grad += FLT_EPSILON;
//            ori.at<float>(iy, ix) = std::atan2f(y_grad, x_grad) + M_PI;
//            mag.at<float>(iy, ix) = std::sqrtf(x_grad*x_grad + y_grad*y_grad);
//        }
//    }
//}

int GradFeatures::compBinIdx(const float orientation) const
{
    int i = 0;
    for(i=0; i<kNumGrad; ++i)
    {
        if(orientation <= m_bin_step * (i+1))
            break;
    }

//    // alternative
//    int k = int(orientation / m_bin_step);
//    std::cout << "i - "<<i<< "; k - "<<k<<std::endl;
//    if (k != i)
//        int foo = 1;
    return i;
}

void GradFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();
    cv::Mat grayImg = s.GetImage().GetImage(0)(roi);

    cv::Mat oriImg(grayImg.rows, grayImg.cols, CV_32FC1);
    cv::Mat magImg(grayImg.rows, grayImg.cols, CV_32FC1);
    compGrad(grayImg, oriImg, magImg);

    // continuous?
    int height = rect.Height();
    int width = rect.Width();
    if(oriImg.isContinuous() && magImg.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int iy, ix;
    float *op, *mp;
    for(iy=0; iy < height; ++iy)
    {
        op = oriImg.ptr<float>(iy);
        mp = magImg.ptr<float>(iy);

        for(ix=0; ix < width; ++ix)
        {
            int bin_idx = compBinIdx(op[ix]);
            m_featVec[bin_idx] += mp[ix];
        }
    }
    m_featVec /= rect.Area();
}

