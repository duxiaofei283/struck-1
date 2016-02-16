#include "HsvFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>

static const int kNumH = 8;
static const int kNumS = 4;
static const int kNumV = 4;

HsvFeatures::HsvFeatures(const Config &conf)
{
    int num_bins = kNumH * kNumS + kNumV;
    SetCount(num_bins);
    std::cout<< "hsv histogram bins: "<< GetCount() << std::endl;

    m_h_step = 180.0 / kNumH;
    m_s_step = 256.0 / kNumS;
    m_v_step = 256.0 / kNumV;

}

//HsvFeatures::HsvFeatures()
//{
//    int num_bins = kNumH * kNumS + kNumV;
//    SetCount(num_bins);
//    std::cout<< "hsv histogram bins: "<< GetCount() << std::endl;
//}

void HsvFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();
    cv::Mat hsv_img = s.GetImage().GetHsvImage()(roi);

    int height = rect.Height();
    int width = rect.Width();

    // continuous?
    if(hsv_img.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy;
    uchar *p;
    for(iy=0; iy < height; ++iy)
    {
        p = hsv_img.ptr<uchar>(iy);

        for(ix=0; ix < width; ++ix)
        {
            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
            auto bin_idx = compBinIdx(pixel);
            m_featVec[bin_idx]++;
        }
    }

    m_featVec /= rect.Area();

}

int HsvFeatures::compBinIdx(const cv::Vec3b &pixel) const
{
    auto h = pixel[0];
    auto s = pixel[1];
    auto v = pixel[2];

    int h_idx = int(h/m_h_step);
    int s_idx = int(s/m_s_step);
    int v_idx = int(v/m_v_step);

    int bin_idx;
    if(s < 15 || v < 30)
        bin_idx = kNumH * kNumS + v_idx;
    else
        bin_idx = h_idx * kNumS + s_idx;

    return bin_idx;
}

void HsvFeatures::compWBinIdx(const cv::Vec3b& pixel, int& bin_idx, float& weight)
{
    auto h = pixel[0];
    auto s = pixel[1];
    auto v = pixel[2];

    int h_idx = int(h/m_h_step);
    float h_center = h_idx * m_h_step + m_h_step * 0.5;
    int s_idx = int(s/m_s_step);
    float s_center = s_idx * m_s_step + m_s_step * 0.5;
    int v_idx = int(v/m_v_step);
    float v_center = v_idx * m_v_step + m_v_step * 0.5;

    if(s < 15 || v < 30)
    {
        bin_idx = kNumH * kNumS + v_idx;
        weight = 1 - std::abs(v - v_center)/m_v_step;
    }
    else
    {
        bin_idx = h_idx * kNumS + s_idx;
        weight = (1 - std::abs(h-h_center)/m_h_step) * (1 - std::abs(s - s_center)/m_s_step);
    }


}
