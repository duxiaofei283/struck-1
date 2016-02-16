#include "RgbFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"

#include <iostream>

static const int kNumR = 8;
static const int kNumG = 8;
static const int kNumB = 8;

RgbFeatures::RgbFeatures(const Config &conf)
{
    int num_bins = kNumR + kNumG + kNumB;
    SetCount(num_bins);
    std::cout<< "RGB histogram bins: "<< GetCount() << std::endl;

    m_r_step = 256.0 / kNumR;
    m_g_step = 256.0 / kNumG;
    m_b_step = 256.0 / kNumB;

}

//void RgbFeatures::UpdateFeatureVector(const Sample &s)
//{
//    IntRect rect = s.GetROI();
//    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

//    m_featVec.setZero();
//    cv::Mat r_img = s.GetImage().GetImage(0)(roi);
//    cv::Mat g_img = s.GetImage().GetImage(1)(roi);
//    cv::Mat b_img = s.GetImage().GetImage(2)(roi);

//    int height = rect.Height();
//    int width = rect.Width();

//    // continuous?
//    if(r_img.isContinuous() && g_img.isContinuous() && b_img.isContinuous())
//    {
//        width *= height;
//        height = 1;
//    }

//    int ix, iy;
//    uchar *r, *g, *b;
//    for(iy=0; iy < height; ++iy)
//    {
//        r = r_img.ptr<uchar>(iy);
//        g = g_img.ptr<uchar>(iy);
//        b = b_img.ptr<uchar>(iy);

//        for(ix=0; ix< width; ++ix)
//        {
//            m_featVec[compBinIdx(r[ix], colorName::R)]++;
//            m_featVec[kNumR + compBinIdx(g[ix], colorName::G)]++;
//            m_featVec[kNumR + kNumG + compBinIdx(b[ix], colorName::B)]++;
//        }
//    }
//    m_featVec /= rect.Area();
//}

void RgbFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();
    cv::Mat rgb_img = s.GetImage().GetRgbImage()(roi);

    int height = rect.Height();
    int width = rect.Width();

    // continuous?
    if(rgb_img.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy;
    uchar *p;
    for(iy=0; iy < height; ++iy)
    {
        p = rgb_img.ptr<uchar>(iy);

        for(ix=0; ix< width; ++ix)
        {
            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
            rgbIndice rgb_bin_idx = compBinIdx(pixel);

            m_featVec[rgb_bin_idx.r_idx]++;
            m_featVec[rgb_bin_idx.g_idx]++;
            m_featVec[rgb_bin_idx.b_idx]++;
        }
    }
    m_featVec /= rect.Area();
}

rgbIndice& RgbFeatures::compBinIdx(const cv::Vec3b &pixel) const
{
    rgbIndice rgb_indice;
    rgb_indice.r_idx = pixel[0] / m_r_step;
    rgb_indice.g_idx = pixel[1] / m_g_step + kNumR;
    rgb_indice.b_idx = pixel[2] / m_b_step + kNumR + kNumG;

    return rgb_indice;
}

int RgbFeatures::compBinIdx(const uchar &intensity, const colorName& cn) const
{
    int bin_idx;
    switch (cn)
    {
    case colorName::R:
        bin_idx = intensity / m_r_step;
        break;
    case colorName::G:
        bin_idx = intensity / m_g_step;
        break;
    case colorName::B:
        bin_idx = intensity / m_b_step;
        break;
    default:
        bin_idx = -1;
        break;
    }

    return bin_idx;

}

int RgbFeatures::getBinNum(const colorName &cn) const
{
    int bin_num;
    switch (cn)
    {
    case colorName::R:
        bin_num = kNumR;
        break;
    case colorName::G:
        bin_num = kNumG;
        break;
    case colorName::B:
        bin_num = kNumB;
        break;
    default:
        bin_num = -1;
        break;
    }
    return bin_num;
}
