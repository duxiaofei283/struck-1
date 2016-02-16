#include "CircleRgbFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"
#include "mutils.h"

#include <iostream>
#include <math.h>

static const int kNumDist = 8;

CircleRgbFeatures::CircleRgbFeatures(const Config &conf) : m_rgb_feature(conf)
{
    m_rgb_num = m_rgb_feature.GetCount();
    SetCount(m_rgb_num * kNumDist);
    std::cout<< "circle rgb histogram bins: "<< GetCount() << std::endl;
    m_feature_map = cv::Mat::zeros(kNumDist, m_rgb_num, CV_32FC1);
    m_distmap_size = cv::Size(-1, -1);
    m_dist_step = -1;
}

//void CircleRgbFeatures::UpdateFeatureVector(const Sample &s)
//{
//    IntRect rect = s.GetROI();
//    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

//    m_featVec.setZero();
////    m_feature_map.setTo(cv::Scalar(0));
//    cv::Mat r_img = s.GetImage().GetImage(0)(roi);
//    cv::Mat g_img = s.GetImage().GetImage(1)(roi);
//    cv::Mat b_img = s.GetImage().GetImage(2)(roi);

//    // compute dist map
//    compDistMap(cv::Size(rect.Width(), rect.Height()));

//    // compute hist
//    int height = rect.Height();
//    int width = rect.Width();
//    std::vector<int> pixelnum_dist_bin(kNumDist, 0);

//    // continuous?
//    if(m_dist_map.isContinuous() && r_img.isContinuous() && g_img.isContinuous() && b_img.isContinuous())
//    {
//        width *= height;
//        height = 1;
//    }

//    int ix, iy;
//    float *p;
//    uchar *r, *g, *b;
//    for(iy=0; iy < rect.Height(); ++iy)
//    {
//        p = m_dist_map.ptr<float>(iy);
//        r = r_img.ptr<uchar>(iy);
//        g = g_img.ptr<uchar>(iy);
//        b = b_img.ptr<uchar>(iy);

//        for(ix=0; ix < rect.Width(); ++ix)
//        {
//           int dist_bin_idx =int(p[ix] / m_dist_step);
//           int r_bin_idx = m_rgb_feature.compBinIdx(r[ix], colorName::R);
//           int g_bin_idx = m_rgb_feature.getBinNum(colorName::R) + m_rgb_feature.compBinIdx(g[ix], colorName::G);
//           int b_bin_idx = m_rgb_feature.getBinNum(colorName::R) + m_rgb_feature.getBinNum(colorName::G) + m_rgb_feature.compBinIdx(b[ix], colorName::B);

////           m_feature_map.at<float>(dist_bin_idx, r_bin_idx)++;
////           m_feature_map.at<float>(dist_bin_idx, g_bin_idx)++;
////           m_feature_map.at<float>(dist_bin_idx, b_bin_idx)++;

//           pixelnum_dist_bin[dist_bin_idx]++;

//           m_featVec[dist_bin_idx * m_rgb_feature.GetCount() + r_bin_idx]++;
//           m_featVec[dist_bin_idx * m_rgb_feature.GetCount() + g_bin_idx]++;
//           m_featVec[dist_bin_idx * m_rgb_feature.GetCount() + b_bin_idx]++;
//        }
//    }

//    //    // version 1:
//    //    m_feature_map /= rect.Area();
//    //    m_featVec /= rect.Area();

//    // version 2:
//    for(int id=0; id<kNumDist; ++id)
//        for(int ic=0; ic < m_rgb_feature.GetCount(); ++ic)
//        {
////            m_feature_map.at<float>(id, ic) /= pixelnum_dist_bin[id] * kNumDist;
//            int fea_bin_idx = id * m_rgb_feature.GetCount() + ic;
//            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
//        }
//}

void CircleRgbFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();
//    m_feature_map.setTo(cv::Scalar(0));
    cv::Mat rgb_img = s.GetImage().GetRgbImage()(roi);

    // compute dist map
    compDistMap(cv::Size(rect.Width(), rect.Height()));

    // compute hist
    int height = rect.Height();
    int width = rect.Width();
    std::vector<int> pixelnum_dist_bin(kNumDist, 0);

    // continuous?
    if(m_dist_map.isContinuous() && rgb_img.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy;
    float *dp;
    uchar *p;
    for(iy=0; iy < rect.Height(); ++iy)
    {
        dp = m_dist_map.ptr<float>(iy);
        p = rgb_img.ptr<uchar>(iy);

        for(ix=0; ix < rect.Width(); ++ix)
        {
           int dist_bin_idx =int(dp[ix] / m_dist_step);
           cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
           rgbIndice rgb_bin_idx = m_rgb_feature.compBinIdx(pixel);

//           m_feature_map.at<float>(dist_bin_idx, r_bin_idx)++;
//           m_feature_map.at<float>(dist_bin_idx, g_bin_idx)++;
//           m_feature_map.at<float>(dist_bin_idx, b_bin_idx)++;

           pixelnum_dist_bin[dist_bin_idx]++;

           m_featVec[dist_bin_idx * m_rgb_num + rgb_bin_idx.r_idx]++;
           m_featVec[dist_bin_idx * m_rgb_num + rgb_bin_idx.g_idx]++;
           m_featVec[dist_bin_idx * m_rgb_num + rgb_bin_idx.b_idx]++;
        }
    }

    //    // version 1:
    //    m_feature_map /= rect.Area();
    //    m_featVec /= rect.Area();

    // version 2:
    for(int id=0; id<kNumDist; ++id)
        for(int ic=0; ic < m_rgb_num; ++ic)
        {
//            m_feature_map.at<float>(id, ic) /= pixelnum_dist_bin[id] * kNumDist;
            int fea_bin_idx = id * m_rgb_num + ic;
            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
        }
}

void CircleRgbFeatures::compDistMap(const cv::Size& patch_size)
{
    if (m_distmap_size == patch_size)
        return;
    else
    {
        m_distmap_size = patch_size;
        cv::Point center = cv::Point(m_distmap_size.width/2, m_distmap_size.height/2);

        // compute dist step
        float max_edge = float(std::max(m_distmap_size.width, m_distmap_size.height));
        float dist = std::max(getDist(cv::Point(0,0), center),
                              getDist(cv::Point(m_distmap_size.width-1, m_distmap_size.height-1), center));
        float max_dist = dist / max_edge + FLT_EPSILON;
        m_dist_step = max_dist / kNumDist;

        m_dist_map.create(m_distmap_size.height, m_distmap_size.width, CV_32FC1);

        int ix, iy;
        float* dp;
        cv::Point pt;
        for(iy=0; iy < m_distmap_size.height; ++iy)
        {
            dp = m_dist_map.ptr<float>(iy);
            for(ix=0; ix < m_distmap_size.width; ++ix)
            {
                pt = cv::Point(ix, iy);
                dp[ix] = getDist(pt, center) / max_edge;
            }
        }
    }
}




