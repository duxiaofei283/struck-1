#include "CircleGradFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"
#include "mutils.h"

#include <iostream>
#include <math.h>

static const int kNumDist = 8;

CircleGradFeatures::CircleGradFeatures(const Config &conf) : m_grad_feature(conf)
{
    m_grad_num = m_grad_feature.GetCount();
    SetCount(m_grad_num * kNumDist);
    std::cout<< "circle histogram bins: "<< GetCount() << std::endl;
//    m_feature_map = cv::Mat::zeros(kNumDist, grad_bin_num, CV_32FC1);
    m_distmap_size = cv::Size(-1, -1);
    m_orimap_size = cv::Size(-1, -1);
    m_dist_step = -1;
}

void CircleGradFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();
//    m_feature_map.setTo(cv::Scalar(0));
    cv::Mat grayImg = s.GetImage().GetImage(0)(roi);

    cv::Mat oriImg(grayImg.rows, grayImg.cols, CV_32FC1);
    cv::Mat magImg(grayImg.rows, grayImg.cols, CV_32FC1);
    m_grad_feature.compGrad(grayImg, oriImg, magImg);

    // compute dist and orientation map
    compDistMap(cv::Size(rect.Width(), rect.Height()));
    compOriMap(cv::Size(rect.Width(), rect.Height()));

    // continuous?
    int height = rect.Height();
    int width = rect.Width();
    if(m_dist_map.isContinuous() && oriImg.isContinuous() && m_ori_map.isContinuous() && magImg.isContinuous())
    {
        width *= height;
        height = 1;
    }
    else
    {
        std::cout<<"not continuous..."<<std::endl;
    }

    // compute hist
    std::vector<int> pixelnum_dist_bin(kNumDist, 0);
    int ix, iy;
    float *dp, *op, *mp, *mop;
    for(iy=0; iy < height; ++iy)
    {
        dp = m_dist_map.ptr<float>(iy);
        op = oriImg.ptr<float>(iy);
        mp = magImg.ptr<float>(iy);
        mop = m_ori_map.ptr<float>(iy);

        for(ix=0; ix < width; ++ix)
        {
            // the relative angle o has to be in range (0, 2pi)
            float o = op[ix] - mop[ix];
            o = (o<0) ? o+2*M_PI : o;
//            float o = oriImg.at<float>(iy, ix) - m_ori_map.at<float>(iy, ix);
            int grad_bin_idx = m_grad_feature.compBinIdx(o);
//            int dist_bin_idx = int(m_dist_map.at<float>(iy, ix) / m_dist_step);
            int dist_bin_idx = int(dp[ix]/ m_dist_step);
//            m_feature_map.at<float>(dist_bin_idx, grad_bin_idx) += magImg.at<float>(iy, ix);
//            m_feature_map.at<float>(dist_bin_idx, grad_bin_idx) += mp[ix];
            pixelnum_dist_bin[dist_bin_idx]++;
            int fea_bin_idx = dist_bin_idx * m_grad_num + grad_bin_idx;
            m_featVec[fea_bin_idx] += mp[ix];
        }
    }
//    // version 1:
//    m_feature_map /= rect.Area();
//    m_featVec /= rect.Area();

    // version 2:
    int id, ic;

    for(id=0; id<kNumDist; ++id)
        for(ic=0; ic < m_grad_num; ++ic)
        {
//            m_feature_map.at<float>(id, ic) /= pixelnum_dist_bin[id] * kNumDist;
            int fea_bin_idx = id * m_grad_num + ic;
            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
        }
}

void CircleGradFeatures::compDistMap(const cv::Size& patch_size)
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

void CircleGradFeatures::compOriMap(const cv::Size &patch_size)
{
    if (m_orimap_size == patch_size)
        return;
    else
    {
        m_orimap_size = patch_size;
        cv::Point center = cv::Point(m_orimap_size.width/2, m_orimap_size.height/2);

        // compute orientation
        m_ori_map.create(m_orimap_size.height, m_orimap_size.width, CV_32FC1);

        int ix, iy;
        float* op;
        float rx, ry;

        for(iy=0; iy < m_orimap_size.height; ++iy)
        {
            op = m_ori_map.ptr<float>(iy);

            for(ix=0; ix < m_orimap_size.width; ++ix)
            {
                rx = ix - center.x;
                ry = iy - center.y;
                if(rx == 0)
                    rx += FLT_EPSILON;
                op[ix] = std::atan2f(ry, rx) + M_PI ;
            }
        }
    }
}



