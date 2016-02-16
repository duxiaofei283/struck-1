#include "CircleFeatures.h"
#include "Config.h"
#include "Sample.h"
#include "Rect.h"
#include "mutils.h"

#include <iostream>
#include <unordered_map>

static const int kNumDist = 8;
static const int kMiniPatch = 3;

CircleFeatures::CircleFeatures(const Config &conf) : m_hsv_feature(conf)
{
    m_hsv_num = m_hsv_feature.GetCount();
    SetCount(m_hsv_num * kNumDist);
    std::cout<< "circle histogram bins: "<< GetCount() << std::endl;
//    m_feature_map = cv::Mat::zeros(kNumDist, hsv_bin_num, CV_32FC1);
    m_distmap_size = cv::Size(-1, -1);
    m_dist_step = -1;
    m_half_miniPatch = kMiniPatch /2;
}

// fast version
void CircleFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

    m_featVec.setZero();

    cv::Mat hsv_bin_map = m_hsv_bin_map(roi);
    cv::Mat hsv_theta_map = m_hsv_theta_map(roi);

    // compute dist map
    compWDistMap(cv::Size(rect.Width(), rect.Height()));

    // compute hist
    int height = rect.Height();
    int width = rect.Width();
    std::vector<float> pixelnum_dist_bin(kNumDist, 0);

    // continuous?
    if(m_dist_map.isContinuous() && m_dist_theta_map.isContinuous() &&
       hsv_bin_map.isContinuous() && hsv_theta_map.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy, dist_bin_idx, hsv_bin_idx;
    int *dp, *hbp;
    float *tp, *htp;
    float hsv_theta, d_theta;

    for(iy=0; iy < height; ++iy)
    {
        dp = m_dist_map.ptr<int>(iy);
        tp = m_dist_theta_map.ptr<float>(iy);

        hbp = hsv_bin_map.ptr<int>(iy);
        htp = hsv_theta_map.ptr<float>(iy);

        for(ix=0; ix < width; ++ix)
        {

            dist_bin_idx = dp[ix];

            // plus (theta) version
            hsv_bin_idx = hbp[ix];
            hsv_theta = htp[ix];
            d_theta = tp[ix];

            pixelnum_dist_bin[dist_bin_idx] += hsv_theta * d_theta;
            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
            m_featVec[fea_bin_idx] += hsv_theta * d_theta;
        }
    }

    Eigen::VectorXd test(GetCount());
    test = m_featVec;

    // normalize

    auto start = std::clock();
    for(int id=0; id<kNumDist; ++id)
        for(int ic=0; ic < m_hsv_num; ++ic)
        {
            int fea_bin_idx = id * m_hsv_num + ic;
            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
        }
}


//// slow version
//void CircleFeatures::UpdateFeatureVector(const Sample &s)
//{
//    IntRect rect = s.GetROI();
//    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

//    m_featVec.setZero();
////    m_feature_map.setTo(cv::Scalar(0));
//    cv::Mat hsv_img = s.GetImage().GetHsvImage()(roi);

//    // compute dist map
////    compDistMap(cv::Size(rect.Width(), rect.Height()));
//    compWDistMap(cv::Size(rect.Width(), rect.Height()));

//    // compute hist
//    int height = rect.Height();
//    int width = rect.Width();
//    std::vector<float> pixelnum_dist_bin(kNumDist, 0);

//    // continuous?
//    if(m_dist_map.isContinuous() && hsv_img.isContinuous() && m_dist_theta_map.isContinuous())
//    {
//        width *= height;
//        height = 1;
//    }

//    int ix, iy, dist_bin_idx;
//    int *dp;
//    float *tp;
//    uchar *p;
//    int row_idx, col_idx, miniPatch_size;
//    int hsv_bin_idx;
//    float hsv_theta;
//    float d_theta;

//    for(iy=0; iy < height; ++iy)
//    {
//        dp = m_dist_map.ptr<int>(iy);
//        tp = m_dist_theta_map.ptr<float>(iy);
////        p =  hsv_img.ptr<uchar>(iy);

//        for(ix=0; ix < width; ++ix)
//        {
////            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);

////            int dist_bin_idx = int(dp[ix] / m_dist_step);
//            dist_bin_idx = dp[ix];

//            cv::Vec3i pixel(0, 0, 0);
//            miniPatch_size = 0;
//            // average the mini patch
//            for(int i = -m_half_miniPatch; i <= m_half_miniPatch; ++i)
//            {
//                // row index has to be inside patch
//                row_idx = iy + i;
//                if( row_idx>=0 && row_idx<height)
//                {
//                    p = hsv_img.ptr<uchar>(row_idx);
//                }
//                else
//                {
//                    continue;
//                }

//                // col index has to be inside patch
//                for(int j = -m_half_miniPatch; j <= m_half_miniPatch; ++j)
//                {
//                    col_idx = ix + j;
//                    if(col_idx>=0 && col_idx<width)
//                    {
//                        pixel += cv::Vec3i(p[3*col_idx+0], p[3*col_idx+1], p[3*col_idx+2]);
//                        miniPatch_size += 1;
//                    }
//                    else
//                    {
//                        continue;
//                    }
//                }
//            }
//            pixel /= miniPatch_size;

////            // plus one version
////            hsv_bin_idx = m_hsv_feature.compBinIdx(cv::Vec3b(pixel));
//////            int hsv_bin_idx = m_hsv_feature.compBinIdx(pixel);
//////            m_feature_map.at<float>(dist_bin_idx, hsv_bin_idx)++;
////            pixelnum_dist_bin[dist_bin_idx]++;
////            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
////            m_featVec[fea_bin_idx]++;

//            // plus (theta) version
//            m_hsv_feature.compWBinIdx(cv::Vec3b(pixel), hsv_bin_idx, hsv_theta);
//            d_theta = tp[ix];
//            pixelnum_dist_bin[dist_bin_idx] += hsv_theta * d_theta;
//            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
//            m_featVec[fea_bin_idx] += hsv_theta * d_theta;

//        }
//    }

////    // version 1:
////    m_feature_map /= rect.Area();
////    m_featVec /= rect.Area();

//    // version 2:
//    for(int id=0; id<kNumDist; ++id)
//        for(int ic=0; ic < m_hsv_num; ++ic)
//        {
////            m_feature_map.at<float>(id, ic) /= pixelnum_dist_bin[id] * kNumDist;
//            int fea_bin_idx = id * m_hsv_num + ic;
//            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
//        }

//}

//// compare version
//void CircleFeatures::UpdateFeatureVector(const Sample &s)
//{
//    IntRect rect = s.GetROI();
//    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

//    m_featVec.setZero();
//    cv::Mat hsv_img = s.GetImage().GetHsvImage()(roi);
//    cv::Mat hsv_bin_map = m_hsv_bin_map(roi);
//    cv::Mat hsv_theta_map = m_hsv_theta_map(roi);

//    // compute dist map
//    compWDistMap(cv::Size(rect.Width(), rect.Height()));

//    // compute hist
//    int height = rect.Height();
//    int width = rect.Width();
//    std::vector<float> pixelnum_dist_bin(kNumDist, 0);

//    // continuous?
//    if(m_dist_map.isContinuous() && hsv_img.isContinuous() && m_dist_theta_map.isContinuous())
//    {
//        width *= height;
//        height = 1;
//    }

//    int ix, iy, dist_bin_idx;
//    int *dp, *hbp;
//    float *tp, *htp;
//    uchar *p;
//    int row_idx, col_idx, miniPatch_size;
//    int hsv_bin_idx;
//    float hsv_theta;
//    float d_theta;

//    for(iy=0; iy < height; ++iy)
//    {
//        dp = m_dist_map.ptr<int>(iy);
//        tp = m_dist_theta_map.ptr<float>(iy);

//        hbp = hsv_bin_map.ptr<int>(iy);
//        htp = hsv_theta_map.ptr<float>(iy);

//        for(ix=0; ix < width; ++ix)
//        {

//            dist_bin_idx = dp[ix];

//            cv::Vec3i pixel(0, 0, 0);
//            miniPatch_size = 0;
//            // average the mini patch
//            for(int i = -m_half_miniPatch; i <= m_half_miniPatch; ++i)
//            {
//                // row index has to be inside patch
//                row_idx = iy + i;
//                if( row_idx>=0 && row_idx<height)
//                {
//                    p = hsv_img.ptr<uchar>(row_idx);
//                }
//                else
//                {
//                    continue;
//                }

//                // col index has to be inside patch
//                for(int j = -m_half_miniPatch; j <= m_half_miniPatch; ++j)
//                {
//                    col_idx = ix + j;
//                    if(col_idx>=0 && col_idx<width)
//                    {
//                        pixel += cv::Vec3i(p[3*col_idx+0], p[3*col_idx+1], p[3*col_idx+2]);
//                        miniPatch_size += 1;
//                    }
//                    else
//                    {
//                        continue;
//                    }
//                }
//            }
//            pixel /= miniPatch_size;

////            // plus one version
////            hsv_bin_idx = m_hsv_feature.compBinIdx(cv::Vec3b(pixel));
//////            int hsv_bin_idx = m_hsv_feature.compBinIdx(pixel);
//////            m_feature_map.at<float>(dist_bin_idx, hsv_bin_idx)++;
////            pixelnum_dist_bin[dist_bin_idx]++;
////            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
////            m_featVec[fea_bin_idx]++;

//            // plus (theta) version
//            m_hsv_feature.compWBinIdx(cv::Vec3b(pixel), hsv_bin_idx, hsv_theta);

//            // compare
//            int hsv_bin_idx2 = hbp[ix];
//            float hsv_theta2 = htp[ix];

//            if(hsv_bin_idx != hsv_bin_idx2 || hsv_theta != hsv_theta2)
//            {
//                int foo = 1;
//            }

//            d_theta = tp[ix];
//            pixelnum_dist_bin[dist_bin_idx] += hsv_theta * d_theta;
//            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
//            m_featVec[fea_bin_idx] += hsv_theta * d_theta;

//        }
//    }

//    // normalize
//    for(int id=0; id<kNumDist; ++id)
//        for(int ic=0; ic < m_hsv_num; ++ic)
//        {
//            int fea_bin_idx = id * m_hsv_num + ic;
//            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
//        }

//}

// hist version ( not right...
//void CircleFeatures::UpdateFeatureVector(const Sample &s)
//{
//    IntRect rect = s.GetROI();
//    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());

//    m_featVec.setZero();
////    m_feature_map.setTo(cv::Scalar(0));
//    cv::Mat hsv_img = s.GetImage().GetHsvImage()(roi);

//    // compute dist map
////    compDistMap(cv::Size(rect.Width(), rect.Height()));
//    compWDistMap(cv::Size(rect.Width(), rect.Height()));

//    // compute hist
//    int height = rect.Height();
//    int width = rect.Width();
//    std::vector<float> pixelnum_dist_bin(kNumDist, 0);

//    // continuous?
//    if(m_dist_map.isContinuous() && hsv_img.isContinuous() && m_theta_map.isContinuous())
//    {
//        width *= height;
//        height = 1;
//    }

//    int ix, iy;
//    uchar *p;
//    int *hp;
//    float *htp;
//    cv::Vec3b pixel;
//    cv::Mat hsv_bin_map = cv::Mat::zeros(height, width, CV_32SC1);
//    cv::Mat hsv_theta_map = cv::Mat::zeros(height, width, CV_32FC1);
//    for(iy=0; iy<height; ++iy)
//    {
//        hp = hsv_bin_map.ptr<int>(iy);
//        htp = hsv_theta_map.ptr<float>(iy);
//        p =  hsv_img.ptr<uchar>(iy);
//        for(ix=0; ix<width; ++ix)
//        {
//            pixel = cv::Vec3b(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
//            m_hsv_feature.compWBinIdx(pixel, hp[ix], htp[ix]);

//        }
//    }
//    int *dp;
//    float *dtp;
//    int row_idx, col_idx, hsv_bin_idx, dist_bin_idx;
//    float hsv_theta, d_theta;
//    std::vector<int> hsv_bin_lst;
//    std::vector<float> hsv_theta_lst;
//    for(iy=0; iy<height; ++iy)
//    {
//        dp = m_dist_map.ptr<int>(iy);
//        dtp = m_theta_map.ptr<float>(iy);

//        for(ix=0; ix<width; ++ix)
//        {
//            // the mini patch
//            hsv_bin_lst.clear();
//            hsv_theta_lst.clear();
//            for(int i=-m_half_miniPatch; i<=m_half_miniPatch; ++i)
//            {
//                row_idx = iy + i;
//                // row index has to be inside patch
//                if(row_idx>=0 && row_idx<height)
//                {
//                    hp = hsv_bin_map.ptr<int>(row_idx);
//                    htp = hsv_theta_map.ptr<float>(row_idx);
//                }
//                else
//                {
//                    continue;
//                }
//                for(int j=-m_half_miniPatch; j<=m_half_miniPatch; ++j)
//                {
//                    col_idx = ix +j;
//                    // col index has to be inside patch
//                    if(col_idx>=0 && col_idx<width)
//                    {
//                        hsv_bin_lst.push_back(hp[col_idx]);
//                        hsv_theta_lst.push_back(htp[col_idx]);
//                    }
//                    else
//                    {
//                        continue;
//                    }
//                }
//            }

//            // top hsv in the mini patch, plus (theta) version
//            compMiniPatchBinIdx(hsv_bin_lst, hsv_theta_lst, hsv_bin_idx, hsv_theta);
//            d_theta = dtp[ix];
//            dist_bin_idx = dp[ix];
//            pixelnum_dist_bin[dist_bin_idx] += hsv_theta * d_theta;
//            int fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
//            m_featVec[fea_bin_idx] += hsv_theta * d_theta;
//        }
//    }

//    for(int id=0; id<kNumDist; ++id)
//        for(int ic=0; ic < m_hsv_num; ++ic)
//        {
////            m_feature_map.at<float>(id, ic) /= pixelnum_dist_bin[id] * kNumDist;
//            int fea_bin_idx = id * m_hsv_num + ic;
//            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
//        }

//}



void CircleFeatures::compDistMap(const cv::Size& patch_size)
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

        int height = m_distmap_size.height;
        int width = m_distmap_size.width;

        int ix, iy;
        float* dp;
        cv::Point pt;
        for(iy=0; iy < height; ++iy)
        {
            dp = m_dist_map.ptr<float>(iy);
            for(ix=0; ix < width; ++ix)
            {
                pt.x = ix;
                pt.y = iy;
                dp[ix] = getDist(pt, center) / max_edge;
            }
        }
    }
}

void CircleFeatures::compWDistMap(const cv::Size &patch_size)
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

        m_dist_map.create(m_distmap_size.height, m_distmap_size.width, CV_32SC1);
        m_dist_theta_map.create(m_distmap_size.height, m_distmap_size.width, CV_32FC1);

        int height = m_distmap_size.height;
        int width = m_distmap_size.width;

        int ix, iy, dist_bin_idx;
        int* dp;
        float* tp;
        cv::Point pt;
        float pc_dist, d_center;
        for(iy=0; iy < height; ++iy)
        {
            dp = m_dist_map.ptr<int>(iy);
            tp = m_dist_theta_map.ptr<float>(iy);
            for(ix=0; ix < width; ++ix)
            {
                pt.x = ix;
                pt.y = iy;
                pc_dist = getDist(pt, center) / max_edge;
                dist_bin_idx = int(pc_dist / m_dist_step);
                d_center = dist_bin_idx * m_dist_step + m_dist_step * 0.5;
                float theta = 1 - std::abs(pc_dist- d_center)/m_dist_step;
                dp[ix] = dist_bin_idx;
                tp[ix] = theta;
            }
        }
    }
}

void CircleFeatures::compMiniPatchBinIdx(const std::vector<int> &hsv_idx_lst, const std::vector<float> &theta_lst, int &hsv_idx, float &theta)
{
    assert(hsv_idx_lst.size() == theta_lst.size());

//    for(int i=0; i<hsv_idx_lst.size(); ++i)
//    {
//        std::cout << hsv_idx_lst[i]  <<", "<< theta_lst[i] <<std::endl;
//    }
//    std::cout<<std::endl;

    std::unordered_map<int, float> freq_map;

    for(int i=0; i<hsv_idx_lst.size(); ++i)
    {
        std::unordered_map<int, float>::iterator it(freq_map.find(hsv_idx_lst[i]));
        if(it != freq_map.end())
            it->second += theta_lst[i];
        else
            freq_map[hsv_idx_lst[i]] = theta_lst[i];
    }

    theta = 0;
    for(auto it=freq_map.cbegin(); it != freq_map.cend(); ++it)
        if(it->second > theta)
        {
            hsv_idx = it->first;
            theta = it->second;
        }

}

// fast version
void CircleFeatures::Eval(const MultiSample &s, std::vector<Eigen::VectorXd> &featVecs)
{
    ImageRep si = s.GetImage();
    int img_height = si.GetHsvImage().rows;
    int img_width = si.GetHsvImage().cols;

    // get the large rect
    float x_min = FLT_MAX;
    float y_min = FLT_MAX;
    float x_max = -FLT_MAX;
    float y_max = -FLT_MAX;
    FloatRect r;
    for(int i=0; i < s.GetRects().size(); ++i)
    {
        r = s.GetRects()[i];
        if(x_min > r.XMin())
            x_min = r.XMin();
        if(x_max < r.XMax())
            x_max = r.XMax();
        if(y_min > r.YMin())
            y_min = r.YMin();
        if(y_max < r.YMax())
            y_max = r.YMax();
    }
    // get a pixel larger
    x_min = (x_min <= 0) ? x_min : (x_min-1);
    y_min = (y_min <= 0) ? y_min : (y_min-1);
    x_max = (x_max >= img_width) ? x_max : (x_max+1);
    y_max = (y_max >= img_height) ? y_max : (y_max+1);
    // ??? problem? float to int
    IntRect big_rect(x_min, y_min, x_max-x_min, y_max-y_min);


    // version fast
    uchar *p;
    uchar *p1;
    int *hbp;
    IntRect local_rect;
    cv::Mat local_hsv_img = cv::Mat(img_height, img_width, CV_8UC3);
    local_hsv_img.setTo(cv::Scalar(-1));

    for(int iy=big_rect.YMin(); iy < big_rect.YMax(); ++iy)
    {
        p1 = local_hsv_img.ptr<uchar>(iy);
        for(int ix=big_rect.XMin(); ix < big_rect.XMax(); ++ix)
        {
            // get local patch rect
            x_min = (ix - m_half_miniPatch <= 0) ? 0 : ix - m_half_miniPatch;
            y_min = (iy - m_half_miniPatch <= 0) ? 0 : iy - m_half_miniPatch;
            x_max = (ix + m_half_miniPatch + 1 >= img_width ) ? img_width  : ix + m_half_miniPatch + 1;
            y_max = (iy + m_half_miniPatch + 1 >= img_height) ? img_height : iy + m_half_miniPatch + 1;
            local_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);
            for(int channel=0; channel<3; ++channel)
                p1[3*ix+channel] = uchar(si.MeanHsv(local_rect, channel));

        }
    }


    // compute the hsv bin index map
    m_hsv_bin_map = cv::Mat(img_height, img_width, CV_32SC1);
    m_hsv_bin_map.setTo(-1);
    m_hsv_theta_map = cv::Mat(img_height, img_width, CV_32FC1);
    m_hsv_theta_map.setTo(-1.f);
    float *htp;
    for(int iy=big_rect.YMin(); iy < big_rect.YMax(); ++iy)
    {
        p = local_hsv_img.ptr<uchar>(iy);
        hbp = m_hsv_bin_map.ptr<int>(iy);
        htp = m_hsv_theta_map.ptr<float>(iy);
        for(int ix=big_rect.XMin(); ix < big_rect.XMax(); ++ix)
        {
            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
//            hbp[ix] = m_hsv_feature.compBinIdx(pixel);
            m_hsv_feature.compWBinIdx(pixel, hbp[ix], htp[ix]);
        }
    }

    // default implementation
    auto start = std::clock();
    featVecs.resize(s.GetRects().size());
    for (int i = 0; i < (int)featVecs.size(); ++i)
    {
        featVecs[i] = Features::Eval(s.GetSample(i));
    }
    std::cout<<featVecs.size()<<std::endl;
    std::cout<<"feature time "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;
    std::cout<<"==============================================================="<<std::endl;
}

//// compare version
//void CircleFeatures::Eval(const MultiSample &s, std::vector<Eigen::VectorXd> &featVecs)
//{
//    ImageRep si = s.GetImage();
//    cv::Mat hsv_img = si.GetHsvImage();
//    int img_height = hsv_img.rows;
//    int img_width = hsv_img.cols;

//    // get the large rect
//    float x_min = FLT_MAX;
//    float y_min = FLT_MAX;
//    float x_max = -FLT_MAX;
//    float y_max = -FLT_MAX;
//    FloatRect r;
//    for(int i=0; i < s.GetRects().size(); ++i)
//    {
//        r = s.GetRects()[i];
//        if(x_min > r.XMin())
//            x_min = r.XMin();
//        if(x_max < r.XMax())
//            x_max = r.XMax();
//        if(y_min > r.YMin())
//            y_min = r.YMin();
//        if(y_max < r.YMax())
//            y_max = r.YMax();
//    }
//    // get a pixel larger
//    x_min = (x_min <= 0) ? x_min : (x_min-1);
//    y_min = (y_min <= 0) ? y_min : (y_min-1);
//    x_max = (x_max >= img_width) ? x_max : (x_max+1);
//    y_max = (y_max >= img_height) ? y_max : (y_max+1);
//    // ??? problem? float to int
//    IntRect big_rect(x_min, y_min, x_max-x_min, y_max-y_min);


//    // version 1
//    uchar *p;
//    uchar *p1;
//    int *hbp;
//    IntRect local_rect;
//    cv::Mat local_hsv_img = cv::Mat(img_height, img_width, CV_8UC3);
//    local_hsv_img.setTo(cv::Scalar(-1));
//    // version 2
//    uchar *p2;
//    cv::Mat local_hsv_img2 = cv::Mat(img_height, img_width, CV_8UC3);
//    local_hsv_img2.setTo(cv::Scalar(-1));
//    int row_idx, col_idx;

//    for(int iy=big_rect.YMin(); iy < big_rect.YMax(); ++iy)
//    {
//        p1 = local_hsv_img.ptr<uchar>(iy);
//        p2 = local_hsv_img2.ptr<uchar>(iy);
//        for(int ix=big_rect.XMin(); ix < big_rect.XMax(); ++ix)
//        {
//            // version 1
//            // get local patch rect
//            x_min = (ix - m_half_miniPatch <= 0) ? 0 : ix - m_half_miniPatch;
//            y_min = (iy - m_half_miniPatch <= 0) ? 0 : iy - m_half_miniPatch;
//            x_max = (ix + m_half_miniPatch + 1 >= img_width ) ? img_width  : ix + m_half_miniPatch + 1;
//            y_max = (iy + m_half_miniPatch + 1 >= img_height) ? img_height : iy + m_half_miniPatch + 1;
//            local_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);
//            for(int channel=0; channel<3; ++channel)
//                p1[3*ix+channel] = si.MeanHsv(local_rect, channel);

//            std::cout<<int(p1[3*ix+0])<<","<<int(p1[3*ix+1])<<","<<int(p1[3*ix+2])<<std::endl;

//            // version 2
//            // average local patch
//            cv::Vec3i pixel(0, 0, 0);
//            int localPatch_size = 0;
//            for(int i=-m_half_miniPatch; i <= m_half_miniPatch; ++i)
//            {
//                // row index has to be inside image
//                row_idx = iy + i;
//                if(row_idx>=0 && row_idx<img_height)
//                    p = hsv_img.ptr<uchar>(row_idx);
//                else
//                    continue;

//                for(int j=-m_half_miniPatch; j<=m_half_miniPatch; ++j)
//                {
//                    // col index has to be inside image
//                    col_idx = ix + j;
//                    if(col_idx>=0 && col_idx<img_width)
//                    {
//                        pixel += cv::Vec3i(p[3*col_idx+0], p[3*col_idx+1], p[3*col_idx+2]);
//                        localPatch_size += 1;
//                    }
//                    else
//                        continue;
//                }
//            }
//            // asign
//            std::cout<<pixel[0]<<","<<pixel[1]<<","<<pixel[2]<<std::endl;
//            pixel /= localPatch_size;
//            std::cout<<pixel[0]<<","<<pixel[1]<<","<<pixel[2]<<std::endl;
//            for(int channel=0; channel<3; ++channel)
//                p2[3*ix+channel] = uchar(pixel[channel]);

//            // compare
//            for(int channel=0; channel<3; ++channel)
//            {
//                 std::cout <<channel<<" v1 " << p1[3*ix+channel]<< "; v2 "<< p2[3*ix+channel]<< std::endl;
//                 if(std::abs(p1[3*ix+channel] - p2[3*ix+channel])>1)
//                     int foo = 1;
//            }


//        }
//    }




//    // compute the hsv bin index map
////    uchar *p;
//    m_hsv_bin_map = cv::Mat(img_height, img_width, CV_32SC1);
//    m_hsv_bin_map.setTo(-1);
//    m_hsv_theta_map = cv::Mat(img_height, img_width, CV_32FC1);
//    m_hsv_theta_map.setTo(-1.f);
////    int *hbp;
//    float *htp;
//    for(int iy=big_rect.YMin(); iy < big_rect.YMax(); ++iy)
//    {
//        p = local_hsv_img.ptr<uchar>(iy);
//        hbp = m_hsv_bin_map.ptr<int>(iy);
//        htp = m_hsv_theta_map.ptr<float>(iy);
//        for(int ix=big_rect.XMin(); ix < big_rect.XMax(); ++ix)
//        {
//            cv::Vec3b pixel(p[3*ix+0], p[3*ix+1], p[3*ix+2]);
////            hbp[ix] = m_hsv_feature.compBinIdx(pixel);
//            m_hsv_feature.compWBinIdx(pixel, hbp[ix], htp[ix]);
//        }
//    }

//    // default implementation
//    auto start = std::clock();
//    featVecs.resize(s.GetRects().size());
//    for (int i = 0; i < (int)featVecs.size(); ++i)
//    {
////        featVecs[i] = Eval(s.GetSample(i));
//        featVecs[i] = Features::Eval(s.GetSample(i));
//    }

//    std::cout<<"feature time "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;
//    std::cout<<"==============================================================="<<std::endl;
//}



