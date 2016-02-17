#include "CirclePatchFeatures.h"
#include "Config.h"
#include "mutils.h"

static const int kNumDist = 8;
static const int kNumOrien = 8;
static const int kLocalSize = 3;

CirclePatchFeatures::CirclePatchFeatures(const Config &conf) : m_hsv_feature(conf)
{
    m_hsv_num = m_hsv_feature.GetCount();
    m_fea_size = m_hsv_num *kNumDist;
    m_miniPatch_num = kNumOrien * kNumDist;
    SetCount(m_fea_size);
    std::cout << "circle patch feature bins : "<<GetCount() << std::endl;

    m_distmap_size = cv::Size(-1, -1);
    m_orimap_size = cv::Size(-1, -1);

    m_half_localPatch = kLocalSize / 2;

    // assign mini patch feature
    for(int i=0; i<m_miniPatch_num; ++i)
    {
        m_patch_featVec.push_back(Eigen::VectorXd::Zero(m_hsv_num));
    }

}

void CirclePatchFeatures::UpdateFeatureVector(const Sample &s)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
    m_featVec.setZero();
    cv::Mat hsv_bin_map = m_hsv_bin_map(roi);
    cv::Mat hsv_theta_map = m_hsv_theta_map(roi);

    for(int i=0; i<m_miniPatch_num; ++i)
        m_patch_featVec[i].setZero();

    // compute dist and orientation map
    compDistMap(cv::Size(rect.Width(), rect.Height()));
    compOriMap(cv::Size(rect.Width(), rect.Height()), s.GetRotation());

    int height = rect.Height();
    int width = rect.Width();
    std::vector<float> pixelnum_dist_bin(kNumDist, 0);
    std::vector<float> pixelnum_patch(m_miniPatch_num, 0);
    // continuous?
    if(m_dist_map.isContinuous() && m_ori_map.isContinuous() &&
       hsv_bin_map.isContinuous() && hsv_theta_map.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy, hsv_bin_idx, ori_bin_idx;
    int patch_bin_idx, patch_fea_bin_idx;
    int dist_bin_idx, fea_bin_idx;
    int *dp, *hbp, *op;
    float *htp;
    float hsv_theta;
    for(iy = 0; iy < height; ++iy)
    {
        dp = m_dist_map.ptr<int>(iy);
        op = m_ori_map.ptr<int>(iy);
        hbp = hsv_bin_map.ptr<int>(iy);
        htp = hsv_theta_map.ptr<float>(iy);

        for(ix = 0; ix < width; ++ix)
        {
            dist_bin_idx = dp[ix];
            ori_bin_idx = op[ix];
            // plus (half theta) version
            hsv_bin_idx = hbp[ix];
            hsv_theta = htp[ix];

            // mini patch feature vector
            patch_bin_idx = dist_bin_idx * kNumOrien + ori_bin_idx;
            pixelnum_patch[patch_bin_idx] += hsv_theta;
            m_patch_featVec[patch_fea_bin_idx][hsv_bin_idx] += hsv_theta;

            // circle feature vector
            pixelnum_dist_bin[dist_bin_idx] += hsv_theta;
            fea_bin_idx = dist_bin_idx * m_hsv_num + hsv_bin_idx;
            m_featVec[fea_bin_idx] += hsv_theta;
        }
    }

    // normalize patch feature vector
    for(int id=0; id<m_miniPatch_num; ++id)
    {
        m_patch_featVec[id] /= pixelnum_patch[id] * m_miniPatch_num;
    }

    // normalize circle feature vector
    for(int id=0; id<kNumDist; ++id)
        for(int ic=0; ic<m_hsv_num; ++ic)
        {
            fea_bin_idx = id * m_hsv_num + ic;
            m_featVec[fea_bin_idx] /= pixelnum_dist_bin[id] * kNumDist;
        }

}

void CirclePatchFeatures::UpdatePatchFeatureVector(const Sample &s, Eigen::VectorXd& featVec)
{
    IntRect rect = s.GetROI();
    cv::Rect roi(rect.XMin(), rect.YMin(), rect.Width(), rect.Height());
    cv::Mat hsv_bin_map = m_hsv_bin_map(roi);
    cv::Mat hsv_theta_map = m_hsv_theta_map(roi);

    for(int i=0; i<m_miniPatch_num; ++i)
        m_patch_featVec[i].setZero();

    // compute dist and orientation map
    compDistMap(cv::Size(rect.Width(), rect.Height()));
    compOriMap(cv::Size(rect.Width(), rect.Height()), s.GetRotation());

    int height = rect.Height();
    int width = rect.Width();
    std::vector<float> pixelnum_patch(m_miniPatch_num, 0);
    // continuous?
    if(m_dist_map.isContinuous() && m_ori_map.isContinuous() &&
       hsv_bin_map.isContinuous() && hsv_theta_map.isContinuous())
    {
        width *= height;
        height = 1;
    }

    int ix, iy, hsv_bin_idx, ori_bin_idx;
    int patch_bin_idx, patch_fea_bin_idx;
    int dist_bin_idx, fea_bin_idx;
    int *dp, *hbp, *op;
    float *htp;
    float hsv_theta;
    for(iy = 0; iy < height; ++iy)
    {
        dp = m_dist_map.ptr<int>(iy);
        op = m_ori_map.ptr<int>(iy);
        hbp = hsv_bin_map.ptr<int>(iy);
        htp = hsv_theta_map.ptr<float>(iy);

        for(ix = 0; ix < width; ++ix)
        {
            dist_bin_idx = dp[ix];
            ori_bin_idx = op[ix];
            // plus (half theta) version
            hsv_bin_idx = hbp[ix];
            hsv_theta = htp[ix];

            // mini patch feature vector
            patch_bin_idx = dist_bin_idx * kNumOrien + ori_bin_idx;
            pixelnum_patch[patch_bin_idx] += hsv_theta;
            m_patch_featVec[patch_fea_bin_idx][hsv_bin_idx] += hsv_theta;
        }
    }

    // normalize patch feature vector
    for(int id=0; id<m_miniPatch_num; ++id)
    {
        m_patch_featVec[id] /= pixelnum_patch[id] * m_miniPatch_num;
    }

    // concatenate patch feature vector
    featVec = Eigen::VectorXd::Zero(m_hsv_num * m_miniPatch_num);
    for(int id=0; id < m_miniPatch_num; ++id)
        for(int ic=0; ic < m_hsv_num; ++ic)
        {
            fea_bin_idx = id * m_hsv_num + ic;
            featVec[fea_bin_idx] = m_patch_featVec[id][ic];
        }
}

void CirclePatchFeatures::compDistMap(const cv::Size &patch_size)
{
    if(m_distmap_size == patch_size)
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
        float dist_bin_step = max_dist / kNumDist;

        m_dist_map.create(m_distmap_size.height, m_distmap_size.width, CV_32SC1);

        int ix, iy;
        int* dp;
        cv::Point pt;
        for(iy=0; iy < m_distmap_size.height; ++iy)
        {
            dp = m_dist_map.ptr<int>(iy);

            for(ix=0; ix < m_distmap_size.width; ++ix)
            {
                pt = cv::Point(ix, iy);
                dp[ix] = int(getDist(pt, center) / (max_edge * dist_bin_step));
            }
        }

    }
}

void CirclePatchFeatures::compOriMap(const cv::Size &patch_size, const float base_rotation)
{
    if (m_orimap_size == patch_size)
        return;
    else
    {
        m_orimap_size = patch_size;
        cv::Point center = cv::Point(m_orimap_size.width/2, m_orimap_size.height/2);

        // compute orientation map
        m_ori_map.create(m_orimap_size.height, m_orimap_size.width, CV_32SC1);
        float grad_bin_step = 2 * M_PI / kNumOrien;

        int ix, iy;
        int* op;
        float rx, ry;
        float angle;
        for(iy=0; iy < m_orimap_size.height; ++iy)
        {
            op = m_ori_map.ptr<int>(iy);

            for(ix=0; ix < m_orimap_size.width; ++ix)
            {
                rx = ix - center.x;
                ry = iy - center.y;
                if(rx == 0)
                    rx += FLT_EPSILON;
                angle = std::atan2f(ry, rx) + M_PI - base_rotation;
                // the relative angle should be in range (0, 2pi)
                angle = (angle<0) ? angle+2*M_PI : angle;
                op[ix] = int(angle / grad_bin_step);
            }
        }
    }
}

void CirclePatchFeatures::Eval(const MultiSample &s, std::vector<Eigen::VectorXd> &featVecs)
{
    // prepare hsv map
    prepMap(s);

    // default implementation
    auto start = std::clock();
    featVecs.resize(s.GetRects().size());
    for (int i = 0; i < (int)featVecs.size(); ++i)
    {
        featVecs[i] = Features::Eval(s.GetSample(i));
    }
    std::cout<<"feature time "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;
    std::cout<<"==============================================================="<<std::endl;

}

void CirclePatchFeatures:: RotEval(const MultiSample &s, std::vector<Eigen::VectorXd>& featVecs)
{
    // prepare hsv map
    prepMap(s);

    featVecs.resize(s.GetRects().size());
    for (int i = 0; i < (int)featVecs.size(); ++i)
    {
        UpdatePatchFeatureVector(s.GetSample(i), featVecs[i]);
    }
    std::cout<<"rotate feature time "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;
    std::cout<<"==============================================================="<<std::endl;



}

void CirclePatchFeatures::prepMap(const MultiSample &s)
{
    ImageRep si = s.GetImage();
    int img_height = si.GetHsvImage().rows;
    int img_width = si.GetHsvImage().cols;
//    cv::Mat hsv_img = si.GetHsvImage();

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
    // get a pixel larger (updated)
    x_min = (x_min <= 1) ? 0 : (x_min-1);
    y_min = (y_min <= 1) ? 0 : (y_min-1);
    x_max = (x_max >= img_width-1) ? img_width : (x_max+1);
    y_max = (y_max >= img_height-1) ? img_height : (y_max+1);
    // ??? problem? float to int
    IntRect big_rect(x_min, y_min, x_max-x_min, y_max-y_min);

    m_hsv_bin_map = cv::Mat(img_height, img_width, CV_32SC1);
    m_hsv_bin_map.setTo(-1);
    m_hsv_theta_map = cv::Mat(img_height, img_width, CV_32FC1);
    m_hsv_theta_map.setTo(-1.f);
    int *hbp;
    float *htp;
    cv::Vec3b pixel;
    IntRect local_rect;
    for(int iy=big_rect.YMin(); iy < big_rect.YMax(); ++iy)
    {
        hbp = m_hsv_bin_map.ptr<int>(iy);
        htp = m_hsv_theta_map.ptr<float>(iy);

        for(int ix=big_rect.XMin(); ix < big_rect.XMax(); ++ix)
        {
            // get local patch rect
            x_min = (ix - m_half_localPatch <= 0) ? 0 : ix - m_half_localPatch;
            y_min = (iy - m_half_localPatch <= 0) ? 0 : iy - m_half_localPatch;
            x_max = (ix + m_half_localPatch + 1 >= img_width ) ? img_width  : ix + m_half_localPatch + 1;
            y_max = (iy + m_half_localPatch + 1 >= img_height) ? img_height : iy + m_half_localPatch + 1;
            local_rect = IntRect(x_min, y_min, x_max-x_min, y_max-y_min);
            for(int channel=0; channel<3; ++channel)
                pixel[channel] = uchar(si.MeanHsv(local_rect, channel));

            // compute the hsv bin index map
            m_hsv_feature.compWBinIdx(pixel, hbp[ix], htp[ix]);
        }
    }
}






