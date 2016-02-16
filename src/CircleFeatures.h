#ifndef CIRCLE_FEATURES_H
#define CIRCLE_FEATURES_H

#include "Features.h"
#include "HsvFeatures.h"
class Config;

class CircleFeatures : public Features
{
public:
    CircleFeatures(const Config& conf);

//    inline const cv::Mat& getFeatureMap() const { return m_feature_map; }
    inline const cv::Mat& getDistMap() const {return m_dist_map; }

    virtual void Eval(const MultiSample &s, std::vector<Eigen::VectorXd> &featVecs);

    void compDistMap(const cv::Size &patch_size);
    void compWDistMap(const cv::Size &patch_size);
    void compMiniPatchBinIdx(const std::vector<int>&hsv_idx_lst, const std::vector<float>&theta_lst, int& hsv_idx, float& theta);
private:

    virtual void UpdateFeatureVector(const Sample& s);

    HsvFeatures m_hsv_feature;
    cv::Mat m_hsv_bin_map;
    cv::Mat m_hsv_theta_map;
    int m_hsv_num;
//    cv::Mat m_feature_map;
    cv::Mat m_dist_map;
    cv::Mat m_dist_theta_map;
    cv::Size m_distmap_size;

    float m_dist_step;

    int m_half_miniPatch;
};

#endif

