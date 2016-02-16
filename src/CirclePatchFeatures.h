#ifndef CIRCLE_PATCH_FEATURES_H
#define CIRCLE_PATCH_FEATURES_H

#include "Features.h"
#include "HsvFeatures.h"
#include "gradFeatures.h"
class Config;

class CirclePatchFeatures : public Features
{
public:
    CirclePatchFeatures(const Config &conf);
    void compDistMap(const cv::Size &patch_size);
    void compOriMap(const cv::Size &patch_size);
    virtual void Eval(const MultiSample &s, std::vector<Eigen::VectorXd> &featVecs);

private:
    int m_miniPatch_num;
    int m_fea_size;
    int m_hsv_num;
    int m_half_localPatch;
    cv::Size m_distmap_size;
    cv::Size m_orimap_size;
    HsvFeatures m_hsv_feature;
    std::vector<Eigen::VectorXd> m_patch_featVec;
    std::vector<Eigen::VectorXd> m_circ_featVec;



    cv::Mat m_dist_map;
    cv::Mat m_ori_map;
    float m_base_rotation;
    cv::Mat m_hsv_bin_map;
    cv::Mat m_hsv_theta_map;

    virtual void UpdateFeatureVector(const Sample& s);


};

#endif
