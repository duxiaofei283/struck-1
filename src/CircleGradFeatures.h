#ifndef CIRCLE_GRAD_FEATURES_H
#define CIRCLE_GRAD_FEATURES_H

#include "Features.h"
#include "GradFeatures.h"
class Config;

class CircleGradFeatures : public Features
{
public:
    CircleGradFeatures(const Config& conf);

    inline const cv::Mat& getFeatureMap() const { return m_feature_map; }
    inline const cv::Mat& getDistMap() const {return m_dist_map; }

    void compDistMap(const cv::Size &patch_size);
    void compOriMap(const cv::Size &patch_size);
private:

    virtual void UpdateFeatureVector(const Sample& s);

    GradFeatures m_grad_feature;
    cv::Mat m_feature_map;
    cv::Mat m_dist_map;
    cv::Size m_distmap_size;
    cv::Mat m_ori_map;
    cv::Size m_orimap_size;

    int m_grad_num;
    float m_dist_step;


};

#endif

