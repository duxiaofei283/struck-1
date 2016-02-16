#ifndef CIRCLE_RGB_FEATURES_H
#define CIRCLE_RGB_FEATURES_H

#include "Features.h"
#include "RgbFeatures.h"
class Config;

class CircleRgbFeatures : public Features
{
public:
    CircleRgbFeatures(const Config& conf);

    inline const cv::Mat& getFeatureMap() const { return m_feature_map; }
    inline const cv::Mat& getDistMap() const {return m_dist_map; }

    void compDistMap(const cv::Size &patch_size);
private:

    virtual void UpdateFeatureVector(const Sample& s);

    RgbFeatures m_rgb_feature;
    int m_rgb_num;
    cv::Mat m_feature_map;
    cv::Mat m_dist_map;
    cv::Size m_distmap_size;

    float m_dist_step;

};

#endif

