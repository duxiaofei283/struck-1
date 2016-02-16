#ifndef HSV_FEATURES_H
#define HSV_FEATURES_H

#include "Features.h"

class Config;

class HsvFeatures : public Features
{
public:
    HsvFeatures(const Config& conf);
//    HsvFeatures();

    int compBinIdx(const cv::Vec3b& pixel) const;
    void compWBinIdx(const cv::Vec3b& pixel, int& bin_idx, float& weight);

private:

    float m_h_step;
    float m_s_step;
    float m_v_step;
    virtual void UpdateFeatureVector(const Sample& s);

};

#endif
