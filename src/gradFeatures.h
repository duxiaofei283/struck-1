#ifndef GRAD_FEATURES_H
#define GRAD_FEATURES_H

#include "Features.h"

class Config;

class GradFeatures : public Features
{
public:
    GradFeatures(const Config& conf);

    void compGrad(const cv::Mat& img, cv::Mat &ori, cv::Mat &mag) const;

    int compBinIdx(const float orientation) const;

private:

    float m_bin_step;
    virtual void UpdateFeatureVector(const Sample& s);

};

#endif
