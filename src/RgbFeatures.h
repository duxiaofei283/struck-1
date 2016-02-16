#ifndef RGB_FEATURES_H
#define RGB_FEATURES_H

#include "Features.h"

class Config;

enum colorName {R,G,B};

struct rgbIndice
{
    int r_idx;
    int g_idx;
    int b_idx;
};

class RgbFeatures : public Features
{
public:
    RgbFeatures(const Config& conf);
//    RgbFeatures();

    int compBinIdx(const uchar &intensity, const colorName &cn) const;
    rgbIndice& compBinIdx(const cv::Vec3b &pixel) const;
    int getBinNum(const colorName &cn) const;

private:

    float m_r_step;
    float m_g_step;
    float m_b_step;
    virtual void UpdateFeatureVector(const Sample& s);

};

#endif
