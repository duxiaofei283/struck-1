#ifndef MUTILS_H
#define MUTILS_H

#include <math.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <map>
#include <opencv2/opencv.hpp>
#include "Rect.h"

float getDist(const cv::Point &pt1, const cv::Point &pt2);
float getAngle(const cv::Point &pt1, const cv::Point &pt2);
void getDistAndAngle(const cv::Point &pt, const cv::Point &center, float& dist, float &angle);
std::vector<std::pair<float, unsigned int> > getVecHistogram(std::vector<float> vec);

void computeRingImg(const cv::Point &center, int min_radius, int max_radius, float weight, cv::Mat& img);

void computeRingImg2(const cv::Point &center, int radius, int thickness, float weight, cv::Mat &img);

void computeRingImg3(const cv::Point &center, int min_radius, int max_radius, float weight, cv::Mat &img, const int max_width, const int max_height);

cv::Mat colorMap(const cv::Mat &img);

void deinterlace(cv::Mat& m);

void findMultiMaxVals(const cv::Mat &img, int &max_val_num, double &max_val, cv::Point &max_idx);

int mostCommon(const std::vector<int>& v);

void multiDelMatRow(cv::Mat& m, std::vector<int> &delete_indices);
void multiDelMatCol(cv::Mat& m, std::vector<int> &delete_indices);

cv::Point getCirclePt(const cv::Point &center, float dist, float angle);

std::string zeroPadding(int zero_num, int idx);

float mod(float a, float b);

cv::Rect getRoi(const cv::Point& center, const float width, const float height, const float max_width, const float max_height);



template<typename T, typename A>
void multiDeleteVec(std::vector<T, A> &v, std::vector<int> &delete_indices)
{
    if (delete_indices.empty())
        return;

    std::sort(delete_indices.begin(), delete_indices.end(), std::greater<int>());
    for(auto it=delete_indices.begin(); it!=delete_indices.end(); ++it)
        v.erase(v.begin() + *it);
}

const std::vector<float> estPrecision(const std::vector<FloatRect>& result, const std::vector<FloatRect>& gt);

#endif // MUTILS_H
