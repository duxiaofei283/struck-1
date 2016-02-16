#define _USE_MATH_DEFINES

#include <math.h>
#include "mutils.h"


#define MAX_THRES 50


float getDist(const cv::Point& pt1, const cv::Point& pt2)
{
    cv::Point rel = pt1 - pt2;
    return sqrtf(rel.x*rel.x+rel.y*rel.y);
}

float getAngle(const cv::Point& pt1, const cv::Point& pt2)
{
    cv::Point rel = pt1 - pt2;
    float radian = std::atan2(rel.y, rel.x);
    float angle = 180.0 * radian / M_PI; // range(-180, 180)
    angle = mod(angle, 360); // convert range to (0, 360)
    return angle;
}

void getDistAndAngle(const cv::Point& pt, const cv::Point& center, float& dist, float& angle)
{
    cv::Point rel = pt - center;
    dist = sqrtf(rel.x*rel.x+rel.y*rel.y);
    float radian = std::atan2(rel.y, rel.x);
    angle = 180.0 * radian / M_PI;  // range(-180, 180)
    angle = mod(angle, 360);  // convert range to (0, 360)
}

std::vector<std::pair<float, unsigned int> > getVecHistogram(std::vector<float> vec)
{
    std::map<float, unsigned int> counts;
    for(int i=0; i<vec.size(); ++i)
    {
        std::map<float, unsigned int>::iterator it(counts.find(vec[i]));
        if(it != counts.end())
        {
            it->second++;
        }
        else
        {
            counts[vec[i]] = 1;
        }
    }

    std::vector<std::pair<float, unsigned int>> counts_vec;
    for (auto it=counts.begin(); it!=counts.end(); ++it)
        counts_vec.push_back(*it);

    std::sort(counts_vec.begin(), counts_vec.end(), [=](std::pair<float, unsigned int>& a, std::pair<float, unsigned int>& b){return a.second > b.second;});

    return counts_vec;
}

cv::Mat colorMap(const cv::Mat& img)
{
    double min_val, max_val;
    cv::minMaxIdx(img, &min_val, &max_val);
    float ratio = 255.0 / max_val;
    cv::Mat norm_img = cv::Mat(img.size(), CV_32F);
    norm_img = ratio * img;
    norm_img.convertTo(norm_img, CV_8UC1);

    cv::Mat col_img;
    cv::applyColorMap(norm_img, col_img, cv::COLORMAP_JET);
    return col_img;
}


void computeRingImg(const cv::Point& center, int min_radius, int max_radius, float weight, cv::Mat &img)
{
    cv::Mat inner_mask = cv::Mat::zeros(img.size(), CV_8U);
    cv::Mat outer_mask = cv::Mat::zeros(img.size(), CV_8U);

    cv::circle(inner_mask, center, min_radius, cv::Scalar(1,0,0), -1);
    cv::circle(outer_mask, center, max_radius, cv::Scalar(1,0,0), -1);
    cv::Mat mask;
    cv::bitwise_xor(inner_mask, outer_mask, mask);

    mask.convertTo(mask, CV_32F);
    mask *= weight;

    img += mask;
}

void computeRingImg2(const cv::Point &center, int radius, int thickness, float weight, cv::Mat &img)
{
//    auto start = std::clock();
    cv::Mat mask = cv::Mat::zeros(img.size(), CV_32F);
    cv::circle(mask, center, radius, cv::Scalar(weight, 0, 0), thickness);
    img += mask;
//    std::cout<<" computeRingImg2 "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;
}

void computeRingImg3(const cv::Point &center, int min_radius, int max_radius, float weight, cv::Mat &img, const int max_width, const int max_height)
{
//    auto start = std::clock();
    // outer circle
    cv::Point outer_tl = cv::Point(std::max(0, static_cast<int>(center.x - max_radius)),
                                   std::max(0, static_cast<int>(center.y - max_radius)));
    cv::Point outer_br = cv::Point(std::min(max_width-1, static_cast<int>(center.x + max_radius)),
                                   std::min(max_height-1, static_cast<int>(center.y + max_radius)));
    // inner circle
    cv::Rect inner_roi = getRoi(center, min_radius*std::sinf(M_PI_4)*2, min_radius*std::sinf(M_PI_4)*2, max_width, max_height);

    for(int row = outer_tl.y; row<=outer_br.y; ++row)
        for(int col = outer_tl.x; col<=outer_br.x; ++col)
        {
            if(inner_roi.contains(cv::Point(col, row)))
                continue;
            if((std::pow(row-center.y,2) + std::pow(col-center.x,2) >= std::pow(min_radius,2))  &&
               (std::pow(row-center.y,2) + std::pow(col-center.x,2) <= std::pow(max_radius,2)))
                img.at<float>(row, col) += weight;
        }

//    std::cout<<" computeRingImg3 "<<(std::clock()-start)/(double)CLOCKS_PER_SEC<<std::endl;

}


void findMultiMaxVals(const cv::Mat &img, int& max_val_num, double& max_val, cv::Point& max_idx)
{
    double min_val;
    cv::minMaxLoc(img, &min_val, &max_val, 0, &max_idx);

    std::vector<cv::Point> max_indices;
    for(int row=0; row<img.rows; ++row)
        for(int col=0; col<img.cols; ++col)
            if(img.at<float>(row, col) == max_val)
                max_indices.push_back(cv::Point(col, row));
    max_val_num = max_indices.size();
    max_idx = max_indices.at(0);
}


int mostCommon(const std::vector<int> &v)
{
    int max = 0;
    int most_common = -1;
    std::map<int, int> counts;
    for(auto i_v=v.begin(); i_v != v.end(); ++i_v)
    {
        counts[*i_v]++;
        if(counts[*i_v] > max)
        {
            max = counts[*i_v];
            most_common = *i_v;
        }
    }

    return most_common;
}

void multiDelMatRow(cv::Mat &m, std::vector<int> &delete_indices)
{
    if (delete_indices.empty())
        return;

    std::sort(delete_indices.begin(), delete_indices.end());
    cv::Mat f_m = cv::Mat(m.rows-delete_indices.size(), m.cols, m.type());
    auto i_d = delete_indices.begin();
    int f_idx = 0;
    for(int m_idx=0; m_idx<m.rows; ++m_idx)
        if(m_idx != *i_d || i_d == delete_indices.end())
            m.row(m_idx).copyTo(f_m.row(f_idx++));
        else
            i_d++;

    f_m.copyTo(m);
}


void multiDelMatCol(cv::Mat &m, std::vector<int> &delete_indices)
{
    if (delete_indices.empty())
        return;
    std::sort(delete_indices.begin(), delete_indices.end());
    cv::Mat f_m = cv::Mat(m.rows, m.cols - delete_indices.size(), m.type());
    auto i_d = delete_indices.begin();
    int f_idx = 0;
    for(int m_idx=0; m_idx<m.cols; ++m_idx)
        if(m_idx != *i_d || i_d == delete_indices.end())
            m.col(m_idx).copyTo(f_m.col(f_idx++));
        else
            i_d++;

    f_m.copyTo(m);
}


cv::Point getCirclePt(const cv::Point &center, float dist, float angle)
{
    if (angle > 180)
        angle -= 360;
    float radian = angle * M_PI / 180.0;
    float rel_x = std::cosf(radian) * dist;
    float rel_y = std::sinf(radian) * dist;
    cv::Point pt = center + cv::Point(rel_x, rel_y);
    return pt;
}


std::string zeroPadding(int zero_num, int idx)
{
    std::stringstream ss;
    ss << std::setw(zero_num) << std::setfill('0') << idx;
    std::string s = ss.str();
    return s;
}

void deinterlace(cv::Mat &m)
{
    for(int i=0; i<m.rows; ++i)
        if( i % 2 == 1)
            m.row(i-1).copyTo(m.row(i));
}


float mod(float a, float b)
{
    float ret =fmodf(a, b);
    if(ret < 0)
        ret += b;
    return ret;
}


cv::Rect getRoi(const cv::Point &center, const float width, const float height, const float max_width, const float max_height)
{
    cv::Point tl = cv::Point(std::max(0, static_cast<int>(center.x - width/2)),
                             std::max(0, static_cast<int>(center.y - height/2))) ;
    cv::Point br = cv::Point(std::min(static_cast<int>(max_width-1), static_cast<int>(center.x + width/2)),
                             std::min(static_cast<int>(max_height-1), static_cast<int>(center.y + height/2)));
    cv::Rect roi = cv::Rect(tl, br);
    return roi;
}

const std::vector<float> estPrecision(const std::vector<FloatRect>& result, const std::vector<FloatRect>& gt)
{
    std::vector<float> precision;
    if (gt.empty())
    {
        std::cout<<"No ground truth for this sequence."<<std::endl;
        return precision;
    }
    else
    {
        std::cout<<"gt size:        "<<gt.size()<<std::endl;
        std::cout<<"result size:    "<<result.size()<<std::endl;
//        assert(result.size() == gt.size());
        if(result.size() != gt.size())
        {
            std::cout<<"Assertion failed: (result.size() == gt.size())."<<std::endl;
            return precision;
        }

        // calculate L2 error distance
        std::vector<float> error;
        for(int i=0; i<result.size(); ++i)
        {
            std::cout<<"result center: "<<result[i].XCentre()<<" "<<result[i].YCentre()<<std::endl;
            std::cout<<"gt center    : "<<gt[i].XCentre()<<" "<<gt[i].YCentre()<<std::endl;
            float err = std::sqrtf( std::pow(result[i].XCentre()-gt[i].XCentre(), 2) + std::pow(result[i].YCentre()-gt[i].YCentre(), 2) );
            error.push_back(err);
        }

        // compute presicions
        int pos_frame_num;
        std::vector<float>::iterator it;
        for(int thres=0; thres<MAX_THRES; ++thres)
        {
            pos_frame_num = 0;
            for(it=error.begin(); it != error.end(); it++)
            {
                if(*it <= thres)
                    pos_frame_num++;
            }
            precision.push_back(1.0 * pos_frame_num / error.size());
        }

        // plot the precisions
        // todo
        return precision;
    }

}




