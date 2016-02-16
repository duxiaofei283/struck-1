/* 
 * Struck: Structured Output Tracking with Kernels
 * 
 * Code to accompany the paper:
 *   Struck: Structured Output Tracking with Kernels
 *   Sam Hare, Amir Saffari, Philip H. S. Torr
 *   International Conference on Computer Vision (ICCV), 2011
 * 
 * Copyright (C) 2011 Sam Hare, Oxford Brookes University, Oxford, UK
 * 
 * This file is part of Struck.
 * 
 * Struck is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * Struck is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with Struck.  If not, see <http://www.gnu.org/licenses/>.
 * 
 */

#ifndef TRACKER_H
#define TRACKER_H

#include "Rect.h"

#include <vector>
#include <Eigen/Core>
#include <opencv/cv.h>

class Config;
class Features;
class Kernel;
class LaRank;
class ImageRep;

class Tracker
{
public:
	Tracker(const Config& conf);
	~Tracker();
	
	void Initialise(const cv::Mat& frame, FloatRect bb);
	void Reset();
	void Track(const cv::Mat& frame);
	void Debug();

    void genMultiScaleSample(const ImageRep& img, const FloatRect& rect, std::vector<FloatRect>& scale_rects);

	
	inline const FloatRect& GetBB() const { return m_bb; }
	inline bool IsInitialised() const { return m_initialised; }
	
private:
	const Config& m_config;
	bool m_initialised;
	std::vector<Features*> m_features;
	std::vector<Kernel*> m_kernels;
	LaRank* m_pLearner;
	FloatRect m_bb;
    FloatRect m_initbb;
	cv::Mat m_debugImage;
	bool m_needsIntegralImage;
	bool m_needsIntegralHist;
    bool m_needColor;
    bool m_needsIntegralHsv;
	
    float m_scale;
    std::vector<float> m_scales;
    int m_scaleNum;

	void UpdateLearner(const ImageRep& image);
	void UpdateDebugImage(const std::vector<FloatRect>& samples, const FloatRect& centre, const std::vector<double>& scores);
};

#endif
