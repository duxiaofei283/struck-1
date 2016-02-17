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

#include "Tracker.h"
#include "Config.h"
#include "ImageRep.h"
#include "Sampler.h"
#include "Sample.h"
#include "GraphUtils/GraphUtils.h"

#include "HaarFeatures.h"
#include "RawFeatures.h"
#include "HistogramFeatures.h"
#include "MultiFeatures.h"
#include "HsvFeatures.h"
#include "CircleFeatures.h"
#include "CircleGradFeatures.h"
#include "CircleRgbFeatures.h"
#include "CirclePatchFeatures.h"

#include "Kernels.h"

#include "LaRank.h"
#include "mutils.h"

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

#include <Eigen/Core>

#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;
using namespace Eigen;

static const float kRotOffset = M_PI/6;
static const float kNumHalfRot = 1;

Tracker::Tracker(const Config& conf) :
	m_config(conf),
	m_initialised(false),
	m_pLearner(0),
	m_debugImage(2*conf.searchRadius+1, 2*conf.searchRadius+1, CV_32FC1),
	m_needsIntegralImage(false)
{
	Reset();
}

Tracker::~Tracker()
{
	delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
}

void Tracker::Reset()
{
	m_initialised = false;
	m_debugImage.setTo(0);
	if (m_pLearner) delete m_pLearner;
	for (int i = 0; i < (int)m_features.size(); ++i)
	{
		delete m_features[i];
		delete m_kernels[i];
	}
	m_features.clear();
	m_kernels.clear();
	
	m_needsIntegralImage = false;
	m_needsIntegralHist = false;
    m_needColor = false;
    m_needsIntegralHsv = false;

    // scale
    m_scale = 1.0f;
    m_scaleNum = 1;
    for(int i=0; i<m_scaleNum; ++i)
    {
        m_scales.push_back(pow(1.01, i-m_scaleNum/2));
    }

    // rotation
    m_rotation = 0.f;
    m_rotHalfNum = kNumHalfRot;
    for(int i=m_rotHalfNum; i >= 1; --i)
        m_rotations.push_back(mod(-i*kRotOffset/m_rotHalfNum, 2*M_PI));
    m_rotations.push_back(0.f);
    for(int i=1; i<=m_rotHalfNum; ++i)
        m_rotations.push_back(mod(i * kRotOffset/m_rotHalfNum, 2*M_PI));

	
	int numFeatures = m_config.features.size();
	vector<int> featureCounts;
	for (int i = 0; i < numFeatures; ++i)
	{
		switch (m_config.features[i].feature)
		{
		case Config::kFeatureTypeHaar:
			m_features.push_back(new HaarFeatures(m_config));
			m_needsIntegralImage = true;
			break;			
		case Config::kFeatureTypeRaw:
			m_features.push_back(new RawFeatures(m_config));
			break;
		case Config::kFeatureTypeHistogram:
			m_features.push_back(new HistogramFeatures(m_config));
			m_needsIntegralHist = true;
			break;
        case Config::kFeatureTypeHsv:
            m_features.push_back(new HsvFeatures(m_config));
            m_needColor = true;
            break;
        case Config::kFeatureTypeCircleHsv:
            m_features.push_back(new CircleFeatures(m_config));
            m_needsIntegralHsv = true;
            m_needColor = true;
            break;
        case Config::kFeatureTypeCircleGrad:
            m_features.push_back(new CircleGradFeatures(m_config));
            break;
        case Config::kFeatureTypeCircleRgb:
            m_features.push_back(new CircleRgbFeatures(m_config));
            m_needsIntegralImage = true;
            m_needColor = true;
            break;
        case Config::kFeatureTypeCirclePatch:
            m_features.push_back(new CirclePatchFeatures(m_config));
            m_needsIntegralHsv = true;
            m_needColor = true;
            break;
        default:
            return;
		}
		featureCounts.push_back(m_features.back()->GetCount());
		
		switch (m_config.features[i].kernel)
		{
		case Config::kKernelTypeLinear:
			m_kernels.push_back(new LinearKernel());
			break;
		case Config::kKernelTypeGaussian:
			m_kernels.push_back(new GaussianKernel(m_config.features[i].params[0]));
			break;
		case Config::kKernelTypeIntersection:
			m_kernels.push_back(new IntersectionKernel());
			break;
		case Config::kKernelTypeChi2:
			m_kernels.push_back(new Chi2Kernel());
			break;
		}
	}
	
	if (numFeatures > 1)
	{
		MultiFeatures* f = new MultiFeatures(m_features);
		m_features.push_back(f);
		
		MultiKernel* k = new MultiKernel(m_kernels, featureCounts);
		m_kernels.push_back(k);		
	}
	
	m_pLearner = new LaRank(m_config, *m_features.back(), *m_kernels.back());
}
	

void Tracker::Initialise(const cv::Mat& frame, FloatRect bb)
{
	m_bb = IntRect(bb);
    m_initbb = IntRect(bb);
    ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHsv, m_needsIntegralHist, m_needColor);
	for (int i = 0; i < 1; ++i)
	{
		UpdateLearner(image);
	}
	m_initialised = true;
}

void Tracker::Track(const cv::Mat& frame)
{
	assert(m_initialised);
	
    ImageRep image(frame, m_needsIntegralImage, m_needsIntegralHsv, m_needsIntegralHist, m_needColor);
	
    vector<FloatRect> rects = Sampler::PixelSamples(m_bb, m_config.searchRadius * m_scale);
	
	vector<FloatRect> keptRects;
	keptRects.reserve(rects.size());
    std::vector<float> rots;
	for (int i = 0; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
        rots.push_back(m_rotation);
	}
	
    MultiSample sample(image, keptRects, rots);
	
	vector<double> scores;
	m_pLearner->Eval(sample, scores);
	
	double bestScore = -DBL_MAX;
	int bestInd = -1;
	for (int i = 0; i < (int)keptRects.size(); ++i)
	{		
		if (scores[i] > bestScore)
		{
			bestScore = scores[i];
			bestInd = i;
		}
	}

    UpdateDebugImage(keptRects, m_bb, scores);

//    //---- scale the best rect ----------------------------------
//    std::vector<FloatRect> scaleRects;
//    genMultiScaleSample(image, keptRects[bestInd], scaleRects);
//    MultiSample scaleSample(image, scaleRects);
//    m_pLearner->Eval(scaleSample, scores);

//    bestScore = -DBL_MAX;
//    bestInd = -1;
//    for(int i=0; i< (int)scaleRects.size(); ++i)
//    {
//        if(scores[i] > bestScore)
//        {
//            bestScore = scores[i];
//            bestInd = i;
//        }
//    }
//    //-----------------------------------------------------------

    //--- rotate the best rect ---------------------------------
    std::vector<FloatRect> rotRects;
    rots.clear();
    float r;
    for(int i=0; i<(int)m_rotations.size(); ++i)
    {
        r = mod(m_rotation + m_rotations[i], 2*M_PI);
        rots.push_back(r);
        rotRects.push_back(keptRects[bestInd]);
    }
    MultiSample rotSample(image, rotRects, rots);
    m_pLearner->Eval(rotSample, scores);
    bestScore = -DBL_MAX;
    bestInd = -1;
    for(int i=0; i< (int)rotRects.size(); ++i)
    {
        if(scores[i] > bestScore)
        {
            bestScore = scores[i];
            bestInd = i;
        }
    }
    //-----------------------------------------------------------


    if (bestInd != -1)
    {
//        m_bb = keptRects[bestInd];
//        m_bb = scaleRects[bestInd];
        m_bb = rotRects[bestInd];
        m_scale = std::max(1.0 * m_bb.Width()/ m_initbb.Width(), 1.0 * m_bb.Height()/ m_initbb.Height());
        m_rotation = rots[bestInd];
//        std::cout<<"tracked rect size:  [ "<< m_bb.Width() << "/" << m_initbb.Width()<<" , "<< m_bb.Height() <<"/" << m_initbb.Height()<< " ]"<<std::endl;
        std::cout <<"search radius:      "<< m_config.searchRadius * m_scale << std::endl;
        std::cout <<"rotation:  "<< 180 * m_rotation / M_PI << std::endl;
        UpdateLearner(image);
#if VERBOSE
        cout << "track score: " << bestScore << endl;
#endif
	}
}

void Tracker::UpdateDebugImage(const vector<FloatRect>& samples, const FloatRect& centre, const vector<double>& scores)
{
	double mn = VectorXd::Map(&scores[0], scores.size()).minCoeff();
	double mx = VectorXd::Map(&scores[0], scores.size()).maxCoeff();
	m_debugImage.setTo(0);
	for (int i = 0; i < (int)samples.size(); ++i)
	{
		int x = (int)(samples[i].XMin() - centre.XMin());
		int y = (int)(samples[i].YMin() - centre.YMin());
		m_debugImage.at<float>(m_config.searchRadius+y, m_config.searchRadius+x) = (float)((scores[i]-mn)/(mx-mn));
	}
}

void Tracker::Debug()
{
	imshow("tracker", m_debugImage);
    m_pLearner->Debug();
}

void Tracker::genMultiScaleSample(const ImageRep &img, const FloatRect &rect, std::vector<FloatRect> &scale_rects)
{
    float w  = rect.Width();
    float h = rect.Height();
    float xc = rect.XCentre();
    float yc = rect.YCentre();

    FloatRect r;
    for(int i = 0; i< m_scaleNum; ++i)
//        for(int j=0; j< m_scaleNum; ++j)
    {
        r.SetWidth(w * m_scales[i]);
        r.SetHeight(h * m_scales[i]);
        r.SetXMin(xc - r.Width()/2);
        r.SetYMin(yc - r.Height()/2);
        if(!r.IsInside(img.GetRect()) || r.Width() < m_initbb.Width() * 0.2 || r.Height() < m_initbb.Height() * 0.2)
            continue;
        scale_rects.push_back(r);
    }
}

//
void Tracker::UpdateLearner(const ImageRep& image)
{
	// note these return the centre sample at index 0
    vector<FloatRect> rects = Sampler::RadialSamples(m_bb, 2*m_config.searchRadius*m_scale, 5, 16);
	//vector<FloatRect> rects = Sampler::PixelSamples(m_bb, 2*m_config.searchRadius, true);
	
	vector<FloatRect> keptRects;
	keptRects.push_back(rects[0]); // the true sample
    vector<float> rots;
    rots.push_back(m_rotation);
	for (int i = 1; i < (int)rects.size(); ++i)
	{
		if (!rects[i].IsInside(image.GetRect())) continue;
		keptRects.push_back(rects[i]);
        rots.push_back(m_rotation);
	}
		
#if VERBOSE		
	cout << keptRects.size() << " samples" << endl;
#endif
		
    MultiSample sample(image, keptRects, rots);
	m_pLearner->Update(sample, 0);
}
