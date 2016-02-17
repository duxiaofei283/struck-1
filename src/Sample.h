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

#ifndef SAMPLE_H
#define SAMPLE_H

#include "ImageRep.h"
#include "Rect.h"

#include <vector>

class Sample
{
public:
    Sample(const ImageRep& image, const FloatRect& roi, const float rotation=0) :
		m_image(image),
        m_roi(roi),
        m_rot(rotation)
	{
	}
	
	inline const ImageRep& GetImage() const { return m_image; }
	inline const FloatRect& GetROI() const { return m_roi; }
    inline const float GetRotation() const { return m_rot; }

private:
	const ImageRep& m_image;
	FloatRect m_roi;
    float m_rot;
};

class MultiSample
{
public:
    MultiSample(const ImageRep& image, const std::vector<FloatRect>& rects, const std::vector<float>& rotations) :
		m_image(image),
        m_rects(rects),
        m_rots(rotations)
	{
	}
	
	inline const ImageRep& GetImage() const { return m_image; }
	inline const std::vector<FloatRect>& GetRects() const { return m_rects; }
    inline const std::vector<float>& GetRotations() const { return m_rots;}
    inline Sample GetSample(int i) const { return Sample(m_image, m_rects[i], m_rots[i]); }

private:
	const ImageRep& m_image;
	std::vector<FloatRect> m_rects;
    std::vector<float> m_rots;
};

#endif
