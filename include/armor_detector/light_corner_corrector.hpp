#ifndef ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_
#define ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_

// opencv
#include <opencv2/opencv.hpp>
#include "armor_detector/armor.hpp"
namespace rm_auto_aim
{
    struct SymmetryAxis {
        cv::Point2f centroid;
        cv::Point2f direction;
        float mean_val; // Mean brightness
    };

    // First, the PCA algorithm is used to find the symmetry axis of the light bar,
    // and then along the symmetry axis to find the corner points of the light bar based on the gradient of brightness.

    // Correct the corners of the armor's lights
    void correctCorners(OneArmor &armor, const cv::Mat &gray_img);

    // Find the symmetry axis of the light
     SymmetryAxis findSymmetryAxis(const cv::Mat &gray_img, const Light &light);

    // Find the corner of the light
     cv::Point2f findCorner(const cv::Mat &gray_img,
                           const Light &light,
                           const SymmetryAxis &axis,
                           std::string order);

}

#endif  // ARMOR_DETECTOR_LIGHT_CORNER_CORRECTOR_HPP_