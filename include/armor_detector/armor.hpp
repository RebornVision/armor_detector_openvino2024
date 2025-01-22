// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__ARMOR_HPP_
#define ARMOR_DETECTOR__ARMOR_HPP_

#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
// STL
#include <algorithm>
#include <string>

namespace rm_auto_aim
{
    const int RED = 1;
    const int BLUE = 0;

    enum class ArmorType { SMALL, LARGE, INVALID };
    const std::string ARMOR_TYPE_STR[3] = {"small", "large", "invalid"};


    struct Light : public cv::RotatedRect
    {
        Light() = default;
        explicit Light(cv::RotatedRect box) : cv::RotatedRect(box)
        {
            cv::Point2f p[4];
            box.points(p);
            std::sort(p, p + 4, [](const cv::Point2f &a, const cv::Point2f &b)
            { return a.y < b.y; });
            top = (p[0] + p[1]) / 2;
            bottom = (p[2] + p[3]) / 2;

            length = cv::norm(top - bottom);
            width = cv::norm(p[0] - p[1]);

            // 归一化
            axis = top - bottom;
            axis = axis / cv::norm(axis);

            tilt_angle = std::atan2(std::abs(top.x - bottom.x), std::abs(top.y - bottom.y));
            tilt_angle = tilt_angle / CV_PI * 180;
        }

        int color;
        cv::Point2f top, bottom,center;
        cv::Point2f axis;
        double length;
        double width;
        float tilt_angle;
    };

    struct Armor
    {
        Armor() = default;
        ~Armor() = default;

        cv::Point tl,tr,bl,br;
        float modelconf;
        cv::Point center;
        ArmorType type;
        int color;
        // Number part
        cv::Mat number_img;
        std::string number;
        float confidence;
        std::string classfication_result;
    };



    struct OneClassArmor
    {
        std::vector<float> class_scores;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> objects_keypoints;
        int class_ids;
    };

    struct OneArmor
    {
        OneArmor() = default;
        OneArmor(const Light &l1, const Light &l2) {
            if (l1.center.x < l2.center.x) {
                left_light = l1, right_light = l2;
            } else {
                left_light = l2, right_light = l1;
            }

            center = (left_light.center + right_light.center) / 2;
        }

        // Light pairs part
        Light left_light, right_light;
        cv::Point2f center;

        float class_scores;
        cv::Rect box;
        cv::Point2f objects_keypoints[4];
        int class_ids;//color
    };

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__ARMOR_HPP_
