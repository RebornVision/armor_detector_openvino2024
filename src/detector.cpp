// Copyright (c) 2022 ChenJun
// Licensed under the MIT License.

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <vector>

#include "armor_detector/detector.hpp"
#include "armor_detector/tradition.hpp"
#include "auto_aim_interfaces/msg/debug_armor.hpp"
#include "auto_aim_interfaces/msg/debug_light.hpp"

namespace rm_auto_aim
{
Detector::Detector( const std::string MODEL_PATH,const int & color, const ArmorParams & a) : MODEL_PATH(MODEL_PATH),detect_color(color), a(a) {}

std::vector<Armor> Detector::detect(cv::Mat & input)
{
  // 初始化模型和配置
  static bool is_first_frame = true;
  if (is_first_frame) {
      is_first_frame = false;
      armor_detector.init(input.cols, input.rows, MODEL_PATH);
  }
  armor_detector.startInferAndNMS(input, Detector::detect_color);
  //是否使用传统方法提高四点精度（需调节检测二值化阈值）
  std::vector<OneArmor> armors_data = tradition(input, armor_detector.get_armor(),a.binary_threshold_light);
  armors_ = getArmors(armors_data);//如果上面代码被注释，这里改为armors_data
  if (!armors_.empty()) {
    classifier->extractNumbers(input, armors_);
    classifier->classify(armors_);
  }
  armor_detector.clear_armor();
  return armors_;
}

std::vector<Armor> Detector::getArmors(const std::vector<OneArmor> & armors_data)
{
  std::vector<Armor> armors;
  // this->debug_armors.data.clear(); // 清空调试用的护甲数据
  for (const auto& i : armors_data) {
    Armor armor;
    armor.tl = i.objects_keypoints[0];
    armor.tr = i.objects_keypoints[3];
    armor.bl = i.objects_keypoints[1];
    armor.br = i.objects_keypoints[2];
    armor.color = i.class_ids;
    auto type = isArmor(i);
    armor.modelconf = i.class_scores;
    // 检测当前灯光配对是否构成护甲，并确定护甲类型
    armor.type = type;
    armor.center = (i.objects_keypoints[0] + i.objects_keypoints[1] + i.objects_keypoints[2] +
                    i.objects_keypoints[3]) /
                   4;
    armors.emplace_back(armor);
  }

  return armors;
}

ArmorType Detector::isArmor(const OneArmor & armor_data) const
{
  float avg_light_length =
    (cv::norm(armor_data.objects_keypoints[0] - armor_data.objects_keypoints[1]) +
     cv::norm(armor_data.objects_keypoints[2] - armor_data.objects_keypoints[3])) /
    2;
  float center_distance =
    cv::norm(armor_data.objects_keypoints[0] - armor_data.objects_keypoints[3]) / avg_light_length;
  ArmorType type;
  type = center_distance > a.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;

  return type;
}

/**
 * 获取所有护甲的数字图像拼接而成的图像。
 * 该函数遍历护甲列表，将每个护甲的数字图像垂直拼接起来，形成一个包含所有护甲数字图像的图像。
 * 如果护甲列表为空，则返回一个20x28的单通道黑色图像。
 *
 * @return cv::Mat 返回包含所有护甲数字图像的图像。如果护甲列表为空，返回一个20x28的黑色图像。
 */
cv::Mat Detector::getAllNumbersImage()
{
  // 如果护甲列表为空，直接返回一个20x28的单通道黑色图像
  if (armors_.empty()) {
    return cv::Mat(cv::Size(20, 28), CV_8UC1);
  } else {
    std::vector<cv::Mat> number_imgs;
    number_imgs.reserve(armors_.size());
    for (auto & armor : armors_) {
      number_imgs.emplace_back(armor.number_img);
    }
    cv::Mat all_num_img;
    // 将所有数字图像垂直拼接起来
    cv::vconcat(number_imgs, all_num_img);
    return all_num_img;
  }
}

void Detector::drawResults(cv::Mat & img)
{
  // Draw armors
  for (const auto & armor : armors_) {
    cv::circle(img, armor.tl, 3, cv::Scalar(0, 255, 0), 1);
    cv::circle(img, armor.tr, 3, cv::Scalar(0, 255, 0), 1);
    cv::circle(img, armor.bl, 3, cv::Scalar(0, 255, 0), 1);
    cv::circle(img, armor.bl, 3, cv::Scalar(0, 255, 0), 1);
    switch (armor.color) {
      case 0:
        cv::line(img, armor.tl, armor.tr, cv::Scalar(0, 0, 255), 2);
        cv::line(img, armor.bl, armor.br, cv::Scalar(0, 0, 255), 2);
        break;
      case 1:
        cv::line(img, armor.tl, armor.tr, cv::Scalar(255, 0, 0), 2);
        cv::line(img, armor.bl, armor.br, cv::Scalar(255, 0, 0), 2);
        break;
    }
  }

  std::stringstream result_model;

  // Show numbers and confidence
  for (const auto & armor : armors_) {
    // result_model
    //   << "model time: " << std::fixed
    //   << std::setprecision(
    //        1)  // std::fixed和std::set precision(1)用于设置输出浮点数时保留一位小数，并以固定格式显示
    //   << armor.modelconf * 100.0 << "%";
    cv::putText(
      img, armor.classfication_result, armor.tl, cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);
    cv::putText(
            img, result_model.str(), armor.tr, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255),
            2);
  }
}
}  // namespace rm_auto_aim
