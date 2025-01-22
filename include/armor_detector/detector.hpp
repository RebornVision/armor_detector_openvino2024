// Copyright 2022 Chen Jun
// Licensed under the MIT License.

#ifndef ARMOR_DETECTOR__DETECTOR_HPP_
#define ARMOR_DETECTOR__DETECTOR_HPP_

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>

// STD
#include <cmath>
#include <string>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/number_classifier.hpp"
#include "auto_aim_interfaces/msg/debug_armors.hpp"
#include "auto_aim_interfaces/msg/debug_lights.hpp"


// std::call_once所需的头文件
#include <algorithm>
#include <iostream>
#include <mutex>
#include <opencv2/opencv.hpp>     //opencv header file
#include <openvino/openvino.hpp>  //openvino header file
#include <thread>

namespace rm_auto_aim
{

static constexpr int MODUL_INPUT_W = 416;    // 模型图像的宽度
static constexpr int MODUL_INPUT_H = 416;    // 模型图像的高度

static constexpr float BBOX_CONF_THRESH = 0.72;  // 边界框的置信度阈值
static constexpr float NMS_THRESH = 0.3;  // 非极大值抑制（NMS）的阈值

static constexpr int class_num = 2; // 类别数
static const std::vector<std::string> class_names = {"blue","red"}; // 类别名称 


static const ov::element::Type input_type = ov::element::u8;
static const ov::Layout input_layout{"NHWC"};
static const ov::preprocess::ColorFormat input_ColorFormat = ov::preprocess::ColorFormat::BGR;

static const ov::element::Type Moudel_type = ov::element::f32;
static const ov::Shape Moudel_shape = {1, 3, MODUL_INPUT_H, MODUL_INPUT_W};
static const ov::Layout Moudel_layout{"NCHW"};
static const ov::preprocess::ColorFormat Moudel_ColorFormat = ov::preprocess::ColorFormat::RGB;

class ArmorDetector
{
public:
    void init(const size_t &input_w,const size_t &input_h,const std::string MODEL_PATH);
    void startInferAndNMS(cv::Mat& img, int detect_color);
    std::vector<OneArmor>& get_armor();
    void clear_armor();
private:
    std::vector<OneArmor> last_Armors;
    ov::InferRequest infer_request;
    size_t INPUT_W;
    size_t INPUT_H;
    ov::Shape input_shape;
    float scale;
};

class Detector
{
public:
  struct ArmorParams
  {
    double min_light_ratio;

    // 小目标中心点间的最小和最大距离
    double min_small_center_distance;
    double max_small_center_distance;
    // 大目标中心点间的最小和最大距离
    double min_large_center_distance;
    double max_large_center_distance;
    // 水平角度的最大值，用于限制检测范围
    double max_angle;
    long int binary_threshold_light;
    
  };

  Detector(std::string model_path_yolo,const int & color, const ArmorParams & a);

  std::vector<Armor> detect(cv::Mat & input);
  std::vector<Armor> getArmors(const std::vector<OneArmor> & armors_data);
  cv::Mat outputImage;

  // For debug usage
  cv::Mat getAllNumbersImage();
  void drawResults(cv::Mat & img);

  std::string MODEL_PATH;
  int detect_color;
  
  ArmorParams a;

  std::unique_ptr<NumberClassifier> classifier;

private:
  ArmorDetector armor_detector;
  ArmorType isArmor(const OneArmor & armor_data) const;
  std::vector<Armor> armors_;
};

}  // namespace rm_auto_aim

#endif  // ARMOR_DETECTOR__DETECTOR_HPP_
