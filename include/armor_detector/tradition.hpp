//
// Created by gx on 24-4-28.
//
/*
 * 此文件编写方向：
 *      在神经网络推理的矩形框内得到灯条的四个角点
 *      采用覆盖原神经网络四个角点的过程
 */
#ifndef V8_INFERENCE_H_
#define V8_INFERENCE_H_

#include "armor_detector/detector.hpp"
namespace rm_auto_aim{

//const int binary_thres=120;

// const int RED = 0;
// const int BLUE = 1;

//LightParams


// width / height
const double min_ratio=0.1;
const double max_ratio=0.7;
// vertical angle
const double max_angle=45.0;


//functions
cv::Mat preprocessROI(const cv::Mat& img,cv::Rect armor_box,int binary_threshold_light);
bool isLight(const Light &light);
std::vector<Light> findLight(const cv::Mat& rgb_img, const cv::Mat &binary_img);
std::vector<Light> sortLight(std::vector<Light> &lights);
std::vector<OneArmor> tradition(cv::Mat input_img,std::vector<OneArmor>& armors_data,int binary_threshold_light);
}



#endif //V8_INFERENCE_H