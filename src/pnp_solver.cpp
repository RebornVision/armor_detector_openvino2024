// Copyright 2022 Chen Jun

#include "armor_detector/pnp_solver.hpp"

#include <opencv2/calib3d.hpp>
#include <vector>

namespace rm_auto_aim
{

/**
 * PnPSolver 类构造函数
 * 
 * 用于初始化PnPSolver类的一个实例，包括相机内参矩阵和畸变参数。
 * 
 * @param camera_matrix 描述相机内参的3x3矩阵，单位为米。
 * @param dist_coeffs 描述相机畸变的参数向量，通常包括k1, k2, p1, p2, k3等5个参数。
 */
PnPSolver::PnPSolver(
  const std::array<double, 9> & camera_matrix, const std::vector<double> & dist_coeffs)
: camera_matrix_(cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data())).clone()),
  dist_coeffs_(cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data())).clone())
{
  // 定义小号和大号护甲板的三维点，单位转换为米，并按照顺时针顺序从底部左角开始排列
  constexpr double small_half_y = SMALL_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = SMALL_ARMOR_HEIGHT / 2.0 / 1000.0;
  constexpr double large_half_y = LARGE_ARMOR_WIDTH / 2.0 / 1000.0;
  constexpr double large_half_z = LARGE_ARMOR_HEIGHT / 2.0 / 1000.0;

  // 初始化小号护甲板的三维点
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, -small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));
  small_armor_points_.emplace_back(cv::Point3f(0, -small_half_y, -small_half_z));
  // 初始化大号护甲板的三维点
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, -large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, large_half_z));
  large_armor_points_.emplace_back(cv::Point3f(0, -large_half_y, -large_half_z));
}

/**
 * 使用PnP算法求解相机位姿
 * 
 * @param armor 表示被检测的护甲板对象。
 * @param rvec 输出参数，表示求解得到的旋转向量。
 * @param tvec 输出参数，表示求解得到的平移向量。
 * @return 解算成功返回true，失败返回false。
 */
bool PnPSolver::solvePnP(const Armor & armor, cv::Mat & rvec, cv::Mat & tvec)
{
  std::vector<cv::Point2f> image_armor_points;

  // 填充图像点信息
  image_armor_points.emplace_back(armor.bl);
  image_armor_points.emplace_back(armor.tl);
  image_armor_points.emplace_back(armor.tr);
  image_armor_points.emplace_back(armor.br);

  // 根据护甲板类型选择对应的三维点集，然后使用PnP算法求解位姿
  auto object_points = armor.type == ArmorType::SMALL ? small_armor_points_ : large_armor_points_;
  cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);
  return cv::solvePnP(
    object_points, image_armor_points, camera_matrix_, dist_coeffs_, rvec, tvec, true,
    cv::SOLVEPNP_ITERATIVE);
}

/**
 * 计算图像点到相机中心的距离
 * 
 * @param image_point 图像中的一个点。
 * @return 该点到相机中心的距离。
 */
float PnPSolver::calculateDistanceToCenter(const cv::Point2f & image_point)
{
  // 获取相机中心坐标，并计算图像点到相机中心的距离
  float cx = camera_matrix_.at<double>(0, 2);
  float cy = camera_matrix_.at<double>(1, 2);
  return cv::norm(image_point - cv::Point2f(cx, cy));
}

}  // namespace rm_auto_aim
