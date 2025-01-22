# armor_detector

- [armor\_detector](#armor_detector)
  - [识别节点](#识别节点)
    - [DetectorNode](#detectornode)
  - [Detector](#detector)
    - [Classify](#classify)
  - [PnPSolver](#pnpsolver)

## 识别节点

订阅相机参数及图像流进行装甲板的识别并解算三维位置，输出识别到的装甲板在输入frame下的三维位置 (一般是以相机光心为原点的相机坐标系)

### DetectorNode
装甲板识别节点

包含[Detector](#detector)
包含[PnPSolver](#pnpsolver)

订阅：
- 相机参数 `/camera_info`
- 彩色图像 `/image_raw`

发布：
- 识别目标 `/detector/armors`

动态参数：
- 是否发布 debug 信息 `debug`
- 识别目标颜色 `detect_color`
- 二值化的最小阈值 `binary_thres`
- 数字分类器 `classifier`
  - 置信度阈值 `threshold`

## Detector
装甲板识别器
推理部分适配了openvino2024版，并将预处理操作改为openvino的API的配置模型的预处理。
神经网络直接给出四个灯条角点和红蓝分类<br>
考虑到角点精度时限，对神经网络返回的框做了适当的延伸，并在新的框内用传统方法寻找四个角点

### Classify
分类

采用Lenet5进行分类

## PnPSolver
PnP解算器

[Perspective-n-Point (PnP) pose computation](https://docs.opencv.org/4.x/d5/d1f/calib3d_solvePnP.html)

PnP解算器将 `cv::solvePnP()` 封装，接口中传入 `Armor` 类型的数据即可得到 `geometry_msgs::msg::Point` 类型的三维坐标。

考虑到装甲板的四个点在一个平面上，在PnP解算方法上我们选择了 `cv::SOLVEPNP_IPPE` (Method is based on the paper of T. Collins and A. Bartoli. ["Infinitesimal Plane-Based Pose Estimation"](https://link.springer.com/article/10.1007/s11263-014-0725-5). This method requires coplanar object points.)<br>
实际为了进一步提高解算的稳定性，采用了pnp迭代法求解
