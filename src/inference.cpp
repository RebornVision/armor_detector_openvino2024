#include "armor_detector/detector.hpp"

namespace rm_auto_aim
{
  void ArmorDetector::init(const size_t &input_w, const size_t &input_h, const std::string MODEL_PATH){
    INPUT_W = input_w;
    INPUT_H = input_h;
    input_shape = {1, INPUT_H, INPUT_W, 3};
    scale = 1.0 / (std::min(MODUL_INPUT_H * 1.0 / INPUT_H, MODUL_INPUT_W * 1.0 / INPUT_W));
    cv::Size new_unpad = cv::Size(int(round(INPUT_W / scale)), int(round(INPUT_H / scale)));
    int dw = (MODUL_INPUT_W - new_unpad.width);
    int dh = (MODUL_INPUT_H - new_unpad.height);
    // -------- 加载模型 --------
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(MODEL_PATH);
    // -------- 配置模型 --------
    ov::preprocess::PrePostProcessor ppp(model);
    //输入
    ppp.input().tensor()
    .set_element_type(input_type)
    .set_layout(input_layout)
    .set_color_format(input_ColorFormat)
    .set_shape(input_shape)
    ;
    //预处理依次按顺序执行，缩放必须放最前面，否则会增加耗时，pad必须在转换类型之前，否则会出现错误
    ppp.input().preprocess()
    .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR, new_unpad.height, new_unpad.width)
    .convert_element_type(Moudel_type)
    .convert_layout(Moudel_layout)
    .pad({0, 0, 0, 0},          // batch, channel, height, width
        {0, 0, dh, dw}, 114.0f, ov::preprocess::PaddingMode::CONSTANT)
    .convert_color(Moudel_ColorFormat)
    .scale(255.0f)
    ;
    // //输出
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();

    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
  }
  
  void ArmorDetector::startInferAndNMS(cv::Mat& img, int detect_color)
  {
    // Set input tensor for model with one input
    ov::Tensor input_tensor(input_type, input_shape, img.ptr(0));

    infer_request.set_input_tensor(input_tensor);
    // -------- Start inference --------
    infer_request.infer();
    // -------- Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();

    // -------- Postprocess the result --------
    float *data = output.data<float>();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,14]
    
    for (int cls=4 ; cls < (4+class_num); ++cls) {
      if (detect_color == cls - 4) {
        continue;
      }
      OneClassArmor SingleData;
      for (int i = 0; i < output_buffer.rows; i++) {
        float class_score = output_buffer.at<float>(i, cls);
        //保证当前对应的板子信息匹配
        float max_class_score = 0.0;
        for (int j = 4; j < (4+class_num); j++) {
          if (max_class_score < output_buffer.at<float>(i, j)) {
            max_class_score = output_buffer.at<float>(i, j);
          }
        }
        if (class_score != max_class_score) {
          continue;
        }

        if (class_score > BBOX_CONF_THRESH) {
          SingleData.class_scores.push_back(class_score);

          float cx = output_buffer.at<float>(i, 0);
          float cy = output_buffer.at<float>(i, 1);
          float w = output_buffer.at<float>(i, 2);
          float h = output_buffer.at<float>(i, 3);

          // Get the box
          int left = int((cx - 0.5 * w-2) * scale);
          int top = int((cy - 0.5 * h - 2) * scale);
          int width = int(w * 1.2 * scale);
          int height = int(h * 1.2 * scale);

          // Get the keypoints
          std::vector<float> keypoints;
          cv::Mat kpts = output_buffer.row(i).colRange( (4+class_num), output_buffer.cols);
          for (int i = 0; i < 4; i++) {
              float x = kpts.at<float>(0, i * 2 + 0) * scale;
              float y = kpts.at<float>(0, i * 2 + 1) * scale;

              keypoints.push_back(x);
              keypoints.push_back(y);
          }

          SingleData.boxes.push_back(cv::Rect(left, top, width, height));
          SingleData.objects_keypoints.push_back(keypoints);
        }
      }

      SingleData.class_ids = cls - 4;
      //NMS处理
      std::vector<int> indices;
      cv::dnn::NMSBoxes(SingleData.boxes, SingleData.class_scores, BBOX_CONF_THRESH, NMS_THRESH, indices);
      for (auto i : indices) {
        OneArmor armor;
        armor.box = SingleData.boxes[i];
        armor.class_scores = SingleData.class_scores[i];
        armor.class_ids = SingleData.class_ids;

        for (int j = 0; j < 4; j++) {
          int x = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 0]), 0, static_cast<int>(INPUT_W));
          int y = std::clamp(int(SingleData.objects_keypoints[i][j * 2 + 1]), 0, static_cast<int>(INPUT_H));
          armor.objects_keypoints[j] = cv::Point(x, y);
        }
        last_Armors.push_back(armor);
      }
    }
  }
  void ArmorDetector::clear_armor()
  {
      last_Armors.clear();
  }
  std::vector<OneArmor>& ArmorDetector::get_armor()
  {
      return last_Armors;
  }
}  // namespace rm_auto_aim