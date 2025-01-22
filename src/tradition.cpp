//
// Created by gx on 24-4-28.
//
#include "armor_detector/tradition.hpp"
#include "armor_detector/light_corner_corrector.hpp"

namespace rm_auto_aim{

    static void sort_keypoints(cv::Point2f keypoints[4]) {
    // Sort points based on their y-coordinates (ascending)
    std::sort(keypoints, keypoints + 4, [](const cv::Point& a, const cv::Point& b) {
        return a.y < b.y;
    });

    // Top points will be the first two, bottom points will be the last two
    cv::Point top_points[2] = { keypoints[0], keypoints[1] };
    cv::Point bottom_points[2] = { keypoints[2], keypoints[3] };

    // Sort the top points by their x-coordinates to distinguish left and right
    std::sort(top_points, top_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Sort the bottom points by their x-coordinates to distinguish left and right
    std::sort(bottom_points, bottom_points + 2, [](const cv::Point& a, const cv::Point& b) {
        return a.x < b.x;
    });

    // Assign sorted points back to the keypoints array
    keypoints[0] = top_points[0];     // top-left
    keypoints[1] = bottom_points[0];  // bottom-left
    keypoints[2] = bottom_points[1];  // bottom-right
    keypoints[3] = top_points[1];     // top-right
}
    cv::Mat preprocessROI(const cv::Mat& img,cv::Rect armor_box,int binary_threshold_light){ 
        //std::cout<<"binary_threshold_light"<<binary_threshold_light<<std::endl;      //返回ROI二值化后的图像
        cv::Mat roi = img(armor_box);
        cv::Mat gray_img;
        cv::cvtColor(roi, gray_img, cv::COLOR_RGB2GRAY);

        cv::Mat binary_img;
        cv::threshold(gray_img, binary_img, binary_threshold_light, 255, cv::THRESH_BINARY);
        // cv::imshow("binary_img",binary_img);
        // cv::waitKey(1);

        return binary_img;

    }
    bool isLight(const Light &light){

        // The ratio of light (short side / long side)
        float ratio = light.width / light.length;
        bool ratio_ok = min_ratio < ratio && ratio < max_ratio;

        bool angle_ok = light.tilt_angle < max_angle;

        bool is_light = ratio_ok && angle_ok;

        return is_light;
    }

    std::vector<Light> findLight(const cv::Mat& rbg_img, const cv::Mat &binary_img ){
        using std::vector;
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<Light> lights;
        for(const auto &contour : contours){
            auto r_rect = cv::minAreaRect(contour);
            auto light = Light(r_rect);

        if(isLight(light)){
            auto rect = light.boundingRect();
            if (  // Avoid assertion failed
                    0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
                    0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
                int sum_r = 0, sum_b = 0;
                auto roi = rbg_img(rect);
                // Iterate through the ROI
                for (int i = 0; i < roi.rows; i++) {
                    for (int j = 0; j < roi.cols; j++) {
                        if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
                            // if point is inside contour
                            sum_r += roi.at<cv::Vec3b>(i, j)[0];
                            sum_b += roi.at<cv::Vec3b>(i, j)[2];
                        }
                    }
                }
                // Sum of red pixels > sum of blue pixels ?
                light.color = sum_r > sum_b ? BLUE : RED;//0 ? 1
                lights.emplace_back(light);
            }
        }
        }
        std::vector<Light>lights_ =  sortLight(lights);
        return lights_;//这里已经成功找到一个装甲板的两个或一个灯条
    }
    std::vector<Light> sortLight(std::vector<Light> &lights){
        if(lights.size()!=2)return lights;
        std::vector<Light> lights_sorted;
        if(lights[0].top.x<lights[1].top.x&&lights[0].bottom.x<lights[1].bottom.x){
            lights_sorted.push_back(lights[0]);
            lights_sorted.push_back(lights[1]);
        }else{
            lights_sorted.push_back(lights[1]);
            lights_sorted.push_back(lights[0]);
        }
        return lights_sorted;
    }
    std::vector<OneArmor> tradition(cv::Mat input_img,std::vector<OneArmor>& armors_data,int binary_threshold_light){
        std::vector<OneArmor> armors;
        for(auto armor:armors_data){
            int min_ = std::min(input_img.rows,input_img.cols);
            if(armor.box.x<0||armor.box.y<0||(armor.box.height+armor.box.y)>min_||(armor.box.width+armor.box.x)>min_) {
                armors.push_back(armor);
                continue;}
            cv::Mat roi_binary_img = preprocessROI(input_img,armor.box,binary_threshold_light);
            cv::Mat rbg_img = input_img(armor.box);
            cv::Mat gray_img ;
            cv::cvtColor(rbg_img,gray_img,cv::COLOR_RGB2GRAY);
            std::vector<Light>lights = findLight(rbg_img,roi_binary_img);
            if(lights.size()!=2){
                armors.push_back(armor);
                continue;}
            if(lights[0].length*1.0/lights[1].length<0.8||lights[0].length*1.0/lights[1].length>1.3){
                armors.push_back(armor);
                continue;

            }
            if(lights.size()==2){
            OneArmor armor_corner_corrected(lights[0],lights[1]);
            correctCorners(armor_corner_corrected,gray_img);
            armor.objects_keypoints[0] = cv::Point2f(armor_corner_corrected.left_light.top.x+armor.box.x,armor_corner_corrected.left_light.top.y+armor.box.y);
            armor.objects_keypoints[1] = cv::Point2f (armor_corner_corrected.left_light.bottom.x+armor.box.x,armor_corner_corrected.left_light.bottom.y+armor.box.y);
            armor.objects_keypoints[2] = cv::Point2f (armor_corner_corrected.right_light.bottom.x+armor.box.x,armor_corner_corrected.right_light.bottom.y+armor.box.y);
            armor.objects_keypoints[3] = cv::Point2f (armor_corner_corrected.right_light.top.x+armor.box.x,armor_corner_corrected.right_light.top.y+armor.box.y);
            // // std::cout<<"find two light"<<std::endl;
            //     armor.objects_keypoints[0] = cv::Point2f(lights[0].top.x+armor.box.x,lights[0].top.y+armor.box.y);
            // // std::cout<<"armor.objects_keypoints[0]"<<armor.objects_keypoints[0]<<std::endl;
            //     armor.objects_keypoints[1] = cv::Point2f (lights[0].bottom.x+armor.box.x,lights[0].bottom.y+armor.box.y);
            //     armor.objects_keypoints[2] = cv::Point2f (lights[1].bottom.x+armor.box.x,lights[1].bottom.y+armor.box.y);
            //     armor.objects_keypoints[3] = cv::Point2f (lights[1].top.x+armor.box.x,lights[1].top.y+armor.box.y);
             //   armor.class_ids = armor.class_ids;
                sort_keypoints(armor.objects_keypoints);
                armors.push_back(armor);
            }
        }
        return armors;
    }

}  // namespace rm_auto_aim















