cmake_minimum_required(VERSION 3.10)
project(armor_detector)

## Use C++14
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## By adding -Wall and -Werror, the compiler does not ignore warnings anymore,
## enforcing cleaner code.
#add_definitions(-Wall -Werror)

## Export compile commands for clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# find_package(Sophus REQUIRED)
find_package(OpenVINO REQUIRED)
if(NOT OpenVINO_FOUND)
    message(FATAL_ERROR "error")
endif()
set(ngraph_DIR "/opt/intel/openvino_2024/runtime/cmake")
set(InferenceEngine_DIR "/opt/intel/openvino_2024/runtime/cmake")
set(OpenVINO_DIR "/opt/intel/openvino_2024/runtime/cmake")
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake")

set(INC_DIR "/opt/intel/openvino_2024/runtime/include")
set(LINK_DIR "/opt/intel/openvino_2024/runtime/lib/intel64")


include_directories(
        #OpenVINO推理引擎的头文件

        "/opt/intel/openvino_2024/runtime/include/ie/"
        "/opt/intel/openvino_2024/runtime/include/ngraph/"
        "/opt/intel/openvino_2024/runtime/include/openvino/"


)

include_directories(
        ${INC_DIR}
)
link_directories(
        ${LINK_DIR}
)
link_libraries(libopenvino.so)
#######################
## Find dependencies ##
#######################

find_package(ament_cmake_auto REQUIRED)
find_package(OpenCV REQUIRED)
ament_auto_find_build_dependencies()

###########
## Build ##
###########

ament_auto_add_library(${PROJECT_NAME} SHARED
  DIRECTORY src
)

target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS}
        # libopenvino.so
        )

rclcpp_components_register_node(${PROJECT_NAME}
  PLUGIN rm_auto_aim::ArmorDetectorNode
  EXECUTABLE armor_detector_node
)
# link_libraries(libopenvino.so)
#############
## Testing ##
#############

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  list(APPEND AMENT_LINT_AUTO_EXCLUDE
    ament_cmake_copyright
    ament_cmake_uncrustify
    ament_cmake_cpplint
  )
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest)
  ament_add_gtest(test_node_startup test/test_node_startup.cpp)
  target_link_libraries(test_node_startup ${PROJECT_NAME})

  ament_add_gtest(test_number_cls test/test_number_cls.cpp)
  target_link_libraries(test_number_cls ${PROJECT_NAME}
          # libopenvino.so
  )

endif()

#############
## Install ##
#############

ament_auto_package(
  INSTALL_TO_SHARE
  model

)
