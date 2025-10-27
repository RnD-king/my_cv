cmake_minimum_required(VERSION 3.18)
project(my_cv)

# -------------------------------------------------------------
# CUDA & C++
# -------------------------------------------------------------
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Jetson Orin Nano (Compute Capability 8.7)
set(CMAKE_CUDA_ARCHITECTURES 87)

# -------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui cudaimgproc cudaarithm cudawarping)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(robot_msgs REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# -------------------------------------------------------------
# 1) CUDA Preprocess Library (GPU 전처리)
# -------------------------------------------------------------
add_library(cuda_preprocess STATIC
  include/yolo_preprocess.cu
)
set_target_properties(cuda_preprocess PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(cuda_preprocess
  CUDA::cudart
)

# -------------------------------------------------------------
# 2) YOLO C++ TensorRT Node
# -------------------------------------------------------------
add_executable(yolo_cpp
  my_cv/yolo_cpp.cpp
  my_cv/yolo_trt.cpp
)

target_include_directories(yolo_cpp PUBLIC
  include
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# TensorRT Core + Plugin (EfficientNMS는 plugin 안에 내장)
target_link_libraries(yolo_cpp
  cuda_preprocess
  ${OpenCV_LIBS}
  nvinfer
  nvinfer_plugin
  cudart
)

ament_target_dependencies(yolo_cpp
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  robot_msgs
)

# -------------------------------------------------------------
# 3) Installation
# -------------------------------------------------------------
install(TARGETS yolo_cpp
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  my_cv/ball_detect.py
  my_cv/ball_and_hoop.py
  my_cv/ball_detect3.py
  my_cv/image_saver.py
  my_cv/hurdle_detect.py
  my_cv/hoop_detect.py
  my_cv/hoop_detect2.py
  my_cv/yolo_line_subscriber.py
  my_cv/color_mask_test.py
  my_cv/ball_and_hoop2.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

# -------------------------------------------------------------
# 4) Lint / Test
# -------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
