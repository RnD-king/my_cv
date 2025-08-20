# cmake_minimum_required(VERSION 3.8)  # CMake 최소 버전 지정
# project(my_cv)  # 프로젝트 이름은 패키지명과 같게

# # ROS 2 필수 빌드 도구
# find_package(ament_cmake REQUIRED)

# # ROS 2 C++ 라이브러리들
# find_package(rclcpp REQUIRED)
# find_package(sensor_msgs REQUIRED)
# find_package(std_msgs REQUIRED)
# find_package(cv_bridge REQUIRED)
# find_package(my_cv_msgs REQUIRED)  # 커스텀 메시지

# # OpenCV 찾기
# find_package(OpenCV REQUIRED)

# # CUDA / TensorRT (경로는 Jetson Orin Nano에 맞게 알아서 잘 인식됨)
# find_package(CUDA REQUIRED)
# include_directories(/usr/include/x86_64-linux-gnu)  # TensorRT 헤더 기본 위치

# # include 디렉토리 등록 (헤더 파일용)
# include_directories(
#   include  # 너가 만든 include/my_cv/yolo.hpp용
#   ${OpenCV_INCLUDE_DIRS}
# )

# # 노드 실행 파일 추가 (C++ 파일 목록)
# add_executable(line_trt_node
#   my_cv/my_cv_node.cpp
#   my_cv/model.cpp
# )

# # 헤더 include 경로 지정
# target_include_directories(line_trt_node PUBLIC
#   include
# )

# # TensorRT 및 기타 라이브러리 링크
# target_link_libraries(line_trt_node
#   ${OpenCV_LIBS}
#   nvinfer  # TensorRT
#   nvonnxparser  # ONNX 파서 (필요 시)
#   cudart  # CUDA 런타임
# )

# # ROS 2 의존성 자동 연결
# ament_target_dependencies(line_trt_node
#   rclcpp
#   sensor_msgs
#   std_msgs
#   cv_bridge
#   my_cv_msgs
# )

# # 설치 설정 (빌드 후 어디로 설치할지)
# install(TARGETS line_trt_node
#   DESTINATION lib/${PROJECT_NAME}
# )

# ament_package()  # ament 패키지로 마무리
