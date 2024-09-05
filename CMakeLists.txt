cmake_minimum_required(VERSION 3.0.2)
project(rknn_ros)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

# set(OpenCV_DIR "/home/ucar/move/src/rknn_ros/include/opencv2")
# find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    roslib
    cv_bridge
    image_transport
    image_geometry
    sensor_msgs
    message_generation
    object_information_msgs  
)


###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES rknn_ros
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  "include"
  ${catkin_INCLUDE_DIRS}
)
link_directories(
  ${catkin_LIB_DIRS} 
  lib
)

# YOLO v5
add_executable(rknn_yolov5_node 
                src/rknn_yolov5.cc 
                src/yolov5_postprocess.cc
                )

add_executable(rknn_vision 
                src/lzhw.cc)
add_dependencies(rknn_yolov5_node object_information_msgs_generate_messages_cpp)
add_dependencies(rknn_vision object_information_msgs_generate_messages_cpp)

target_link_libraries(rknn_yolov5_node 
                        ${catkin_LIBRARIES} 
                        librknnrt.so 
                        )

target_link_libraries(rknn_vision
                        ${catkin_LIBRARIES} 
                        librknnrt.so 
                        )