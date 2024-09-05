// Copyright (c) 2021 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lzhw.hpp"
/*-------------------------------------------
                  Main Functions
-------------------------------------------*/
int main(int argc, char** argv)
{
  int ret;
  std::cout<<"正在打开camera"<<std::endl;
  // system("gnome-terminal -- roslaunch rknn_ros camera.launch");
  // std::string userInput;
    
  // std::cout << "是否正常? (Y/N): ";

  // std::cin >> userInput;

  // if(userInput=="Y"||userInput=="y");
  // else exit(1);
  
  //ROS node relative 
  ros::init(argc, argv, "rknn_yolov5_node"); /**/
  ros::NodeHandle nhLocal("~");
  ros::NodeHandle n;
  std::string node_name = ros::this_node::getName();

  nhLocal.param("prob_threshold", box_conf_threshold, 0.35f);
  nhLocal.param("display_output", display_output, true);

  const std::string package_name = "rknn_ros";
  std::string chip_type;

  nhLocal.param("chip_type", chip_type, std::string("RK3588"));
  std::string path = ros::package::getPath(package_name)+("/models/")+chip_type+("/");
  ROS_INFO("Assets path: %s", path.c_str());

  std::string model_file;
  nhLocal.param("model_file", model_file, std::string("lzhw15.rknn"));
  /* Create the neural network */

  ROS_INFO("Loading mode...\n");
  ROS_INFO("Loading  model path: %s", (path+model_file).c_str());

  int            model_data_size = 0;
  unsigned char* model_data      = load_model((path+model_file).c_str(), &model_data_size);
  ret                            = rknn_init(&ctx, model_data, model_data_size, 0, NULL);
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }

  rknn_sdk_version version;
  ret = rknn_query(ctx, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  ROS_INFO("sdk version: %s driver version: %s\n", version.api_version, version.drv_version);


  ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
  if (ret < 0) {
    printf("rknn_init error ret=%d\n", ret);
    return -1;
  }
  ROS_INFO("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

  rknn_tensor_attr input_attrs[io_num.n_input];
  memset(input_attrs, 0, sizeof(input_attrs));
  for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret                  = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
    if (ret < 0) {
      printf("rknn_init error ret=%d\n", ret);
      return -1;
    }
    dump_tensor_attr(&(input_attrs[i]));
  }

  memset(output_attrs, 0, sizeof(output_attrs));
  for (int i = 0; i < io_num.n_output; i++) {
    output_attrs[i].index = i;
    ret                   = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    dump_tensor_attr(&(output_attrs[i]));
  }

  if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
    ROS_INFO("model is NCHW input fmt\n");
    channel = input_attrs[0].dims[1];
    height  = input_attrs[0].dims[2];
    width   = input_attrs[0].dims[3];
  } else {
    ROS_INFO("model is NHWC input fmt\n");
    height  = input_attrs[0].dims[1];
    width   = input_attrs[0].dims[2];
    channel = input_attrs[0].dims[3];
  }

  ROS_INFO("model input height=%d, width=%d, channel=%d\n", height, width, channel);

  //ROS node relative 
  image_transport::ImageTransport it(n);
  image_pub = it.advertise("/rknn_image", 1);
  obj_pub = n.advertise<object_information_msgs::Object>("/objects", 50);  
  image_transport::Subscriber video = it.subscribe("/usb_cam/image_raw", 1, imageCallback);

  // ros::Subscriber sub = n.subscribe("/rknn_image", 1, imageCallback_lzhw);
  // cv::namedWindow("RKNN Image");

  while (ros::ok()) 
  {
    ros::spinOnce();
  }

  // release
  ret = rknn_destroy(ctx);
  cv::destroyWindow("RKNN Image");
  if (model_data) 
  {
    free(model_data);
  }
  return 0;
}
