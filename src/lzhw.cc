#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ros/package.h>
#include <ros/ros.h>

int main(int argc,char**argv)
{

    std::cout<<"正在打开camera"<<std::endl;
    system("gnome-terminal -- roslaunch rknn_ros camera.launch");
    std::string userInput;
    
    std::cout << "是否正常? (Y/N): ";

    std::cin >> userInput;

    if(userInput=="Y"||userInput=="y")
    {
        std::cout<<"正在打开yolo_ros"<<std::endl;
        system("gnome-terminal -- roslaunch rknn_ros yolov5.launch");
    }
    else
    {
        exit(1);
    }
    std::cout<<"检测图像话题：/rknn_image";
    return 0;
}