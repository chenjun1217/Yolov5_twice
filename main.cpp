#include <iostream>
#include "Yolov5.h"
using namespace std;

int main()
{
    bool is_cuda = 0;
    cv::dnn::Net net;
    Yolov5* Yolo_v5_Once;
    Yolov5* Yolo_v5_Twice;
    std::string Classes_Filepath_Once = "/home/cj/chaintwork/pcl/code_c++/test_Yolov5/config_files/classes.txt";
    std::string Net_Filepath_Once = "/home/cj/chaintwork/pcl/code_c++/test_Yolov5/config_files/yolov5s.onnx";
    std::vector<std::string> class_list_Once = Yolo_v5_Once->load_class_list(Classes_Filepath_Once);
    cv::Mat Frame_Once= cv::imread("/home/cj/chaintwork/pcl/code_c++/Yolov5_Detect_Twice/Test.jpeg");
    Yolo_v5_Once->load_net(net, is_cuda,Net_Filepath_Once);
    std::vector<Detection> output;
    Yolo_v5_Once->detect(Frame_Once, net, output, class_list_Once);
    for (const auto& det : output) {
        cout << "Class ID: " << det.class_id << ", ";
        cout << "Confidence: " << det.confidence << ", ";
        cout << "Box: [" << det.box.x << ", " << det.box.y << ", " << det.box.width << ", " << det.box.height << "]" << endl;
    }

    //Yolo_v5_Once->Draw_Img(Frame_Once, output, class_list_Once);
    //cv::imwrite("Once_output.jpg",Frame_Once);

    //第二次检测
    // INSERT_YOUR_CODE
    std::vector<Detection> output_Twice;
    for (const auto& det : output) {
        if (det.class_id == 0) {
            output_Twice.push_back(det);
        }
    }

    if (!output_Twice.empty()) {
        auto max_area_det = std::max_element(output_Twice.begin(), output_Twice.end(), [](const Detection& a, const Detection& b) {
            return a.box.area() < b.box.area();
        });

        cv::Rect roi(max_area_det->box.x, max_area_det->box.y, max_area_det->box.width, max_area_det->box.height);
        cv::Mat Frame_Twice = Frame_Once(roi);

        std::string Classes_Filepath_Twice = "/home/cj/chaintwork/pcl/code_c++/test_Yolov5/config_files/classes.txt";
        std::string Net_Filepath_Twice = "/home/cj/chaintwork/pcl/code_c++/test_Yolov5/config_files/yolov5s.onnx";
        std::vector<std::string> class_list_Twice = Yolo_v5_Twice->load_class_list(Classes_Filepath_Twice);
        std::vector<Detection> output_Twice;
        Yolo_v5_Twice->load_net(net, is_cuda, Net_Filepath_Twice);
        Yolo_v5_Twice->detect(Frame_Twice, net, output_Twice, class_list_Twice);
        Yolo_v5_Twice->Draw_Img(Frame_Twice, output_Twice, class_list_Twice);
        cv::imwrite("Twice_ouput.jpg",Frame_Twice);

        for (const auto& det : output_Twice) {
            cout << "Class ID: " << det.class_id << ", ";
            cout << "Confidence: " << det.confidence << ", ";
            cout << "Box: [" << det.box.x << ", " << det.box.y << ", " << det.box.width << ", " << det.box.height << "]" << endl;
        }
    }


    cout << "Hello World!" << endl;
    return 0;
}
