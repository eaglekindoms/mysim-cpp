#include <iostream>
#include <opencv2/opencv.hpp>
#include <itpp/itcomm.h>

using namespace cv;
using namespace std;

int main() {
    cout << "Hello World!" << endl;
    Mat img = imread("D:\\WORKSPACE\\ImageJ\\datasets\\OMX_LSEC_Membrane_680nm.tif");
    imshow("test", img);
    Mat resimg;
    //高斯模糊
    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
    imshow("resimg", resimg);//显示图片
    waitKey(0);
    return 0;
}

