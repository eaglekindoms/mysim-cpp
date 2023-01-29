#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include "itpp_mat_utils.h"

using namespace std;
using namespace itpp;


void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg);

int main()
{
  int w=512;
  mat psf=generatorPSF(w,0.3);
//  mat psf1=fftshift(psf);
  cv::Mat cvMat(w,w,CV_64F,psf._data());
//  cv::Mat cvMat1(w,w,CV_64F,psf1._data());
//  fftshift(cvMat1,cvMat);
//  cout<< "cvpsf = "<< cvMat<<endl;
  cv::imshow("cvMat",cvMat);
//  cv::imshow("cvMat1",cvMat1);
//  cv::imwrite("psf.tiff",cvMat);
  cv::waitKey(0);
  return 0;

}

//! [fftshift]
void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg)
{
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    cv::Mat q0(outputImg, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(outputImg, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(outputImg, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(outputImg, cv::Rect(cx, cy, cx, cy));
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
