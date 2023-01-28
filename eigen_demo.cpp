#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
//using namespace cv;
using namespace std;

int main()
{
  MatrixXd m = MatrixXd::Random(300,300);
  m = (m + MatrixXd::Constant(300,300,1.2)) * 0.5;
//  std::cout << "m =" << std::endl << m << std::endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cv::Mat cvMat;
  cv::eigen2cv(m,cvMat);
  cv::imshow("cvMat",cvMat);
  cv::waitKey(0);
  return 0;
//  std::cout << "m * v =" << std::endl << m * v << std::endl;
}
