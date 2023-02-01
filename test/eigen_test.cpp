#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

using Eigen::MatrixXd;
using Eigen::MatrixXcd;
using Eigen::VectorXd;

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
//using namespace cv;
using namespace std;

int main() {
    int w = 300;
    VectorXd vv = VectorXd::LinSpaced(w, 0.1, 1);
    MatrixXd m(w, w);
    MatrixXcd cm(w, w);
    for (int i = 0; i < w; ++i) {
        m.row(i) = vv;
    }
//    m = (m + MatrixXd::Constant(300, 300, 1.2)) * 0.5;
    std::cout << "m =" << std::endl << m << std::endl;
    Eigen::FFT<complex<double>> fft;
//    auto a =fft.fwd(m);
//    VectorXd v(3);
//    v << 1, 2, 3;
    cv::Mat cvMat;
    cv::eigen2cv(m, cvMat);
//    cv::imshow("cvMat", cvMat);
//    cv::waitKey(0);
    return 0;
//  std::cout << "m * v =" << std::endl << m * v << std::endl;
}
