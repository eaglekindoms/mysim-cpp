#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include "itpp_mat_utils.h"
#include "cv_mat_utils.h"

using namespace std;
using namespace itpp;


void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg);

cv::Mat cmat2cvmat(cmat &psf)
{
    int row = psf.rows();
    int col = psf.cols();
    mat temp = log10(real(psf));
    cv::Mat real(row, col, CV_64F, temp._data());
    temp = log10(imag(psf));
    cv::Mat imag(row, col, CV_64F, temp._data());
    cv::Mat planes[] = {real, imag};
    cv::Mat otf;
    cv::merge(planes, 2, otf);
    return otf;
}

int main()
{
    int w = 512;
    mat psf = generatorPSF(w, 0.3);
    cmat otf = fft2(psf);
    cv::Mat cvMat(w, w, CV_64F, psf._data());
    cv::Mat cvMat1 = cvfft2(cvMat);
//    cv::log(cvMat1,cvMat1);
//    cv::normalize(cvMat1, cvMat1, 0, 1, cv::NORM_MINMAX);

//    cout << "otf = " << otf << endl;
//    cout << "cvMat1 = " << cvMat1 << endl;
    cv::imshow("cvMat", cvMat);
    cv::imshow("cvMat1", cvMat1);
    cv::imwrite("psf.tiff", cvMat);
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
