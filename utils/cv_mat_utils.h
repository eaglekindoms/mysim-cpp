#ifndef CV_MAT_UTILS_H
#define CV_MAT_UTILS_H

#include <opencv2/opencv.hpp>

//! [fftshift]
inline void fftshift(const cv::Mat &inputImg, cv::Mat &outputImg) {
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


inline cv::Mat cvfft2(cv::Mat &input) {
    cv::Mat planes[] = {cv::Mat_<float>(input), cv::Mat::zeros(input.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    cv::dft(complexImg, complexImg);
//    std::cout << "complexImg = " << complexImg << std::endl;
    // compute log(1 + sqrt(Re(DFT(img))**2 + Im(DFT(img))**2))
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];
    mag += cv::Scalar::all(1);
    cv::log(mag, mag);
    // crop the spectrum, if it has an odd number of rows or columns
    mag = mag(cv::Rect(0, 0, mag.cols & -2, mag.rows & -2));
    // rearrange the quadrants of Fourier image
    // so that the origin is at the image center
    fftshift(mag, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    return mag;
}

#endif // CV_MAT_UTILS_H
