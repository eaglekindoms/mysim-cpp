#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include "itpp_mat_utils.h"
#include "cv_mat_utils.h"

using namespace std;
using namespace itpp;


void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg);

cv::Mat cmat2cvmat(cmat &otf)
{
    int row = otf.rows();
    int col = otf.cols();
    mat temp = log10(real(otf));
//    cv::Mat cvotf(row, col, CV_64F,temp._data());
//    cv::normalize(cvotf, cvotf, 0, 1, cv::NORM_MINMAX);
    cv::Mat real(row, col, CV_64F, temp._data());
    temp = log10(imag(otf));
    cv::Mat imag(row, col, CV_64F, temp._data());
    cv::Mat planes[] = {real, imag};
    cv::Mat cvotf;
    cv::merge(planes, 2, cvotf);
    return cvotf;
}

int main()
{
    int w = 512;
    mat psf = generatorPSF(w, 0.3);
    cmat otf = fft2(psf);
    cv::Mat cv_psf(w, w, CV_64F, psf._data());
    cv::Mat cv_otf = cvfft2(cv_psf);
//    cv::Mat it_otf = cmat2cvmat(otf);

//    cout << "otf = " << otf << endl;
//    cout << "it_otf = " << it_otf << endl;
    cv::imshow("cv_psf", cv_psf);
    cv::imshow("cv_otf", cv_otf);
//    cv::imshow("it_otf", it_otf);

//    cv::imwrite("psf.tiff", cv_psf);
    cv::waitKey(0);
    return 0;
}
