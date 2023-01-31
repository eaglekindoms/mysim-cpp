﻿#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include "itpp_mat_utils.h"
#include "cv_mat_utils.h"

using namespace std;
using namespace itpp;

int main() {
    int w = 512;
    mat psf = generatorPSF(w, 0.3);
    cv::Mat cv_psf(w, w, CV_64F, psf._data());
    cv::Mat cv_otf = cvfft2(cv_psf);
    mat otf = PSFToOTF(psf);
    cv::Mat it_otf(w, w, CV_64F, otf._data());
//    cout << "it_otf = " << it_otf << endl;

    cv::imshow("cv_psf", cv_psf);
    cv::imshow("cv_otf", cv_otf);
    cv::imshow("it_otf", it_otf);
//    cv::imwrite("psf.tiff", cv_psf);
    cv::waitKey(0);
    return 0;
}