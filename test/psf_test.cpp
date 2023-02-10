#include <opencv2/opencv.hpp>
#include <itpp/signal/transforms.h>
#include <utils/itpp_mat_utils.h>
#include <utils/cv_mat_utils.h>

using namespace std;
using namespace itpp;

void test_generate_psf();

void test_edgeTaper();

int main() {
//    test_generate_psf();
    test_edgeTaper();
    cv::waitKey(0);
    return 0;
}

void test_generate_psf() {
    int w = 512;
    mat psf = generatorPSF(w, 0.63);
    cv::Mat cv_psf(w, w, CV_64F, psf._data());
    cv::Mat cv_otf = cvfft2(cv_psf);
    mat otf = PSFToOTF(psf);
    cv::Mat it_otf(w, w, CV_64F, otf._data());
//    cout << "it_otf = " << it_otf << endl;

    cv::imshow("cv_psf", cv_psf);
    cv::imshow("cv_otf", cv_otf);
    cv::imshow("it_otf", it_otf);
}

void test_edgeTaper() {
//    vec x = linspace(1, 10, 10);
//    cvec fx = fft_real(x, 100);
//    cout << fx << endl;
    int w = 512;
    int wo = w / 2;
    mat psf = generatorPSF(w, 0.63);
    mat otf = PSFToOTF(psf);
    int h = 30;
    otf = pow(otf, 10);
    otf = fftshift(otf);
    cmat cotf = to_cmat(otf);
    otf = ifft2(cotf);
    mat psfd = fftshift(otf);
    psfd = psfd / max(max(psfd));
    psfd = psfd / sum(sum(psfd));
    mat PSFe = psfd.get(wo - h, wo + h - 1, wo - h, wo + h - 1);
//    cout<<PSFe<<endl;
//    cv::Mat cv_psfe(w, w, CV_64F, PSFe._data());
    cv::Mat testpat = cv::imread("dataset/testpat.tiff", cv::IMREAD_GRAYSCALE);
    testpat = testpat.rowRange(256, 768);
    testpat = testpat.colRange(256, 768);
    mat obj = cvmat2mat(testpat);
    mat blurred = edgeTaper(obj, PSFe);
    blurred = blurred / max(max(blurred));
    cv::Mat result(w, w, CV_64F, blurred._data());
    cv::imshow("result", result);
    cv::imshow("testpat", testpat);
}