#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include <utils/itpp_mat_utils.h>
#include <utils/sim_utils.h>
#include <utils/thread_pool.h>
#include <utils/opensim/OpenSIMReconstruct.cpp>
#include <chrono>

using namespace std;
using namespace itpp;

using namespace std;

int main() {
    cout << "Hello OpenSIM!" << endl;
    int w = 512;
    OtfFactory otfFactory = OtfFactory(w, 0.63);
    cv::Mat cv_psf(w, w, CV_64F, otfFactory.psf._data());
    cv::Mat it_otf(w, w, CV_64F, otfFactory.otf._data());
    cv::Mat testpat = cv::imread("dataset/testpat.tiff", cv::IMREAD_GRAYSCALE);
    //get center range(257:768,257:768);
    testpat = testpat.rowRange(256, 768);
    testpat = testpat.colRange(256, 768);
    double k2 = 75.23; // illumination freq
    double modFac = 0.8;// modulation factor
    double noiseLevel = 10.; // in percentage
//    testpat.convertTo(testpat,CV_64F);
    mat obj = cvmat2mat(testpat);
    Vec<mat> patterns = simulateSIMImage(k2, obj, otfFactory, modFac, noiseLevel, 1);
    showPatternImage("raw sim images", patterns, obj.rows(), 1);
    // obtaining the noisy estimates of three frequency components
    // 计时器
    long long t1 = get_cur_time();
    OpenSIMReconstruct opensim;
    SIMParam simParam = opensim.estimateParameters(patterns, otfFactory);
    Vec<cmat> freqComp = opensim.separatedSIMComponents(patterns, simParam, otfFactory);
    long long t2 = get_cur_time();
    std::cout << "Estimate parameter use: " << t2 - t1 << "ms.\n";
    mat reconstructImage = opensim.reconstruct(freqComp, simParam, otfFactory);
    // show raw image
    obj = patterns[9];
    int objMax = max(max(obj, 1));
    obj = obj / objMax;
    cv::Mat groundTruth(w, w, CV_64F, obj._data());
    cv::Mat result(w, w, CV_64F, reconstructImage._data());
    cv::Mat otfShow(w, w, CV_64F, otfFactory.otf._data());
    cv::imshow("testpat", testpat);
    cv::imshow("ground truth", groundTruth);
    cv::imshow("reconstruct result", result);
    cv::imshow("otf", otfShow);
    std::cout << simParam << std::endl;
//    cv::imwrite("objs.tiff", objs);
//    Mat resimg;
//    //高斯模糊
//    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
//    imshow("resimg", resimg);//显示图片
    cv::waitKey(0);
    return 0;
}

