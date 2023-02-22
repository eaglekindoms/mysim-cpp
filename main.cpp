#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include <utils/itpp_mat_utils.h>
#include <utils/sim_utils.h>
#include <utils/thread_pool.h>
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
    Vec<tuple<Vec<cmat>, vec>> components(3);
    ThreadPool pool(3);
    vector<future<tuple<Vec<cmat>, vec>>> poolResult;
    for (int i = 0; i < 9; i = i + 3) {
        poolResult.emplace_back(pool.enqueue([=] {
            cout << "estimates frequency components, index: " << i / 3 << endl;
            return separatedSIMComponents2D(patterns, otfFactory, i);
        }));
//        components[i / 3] = separatedSIMComponents2D(patterns, otf, i);
    }
    for (int i = 0; i < 3; ++i) {
        components[i] = poolResult[i].get();
    }
    // averaging the central frequency components
    cmat fCent = (get<0>(components[0])[0] + get<0>(components[1])[0] + get<0>(components[2])[0]) / 3;
    // Object power parameters determination
    vec OBJParaA = estimateObjectPowerParameters(fCent, otfFactory);
    // Wiener Filtering the noisy frequency components
    Vec<mat> filterComps(9);
    Vec<cmat> freqComp(9);
    vec noiseComp(9);
    vec modFactors(3);
    Vec<vec> freqVectors(3);
    for (int i = 0; i < 3; ++i) {
        tuple<Vec<tuple<cmat, double>>, double> fComp = wienerFilter(components[i], OBJParaA, otfFactory);
        for (int j = 0; j < 3; ++j) {
            filterComps[i * 3 + j] = real(get<0>(get<0>(fComp)[j]));
            freqComp[i * 3 + j] = get<0>(get<0>(fComp)[j]);
            noiseComp[i * 3 + j] = get<1>(get<0>(fComp)[j]);
        }
        modFactors[i] = get<1>(fComp);
        freqVectors[i] = get<1>(components[i]);
    }
//    showPatternImage("filtered sim images", filterComps, obj.rows(), 0);
    Vec<cmat> results = mergeSIMImages(freqComp, noiseComp, modFactors, freqVectors, OBJParaA, otfFactory.otf);
    Vec<mat> reconstructImages(6);
    for (int i = 0; i < 3; ++i) {
        reconstructImages[i] = real(ifft2(fftshift(results[i])));
        double rMax = max(max(reconstructImages[i], 1));
        reconstructImages[i] = reconstructImages[i] / rMax;
        reconstructImages[i + 3] = real(results[i]);
    }
    showPatternImage("reconstruction sim images", reconstructImages, obj.rows(), 0);
    // show raw image
    int objMax = max(max(obj, 1));
    obj = obj / objMax;
    cv::Mat rawObj(w, w, CV_64F, obj._data());
    cv::Mat result(w, w, CV_64F, reconstructImages[0]._data());
//    cv::imshow("testpat", testpat);
    cv::imshow("rawObj", rawObj);
    cv::imshow("result", result);
//    cv::imwrite("objs.tiff", objs);
//    Mat resimg;
//    //高斯模糊
//    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
//    imshow("resimg", resimg);//显示图片
    cv::waitKey(0);
    return 0;
}

