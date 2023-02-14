#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include <utils/itpp_mat_utils.h>
#include <utils/sim_utils.h>

using namespace std;
using namespace itpp;

using namespace std;

int main() {
    cout << "Hello OpenSIM!" << endl;
    int w = 512;
    mat psf = generatorPSF(w, 0.63);
    cv::Mat cv_psf(w, w, CV_64F, psf._data());
    mat otf = PSFToOTF(psf);
    cv::Mat it_otf(w, w, CV_64F, otf._data());
    cv::Mat testpat = cv::imread("dataset/testpat.tiff", cv::IMREAD_GRAYSCALE);
    //get center range(257:768,257:768);
    testpat = testpat.rowRange(256, 768);
    testpat = testpat.colRange(256, 768);
    double k2 = 75.23; // illumination freq
    double modFac = 0.8;// modulation factor
    double noiseLevel = 10.; // in percentage
//    testpat.convertTo(testpat,CV_64F);
    mat obj = cvmat2mat(testpat);
    Vec<mat> patterns = simulateSIMImage(k2, obj, otf, modFac, noiseLevel, 1);
    // showPatternImage(patterns, obj.rows(), 0);
    // obtaining the noisy estimates of three frequency components
    Vec<tuple<Vec<cmat>, vec>> components(3);
    for (int i = 0; i < 9; i = i + 3) {
        cout << "estimates frequency components, index: " << i / 3 << endl;
        components[i / 3] = separatedSIMComponents2D(patterns, otf, i);
    }
    // averaging the central frequency components
    cmat fCent = (get<0>(components[0])[0] + get<0>(components[1])[0] + get<0>(components[2])[0]) / 3;
    // Object power parameters determination
    vec OBJParaA = estimateObjectPowerParameters(fCent, otf);

    // show raw image
    int objMax = max(max(obj, 1));
    obj = obj / objMax;
    cv::Mat rawObj(w, w, CV_64F, obj._data());
    cv::imshow("testpat", testpat);
    cv::imshow("rawObj", rawObj);
//    cv::imwrite("objs.tiff", objs);
//    Mat resimg;
//    //高斯模糊
//    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
//    imshow("resimg", resimg);//显示图片
    cv::waitKey(0);
    return 0;
}

