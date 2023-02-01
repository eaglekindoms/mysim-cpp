#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include <itpp_mat_utils.h>
#include <sim_utils.h>

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
    testpat = testpat.rowRange(257, 768);
    testpat = testpat.colRange(257, 768);
    double k2 = 75.23; // illumination freq
    double modFac = 0.8;// modulation factor
    double noiseLevel = 10.; // in percentage
    cout << "type: " << testpat.type() << endl;
    cout << "channels: " << testpat.channels() << endl;
    mat obj(testpat.rows, testpat.cols);
    for (int i = 0; i < testpat.rows; ++i) {
        for (int j = 0; j < testpat.cols; ++j) {
            obj.set(i,j,testpat.at<uchar>(i,j));
        }
    }
    Vec<mat> patterns= simulateSIMImage(k2,obj,otf,modFac,noiseLevel);
//    cout << "temp" << testpat << endl;
    cv::Mat objs(w, w, CV_64F, patterns[0]._data());
//    cout << "objs" << objs << endl;
//    cv::imshow("testpat", testpat);
//    cv::imshow("objs", objs);
    cv::imwrite("objs.tiff", objs);
//    Mat resimg;
//    //高斯模糊
//    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
//    imshow("resimg", resimg);//显示图片
    cv::waitKey(0);
    return 0;
}

