#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>
#include <itpp_mat_utils.h>
#include <sim_utils.h>

using namespace std;
using namespace itpp;

using namespace std;

mat cvmat2mat(cv::Mat input) {
    cout << "total: " << input.total() << endl;
    cout << "type: " << input.type() << endl;
    cout << "channels: " << input.channels() << endl;
    if (input.channels() > 1) {
        cout << "this func only support single channel image!" << endl;
        std::exit(-1);
    }
    mat out(input.rows, input.cols);
    for (int i = 0; i < input.rows; ++i) {
        for (int j = 0; j < input.cols; ++j) {
            out.set(i, j, input.at<uchar>(j, i));
//            std::printf("%u, ",testpat.at<uchar>(i, j));
        }
    }
    return out;
}

void showPatternImage(Vec<mat> patterns, int w) {
    for (int i = 0; i < 9; ++i) {
        mat temp = patterns[i];
//        mat temp = real(fft2(patterns[i]));
//        temp = fftshift(temp);
        cv::Mat objs(w, w, CV_64F, temp._data());
        string name = "obj:" + std::to_string(i);
        cv::imshow(name, objs);
    }
}

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
    Vec<mat> patterns = simulateSIMImage(k2, obj, otf, modFac, noiseLevel);
    showPatternImage(patterns, obj.rows());
//    cv::Mat rawObj(w, w, CV_64F, obj._data());
//    cv::imshow("testpat", testpat);
//    cv::imshow("rawObj", rawObj);
//    cv::imwrite("objs.tiff", objs);
//    Mat resimg;
//    //高斯模糊
//    cv::GaussianBlur(img, resimg, Size(5, 5), 0);
//    imshow("resimg", resimg);//显示图片
    cv::waitKey(0);
    return 0;
}

