#include <opencv2/opencv.hpp>
#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <iomanip>

using namespace std;
using namespace itpp;

template<class T>
void minMat(Mat<T>  &x, Mat<T> &y){
    int row=x.rows();
    int col=x.cols();
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if(x(i,j)>=y(i,j)) x(i,j)=y(i,j);
        }
    }
}

void fftshift(const cv::Mat& inputImg, cv::Mat& outputImg);
mat generatorPSF(int width, double scale);

int main()
{

  cout << "================================" << endl;
  cout << "    Test of bessel functions " << endl;
  cout << "================================" << endl;

  vec x = linspace(0.01, 10, 20);

  cout << "x = " << x << endl;

  cout << "besselj(0, x) = " << fixed << besselj(0, x) << endl;
  cout << "besselj(1, x) = " << besselj(1, x) << endl;
  cout << "besselj(5, x) = " << round_to_infty(besselj(5, x)) << endl;
  cout << "besselj(0.3, x) = " << besselj(0.3, x) << endl;
  cout << "besselj(1.7, x) = " << besselj(1.7, x) << endl;
  cout << "besselj(5.3, x) = " << round_to_infty(besselj(5.3, x)) << endl;

  cout << "bessely(0, x) = " << bessely(0, x) << endl;
  cout << "bessely(1, x) = " << bessely(1, x) << endl;
  cout << "bessely(5, x) = " << round_to_infty(bessely(5, x)) << endl;
  cout << "bessely(0.3, x) = " << bessely(0.3, x) << endl;
  cout << "bessely(1.7, x) = " << bessely(1.7, x) << endl;
  cout << "bessely(5.3, x) = " << round_to_infty(bessely(5.3, x)) << endl;

  cout << "besseli(0, x) = " << besseli(0, x) << endl;
  cout << "besseli(1, x) = " << besseli(1, x) << endl;
  cout << "besseli(5, x) = " << besseli(5, x) << endl;
  cout << "besseli(0.3, x) = " << besseli(0.3, x) << endl;
  cout << "besseli(1.7, x) = " << besseli(1.7, x) << endl;
  cout << "besseli(5.3, x) = " << besseli(5.3, x) << endl;

  cout << "besselk(0, x) = " << besselk(0, x) << endl;
  cout << "besselk(1, x) = " << besselk(1, x) << endl;
  cout << "besselk(5, x) = " << round_to_infty(besselk(5, x)) << endl;
  int w=512;
  mat psf=generatorPSF(w,0.3);
  cv::Mat cvMat(w,w,CV_64F);
  cv::Mat cvMat1(w,w,CV_64F,psf._data());
  fftshift(cvMat1,cvMat);
//  cout<< "cvpsf = "<< cvMat<<endl;
  cv::imshow("cvMat",cvMat);
  cv::imwrite("psf.tiff",cvMat);
  cv::waitKey(0);
  return 0;

}

mat generatorPSF(int width, double scale){
    vec x,y;mat X,Y;
    x=linspace(0,width-1,width);
    for(int i=0;i<width;i++) y = concat(x,y);
    X=mat(y._data(),width,width);
    Y=mat(y._data(),width,width,false);
//    cout << "X = " << X << endl;
//    cout << "Y = " << Y << endl;
    mat tempMat=abs(X-width);
//    cout << "absX = " << tempMat << endl;
    minMat(X,tempMat);
//    cout << "minX = " << X << endl;
    tempMat=abs(Y-width);
    minMat(Y,tempMat);
//    cout << "minY = " << Y << endl;
    mat R=elem_mult(X,X)+elem_mult(Y,Y);
    R=scale*sqrt(R);
    mat PSF=mat(width,width);
//    cout << "R = " << R << endl;
    for (int i = 0; i < width; ++i) {
        vec temp=2*besselj(1, R.get_row(i)+eps);
        tempMat.set_row(i,temp);
    }
    tempMat=elem_div(tempMat,R+eps);
    tempMat=abs(tempMat);
    tempMat=elem_mult(tempMat,tempMat);
//    cout << "tempMat = " << tempMat << endl;
//    mat fht=dht2(tempMat) ;
//    cout << "fht = " << fht << endl;

    PSF=tempMat;
    return PSF;
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
