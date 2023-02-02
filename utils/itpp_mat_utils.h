#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>

using namespace std;
using namespace itpp;

template<class T>
inline void minMat(Mat<T> &x, Mat<T> &y) {
    int row = x.rows();
    int col = x.cols();
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            if (x(i, j) >= y(i, j)) {
                x(i, j) = y(i, j);
            }
        }
    }
}

template<class T>
inline cmat mat2cmat(Mat<T> &x) {
    cmat cm(x.rows(), x.cols());
    for (int i = 0; i < x.rows(); ++i) {
        for (int j = 0; j < x.cols(); ++j) {
            std::complex<double> c(x.get(i, j), 0);
            cm.set(i, j, c);
        }
    }
    return cm;
}

template<class T>
inline Mat<T> fftshift(Mat<T> &x) {
    Mat<T> out = x;
    int i = 0, j = 0;
    while ((i < x.rows() / 2) && (j < x.cols() / 2)) {
        out.swap_rows(i, x.rows() / 2 + i);
        out.swap_cols(j, x.cols() / 2 + j);
        i++;
        j++;
    }
    return out;
}

/**
 * 二维FFT原理
 * 二维FFT是在一维FFT基础上实现，实现过程为：
 * 1.对二维输入数据的每一行进行FFT，变换结果仍然按行存入二维数组中。
 * 2.在1的结果基础上，对每一列进行FFT，再存入原来数组，及得到二维FFT结果。
 * @param 实数矩阵
 * @return
 */
inline cmat fft2(const mat &x) {
//    cout << "have_fourier_transforms: " << have_fourier_transforms() << endl;
    cmat cx(x.rows(), x.cols());
    cvec temp;
    for (int i = 0; i < x.rows(); ++i) {
        temp = fft_real(x.get_row(i));
        cx.set_row(i, temp);
    }
    for (int i = 0; i < x.cols(); ++i) {
        temp = fft(cx.get_col(i));
        cx.set_col(i, temp);
    }
    return cx;
}

/**
 * 二维逆傅里叶变换
 * @param cx
 * @return
 */
inline mat ifft2(const cmat &cx) {
//    cout << "have_fourier_transforms: " << have_fourier_transforms() << endl;
    cmat x(cx.rows(), cx.cols());
    mat out(cx.rows(), cx.cols());
    for (int i = 0; i < cx.rows(); ++i) {
        cvec temp = ifft(cx.get_row(i));
        x.set_row(i, temp);
    }
    for (int i = 0; i < cx.cols(); ++i) {
        vec temp = ifft_real(x.get_col(i));
        out.set_col(i, temp);
    }
    return out;
}

/**
 * @brief generator psf by besselj functions
 * @param width: generate [W:W] mat
 * @param scale: a parameter used to adjust PSF width
 * @return
 */
inline mat generatorPSF(int width, double scale) {
    cout << "================================" << endl;
    cout << "    generator psf by besselj functions " << endl;
    cout << "================================" << endl;
    vec x, y;
    mat X, Y;
    x = linspace(0, width - 1, width);
    for (int i = 0; i < width; i++) {
        y = concat(x, y);
    }
    X = mat(y._data(), width, width);
    Y = mat(y._data(), width, width, false);
    mat tempMat = abs(X - width);
    minMat(X, tempMat);
    tempMat = abs(Y - width);
    minMat(Y, tempMat);
    mat R = elem_mult(X, X) + elem_mult(Y, Y);
    R = scale * sqrt(R);
    mat PSF = mat(width, width);
    for (int i = 0; i < width; ++i) {
        vec temp = 2 * besselj(1, R.get_row(i) + eps);
        tempMat.set_row(i, temp);
    }
    tempMat = elem_div(tempMat, R + eps);
    tempMat = abs(tempMat);
    tempMat = elem_mult(tempMat, tempMat);
    tempMat = fftshift(tempMat);
    //    cout << "tempMat = " << tempMat << endl;
    PSF = tempMat;
    return PSF;
}

/**
 * 将psf转换为otf
 * @param psf
 * @return otf
 */
inline mat PSFToOTF(const mat &psf) {
    cmat otf = fft2(psf);
// 归一化矩阵
    complex<double> otfMax = max(max(abs(otf), 1));
//    cout << "otfmax = " << otfmax << endl;
    otf = otf / otfMax;
    otf = fftshift(otf);
//    cout << "otf = " << otf << endl;
    mat aotf = abs(otf);
    return aotf;
}

/**
 * 求矩阵标准差
 * @param x 矩阵
 * @return 标准差
 */
inline double std2(mat &x) {
    vec col(x._data(), x.cols() * x.rows());
    return sqrt(variance(col));
}

#endif // MAT_UTILS_H
