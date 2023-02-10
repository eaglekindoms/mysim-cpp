#ifndef MAT_UTILS_H
#define MAT_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace itpp;


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
template<class T>
inline cmat fft2(const Mat<T> &input) {
//    cout << "have_fourier_transforms: " << have_fourier_transforms() << endl;
    cmat x = to_cmat(input);
    cmat cx(x.rows(), x.cols());
    cvec temp;
    for (int i = 0; i < x.rows(); ++i) {
        temp = fft(x.get_row(i));
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

/**
 * 对图像边缘进行模糊处理，基于matlab的cpp实现
 * @param image
 * @param psf
 */
inline mat edgeTaper(mat image, mat psf) {
    // 1. Compute the weighting factor alpha used for image windowing,
    // alpha=1 within the interior of the picture and alpha=0 on the edges.
    // Normalize positive PSF
    psf = psf / sum(sum(psf));
    mat alpha(image.rows(), image.cols());
    Vec<vec> beta(2);
    vec psfProjr = zeros(psf.rows());
    vec psfProjc = zeros(psf.rows());
    for (int i = 0, j = 0; i < psf.rows() || j < psf.cols(); i++, j++) {
        if (i < psf.rows()) {
            psfProjr.set(i, sum(psf.get_row(i)));
        }
        if (j < psf.cols()) {
            psfProjc.set(i, sum(psf.get_col(i)));
        }
    }
    auto psf_proj = [](vec proj, int size) -> vec {
        cvec fProj = fft_real(proj, size);
        vec temp = pow(abs(fProj), 2);
        fProj = to_cvec(temp);
        vec z = real(ifft(fProj, size));
        double maxZ = max(z);
        z = concat(z, 1.0);
        z = z / maxZ;
        return z;
    };
    beta.set(0, psf_proj(psfProjr, image.rows() - 1));
    beta.set(1, psf_proj(psfProjc, image.cols() - 1));
    for (int i = 0; i < beta[0].size(); ++i) {
        alpha.set_row(i, (1.0 - beta[0].get(i)) * (1.0 - beta[1]));
    }
    // 2. Blur input image by PSF & weight it and input image with factor alpha
    mat fixPsf = zeros(image.rows(), image.cols());
    fixPsf.set_submatrix(image.rows() / 2 - psf.rows() / 2, image.cols() / 2 - psf.cols() / 2, psf);
    fixPsf = fftshift(fixPsf);
    cmat otf = fft2(fixPsf);
    cmat fImage = elem_mult(fft2(image), otf);
    mat blurredI = ifft2(fImage);
    mat result = elem_mult(alpha, image) + elem_mult(1 - alpha, blurredI);
    double maxI = max(max(image));
    double minI = min(min(image));
    // Bound result image by the same range as input image
    for (int i = 0; i < result.rows(); ++i) {
        for (int j = 0; j < result.cols(); ++j) {
            if (result.get(i, j) > maxI)result.set(i, j, maxI);
            if (result.get(i, j) < minI)result.set(i, j, minI);
        }
    }
    return result;
}

#endif // MAT_UTILS_H
