//
// Created by eagle on 2023/2/1.
//

#ifndef SIM_UTILS_H
#define SIM_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>
#include <utils/optimizer.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <utils/thread_pool.h>
#include <utils/sim_parameter.h>
#include <utils/OtfFactory.h>
#include <utils/neldermead.h>

using Eigen::MatrixXcd;

using namespace std;
using namespace itpp;

/**
 *
 * @param freq 结构光频率
 * @param obj 物像
 * @param otf 系统otf
 * @param modFac 调制因子
 * @param noiseLevel 加噪等级
 * @param addNoise 是否加噪，1-加/0-不加
 * @return 结构光调制图像
 */
Vec<mat>
simulateSIMImage(double freq, mat obj, const OtfFactory &otfFactory, double modFac, double noiseLevel, int addNoise) {
    int width = obj.rows();
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    // 模拟结构光场
    double alpha = 0 * pi / 6; // 初始角
    // 照明频率矢量
    mat kfreq(3, 2);
    for (int i = 0; i < 3; ++i) {
        // 结构光场角度差
        double theta = i * pi / 3.0 + alpha;
        vec thetas(2);
        thetas.set(0, cos(theta));
        thetas.set(1, sin(theta));
        kfreq.set_row(i, (freq / width) * thetas);
    }
    // 平均照明强度
    double intensity = 0.5;
    // 给三个方向的三步相移添加随机误差
    double phaseShift[9];
    vec nn = 1.0 * (0.5 - randu(9)) * pi / 18.0;
    for (int i = 0; i < 9; ++i) {
        phaseShift[i] = ((i % 3) * 2.0 * pi / 3.0) + nn(i);
    }
    // 结构光场像分布
    Vec<mat> patterns(10);
    for (int i = 0, j = 0; i < 10; ++i) {
        if (i == 9) {
            // 均匀照明光场
            patterns[i] = intensity * ones(width, width);
        } else {
            // 结构照明矢量
            patterns[i] = intensity +
                          intensity * modFac *
                          cos(2.0 * pi * (kfreq(j, 0) * (X - width / 2)
                                          + kfreq(j, 1) * (Y - width / 2))
                              + phaseShift[i]);
        }
        if ((i + 1) % 3 == 0)j++;
        // 像分布
        patterns[i] = elem_mult(obj, patterns[i]);
        // 与otf卷积
        patterns[i] = real(ifft2(elem_mult(fft2(patterns[i]), (fftshift(otfFactory.complexOtf)))));
        // Gaussian Noisy
        double sigma = std2(patterns[i]) * noiseLevel / 100.0;
        mat noisy = randn(obj.rows(), obj.cols()) * sigma;
        patterns[i] = patterns[i] + addNoise * noisy;
        //        cout << "patterns " << i << " = " << patterns[i].get_row(0) << endl;
    }
//            cout << "patterns " << " = " << patterns[0] << endl
    return patterns;
}

/**
 * 显示SIM矩阵
 * @param window name
 * @param patterns
 * @param w 图像宽度
 * @param isDivMax 是否除最大值
 */
void showPatternImage(const string &name, Vec<mat> patterns, int w, int isDivMax) {
    const int MAX_PIXEL = 300;
    int imgNum = patterns.size();
    int imgCols = 3;
    //选择图片最大的一边 将最大的边按比例变为512像素
    cv::Size imgOriSize = cv::Size(patterns[0].cols(), patterns[0].rows());
    int imgMaxPixel = max(imgOriSize.height, imgOriSize.width);
    //获取最大像素变为MAX_PIXEL的比例因子
    double prop = imgMaxPixel < MAX_PIXEL ? (double) imgMaxPixel / MAX_PIXEL : MAX_PIXEL / (double) imgMaxPixel;
    cv::Size imgStdSize(imgOriSize.width * prop, imgOriSize.height * prop); //窗口显示的标准图像的Size
    cv::Mat imgStd; //标准图片
    cv::Point2i location(0, 0); //坐标点,从(0,0)开始
    //构建窗口大小 通道与imageVector[0]的通道一样
    cv::Mat imgWindow(imgStdSize.height * ((imgNum - 1) / imgCols + 1), imgStdSize.width * imgCols,
                      CV_64F);
    for (int i = 0; i < imgNum; i++) {
        mat temp = patterns[i];
        if (isDivMax == 1) {
            int tempMax = max(max(temp, 1));
            temp = temp / tempMax;
        }
        cv::Mat obj(w, w, CV_64F, temp._data());
        location.x = (i % imgCols) * imgStdSize.width;
        location.y = (i / imgCols) * imgStdSize.height;
        cv::resize(obj, imgStd, imgStdSize, prop, prop, cv::INTER_AREA); //设置为标准大小
        imgStd.copyTo(imgWindow(cv::Rect(location, imgStdSize)));

    }
    cv::imshow(name, imgWindow);
}


#endif //SIM_UTILS_H
