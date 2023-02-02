//
// Created by eagle on 2023/2/1.
//

#ifndef SIM_UTILS_H
#define SIM_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>

using namespace std;
using namespace itpp;

/**
 *
 * @param freq 结构光频率
 * @param obj 物象
 * @param otf 系统otf
 * @param modFac 调制因子
 * @param noiseLevel 加噪等级
 * @param addNoise 是否加噪，1-加/0-不加
 */
Vec<mat> simulateSIMImage(double freq, mat obj, mat otf, double modFac, double noiseLevel, int addNoise) {
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
    Vec<mat> patterns(9);
    for (int i = 0, j = 0; i < 9; ++i) {
        // 照明矢量
        patterns[i] = intensity +
                      intensity * modFac *
                      cos(2.0 * pi * (kfreq(j, 0) * (X - width / 2)
                                      + kfreq(j, 1) * (Y - width / 2))
                          + phaseShift[i]);
        if ((i + 1) % 3 == 0)j++;
        // 像分布
        patterns[i] = elem_mult(obj, patterns[i]);
        // 与otf卷积
        cmat cotf = mat2cmat(otf);
        patterns[i] = ifft2(elem_mult(fft2(patterns[i]), (fftshift(cotf))));
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
 * @param patterns
 * @param w 图像宽度
 * @param isFreq 是否显示频域图像，1-是/0-否
 */
void showPatternImage(Vec<mat> patterns, int w, int isFreq) {
    for (int i = 0; i < 9; ++i) {
        mat temp = patterns[i];
        if (isFreq == 1) {
            temp = real(fft2(patterns[i]));
            temp = fftshift(temp);
        } else {
            int tempMax = max(max(temp, 1));
            temp = temp / tempMax;
        }
        cv::Mat objs(w, w, CV_64F, temp._data());
        string name = "obj:" + std::to_string(i);
        cv::imshow(name, objs);
    }
}

#endif //SIM_UTILS_H
