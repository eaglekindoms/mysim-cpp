//
// Created by eagle on 2023/2/1.
//

#ifndef SIM_UTILS_H
#define SIM_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>

using namespace std;
using namespace itpp;

/**
 *
 * @param freq 结构光频率
 * @param obj 物象
 * @param otf 系统otf
 * @param modFac 调制因子
 * @param noiseLevel 加噪等级
 */
Vec<mat> simulateSIMImage(double freq, mat obj, mat otf, double modFac, double noiseLevel) {
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
        int patMax = max(max(patterns[i], 1));
        patterns[i] = patterns[i] / patMax;
        //        cout << "patterns " << i << " = " << patterns[i].get_row(0) << endl;
    }
//            cout << "patterns " << " = " << patterns[0] << endl;

    return patterns;
}

#endif //SIM_UTILS_H
