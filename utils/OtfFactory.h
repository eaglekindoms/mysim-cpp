//
// Created by eagle on 2023/2/22.
//

#ifndef OTFFACTORY_H
#define OTFFACTORY_H

#include <itpp/itbase.h>
#include <iostream>
#include <utils/itpp_mat_utils.h>

class OtfFactory {
public:
    mat psf;
    mat otf;
    cmat complexOtf;
    double cutOff;

    OtfFactory(int width, double scale) {
        psf = generatorPSF(width, scale);
        otf = PSFToOTF(psf);
        complexOtf = to_cmat(otf);
        cutOff = otfEdgeF(otf);
    }

    /**
     * @brief generator psf by besselj functions
     * @param width: generate [W:W] mat
     * @param scale: a parameter used to adjust PSF width
     * @return
     */
    static inline mat generatorPSF(int width, double scale) {
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
     * 获取otf截止频率
     * @param otf
     * @return 截止频率
     */
    static double otfEdgeF(mat otf) {
        double kOtf;
        int w = otf.rows();
        int wo = w / 2;
        vec otf1 = otf.get_row(wo);
        double otfMax = max(max(abs(otf)));
        double otfTruncate = 0.01;
        for (int i = 0; i < w && abs(otf1[i]) < otfTruncate * otfMax; i++) {
            kOtf = wo - i;
        }
        cout << "the cutoff of OTF is: " << kOtf << endl;
        return kOtf;
    }

};

#endif //OTFFACTORY_H
