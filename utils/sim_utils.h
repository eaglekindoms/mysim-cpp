//
// Created by eagle on 2023/2/1.
//

#ifndef SIM_UTILS_H
#define SIM_UTILS_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>
#include <utils/optimizer.h>

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
        cmat cotf = to_cmat(otf);
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
 * 获取otf截止频率
 * @param otf
 * @return
 */
double otfEdgeF(mat otf) {
    double kOtf;
    int w = otf.rows();
    int wo = w / 2;
    vec otf1 = otf.get_row(wo);
    double otfMax = max(max(abs(otf)));
    double otfTruncate = 0.01;
    for (int i = 0; i < w && abs(otf1[i]) < otfTruncate * otfMax; i++) {
        kOtf = wo - i;
    }
    return kOtf;
}

/**
 * illumination frequency vector determination
 * @param ftImage0 FT of raw SIM image
 * @param kOtf OTF cut-off frequency
 * @return  maxK2: illumination frequency vector (approx); Ix,Iy: coordinates of illumination frequency peaks
 */
Vec<ivec> approxFreqDuplex(cmat ftImage0, double kOtf) {
    mat ftImage = abs(ftImage0);
    int width = ftImage.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }

    mat Ro = sqrt(pow((X - wo), 2) + pow((Y - wo), 2));
    mat Z0 = zeros(width, width);
    mat Z1 = zeros(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (Ro.get(i, j) > itpp::round(0.5 * kOtf)) {
                Z0.set(i, j, 1);
            }
            if (X.get(i, j) > wo) {
                Z1.set(i, j, 1);
            }
        }
    }
    ftImage = elem_mult(ftImage, Z0);
    ftImage = elem_mult(ftImage, Z1);

    vec dumY = max(ftImage, 1);
    int Iy = max_index(dumY);
    vec dumX = max(ftImage, 2);
    int Ix = max_index(dumX);
    Vec<ivec> result(2);
    int maxK2[] = {Ix - wo, Iy - wo};
    int posit[] = {Ix, Iy};
    result.set(0, ivec(maxK2, 2));
    result.set(1, ivec(posit, 2));
    return result;
}

/**
 *  Compute autocorrelation of FT of raw SIM images
 * @param freq: illumination frequency vector
 * @param ftImage: FT of raw SIM image
 * @param otf: system OTF
 * @param opt: acronym for `OPTIMIZE'; to be set to 1 when this function is used for optimization, or else to 0
 * @return CCop: autocorrelation of fS1aTnoisy
 */
double phaseAutoCorrelationFreqByOpt(vec freq, cmat ftImage, mat otf, bool opt) {
    int width = ftImage.rows();
    int wo = width / 2;
    cmat cotf = to_cmat(otf);
    ftImage = elem_mult(ftImage, to_cmat(1 - pow(otf, 10)));
    cmat fS1aT = elem_mult(ftImage, conj(cotf));

    double kOtf = otfEdgeF(otf);
    bool DoubleMatSize = false;
    if (2.0 * kOtf > wo) {
        // true for doubling fourier domain size, false for keeping it unchanged
        DoubleMatSize = true;
    }
    int t;
    if (DoubleMatSize) {
        t = 2 * width;
        cmat fS1aT_temp = zeros_c(t, t);
        fS1aT_temp.set_submatrix(wo, wo, fS1aT);
        fS1aT = fS1aT_temp;
    } else {
        t = width;
    }
    int to = t / 2;
    mat U(width, width), V(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        U.set_row(i, line);
        V.set_col(i, line);
    }
    cmat S1aT = exp(std::complex<double>(0, -2 * pi) * (freq[1] / t * (U - to) + freq[0] / t * (V - to)));
    S1aT = elem_mult(S1aT, to_cmat(ifft2(fS1aT)));
    cmat fS1aT0 = fft2(S1aT);

    complex<double> mA = sum(sum(elem_mult(fS1aT, conj(fS1aT0))));
    mA = mA / sum(sum(elem_mult(fS1aT0, conj(fS1aT0))));
    double CCop = -abs(mA);
    return CCop;
}

vec estimateFreqVector(mat noisyImage, mat otf) {
    // computing PSFe for edge tapering SIM images
    int w = otf.rows();
    int wo = w / 2;
    mat psfd = pow(otf, 10);
    psfd = fftshift(psfd);
    cmat cpsfd = to_cmat(psfd);
    psfd = ifft2(cpsfd);
    psfd = fftshift(psfd);
    psfd = psfd / max(max(psfd));
    psfd = psfd / sum(sum(psfd));
    int h = 30;
    mat PSFe = psfd.get(wo - h, wo + h - 1, wo - h, wo + h - 1);
    // edge tapering raw SIM image
    mat noisy_et = edgeTaper(noisyImage, PSFe);
    cmat fNoisy_et = fft2(noisy_et);
    fNoisy_et = fftshift(fNoisy_et);
    int kOtf = otfEdgeF(otf);
    cout << "kOtf: " << kOtf << endl;
    Vec<ivec> freqVector = approxFreqDuplex(fNoisy_et, kOtf);
    cout << "freqVector: " << freqVector << endl;
    cmat fS1aTnoisy = fft2(noisyImage);
    fS1aTnoisy = fftshift(fS1aTnoisy);
    cout << "==== fminsearch ====" << endl;
    auto phaseKai2opt0 = [&fS1aTnoisy, &otf](const std::array<double, 2> &x) -> double {
        vec freq(x.data(), 2);
        return phaseAutoCorrelationFreqByOpt(freq, fS1aTnoisy, otf, true);
    };
    std::array<double, 2> start = {double(freqVector[0].get(0)), double(freqVector[0].get(1))};
    std::array<double, 2> step = {0.1, 0.1};
    // very time-consuming, need to be optimized
    nelder_mead_result<double, 2> result = nelder_mead<double, 2>(
            phaseKai2opt0,
            start,
            1.0e-25, // the terminating limit for the variance of function values
            step
    );
    std::cout << "fminsearch Found minimum: " << std::fixed << result.xmin[0] << ' ' << result.xmin[1] << std::endl;
    cout << "fminsearch step: " << result.icount << endl;
    return vec(result.xmin.data(), 2);

}

void estimatePhaseShift(mat noisyImage, vec freq) {}

void estimateSIMParam() {}

void separatedSIMComponents2D(Vec<mat> patterns, mat otf) {
    vec freq = estimateFreqVector(patterns[0], otf);
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
