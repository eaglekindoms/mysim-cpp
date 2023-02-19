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

using Eigen::MatrixXcd;

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
 * @return 结构光调制图像
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
 * @return 截止频率
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
 * @return CCop: autocorrelation of ftImage
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

/**
 *
 * @param noisyImage
 * @param otf
 * @return 估计的频域向量
 */
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
    auto phaseKai2opt0 = [=](const std::array<double, 2> &x) -> double {
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
            step, 1, 500
    );
    std::cout << "fminsearch Found minimum freq: " << std::fixed << result.xmin[0] << ' ' << result.xmin[1]
              << std::endl;
    cout << "fminsearch step: " << result.icount << endl;
    return vec(result.xmin.data(), 2);

}

/**
 * 估计初相位
 * @param noisyImage 带噪sim空域图像
 * @param freq 光场频域向量
 * @return 估计的相位
 */
double estimatePhaseShift(mat noisyImage, vec freq) {
    double phase = 0.0;// 初相位
    int width = noisyImage.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    auto phaseAutoCorrelation = [=](const std::array<double, 1> &x) -> double {
        mat sAo = cos((2 * pi * (freq[1] * (X - wo) + freq[0] * (Y - wo)) / width) + x[0]);
        mat temp = noisyImage - mean(noisyImage);
        double CCop = -sum(sum(elem_mult(temp, sAo)));
        return CCop;
    };
    std::array<double, 1> start = {phase};
    std::array<double, 1> step = {0.1};
    // very time-consuming, need to be optimized
    nelder_mead_result<double, 1> result = nelder_mead<double, 1>(
            phaseAutoCorrelation,
            start,
            1.0e-25, // the terminating limit for the variance of function values
            step, 1, 500
    );
    std::cout << "fminsearch Found minimum phase: " << std::fixed << result.xmin[0] << std::endl;
    cout << "fminsearch step: " << result.icount << endl;
    return result.xmin[0];
}

/**
 * determination of object power parameters Aobj and Bobj
 * @param fCent: FT of central frequency component
 * @param otf: system OTF
 * @return 功率谱参数
 */
vec estimateObjectPowerParameters(cmat fCent, mat otf) {
    int width = fCent.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    cmat Cv = (X - wo) + 1i * (Y - wo);
    mat Ro = abs(Cv);
    double kOtf = otfEdgeF(otf);
    mat Zm = zeros(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) > 0.3 * kOtf;
            bool r2 = Ro(i, j) < 0.4 * kOtf;
            Zm(i, j) = r1 * r2;
        }
    }
    double ObjA = sum(sum(abs(elem_mult(fCent, to_cmat(Zm))))) / sum(sum(Zm));
    double ObjB = -0.5;
    cout << "estimate object power parameters" << endl;
    std::array<double, 2> OBJpara0 = {ObjA, ObjB};
    std::array<double, 2> step = {0.1, 0.1};
    Ro(wo, wo) = 1;// to avoid nan
    // range of frequency over which SSE is computed
    mat Zloop = zeros(width, width);
    // NoisePower determination
    mat Zo = zeros(width, width);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = kOtf + 20;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) < 0.75 * kOtf;
            bool r2 = Ro(i, j) > 0.25 * kOtf;
            Zloop(i, j) = r1 * r2;
            if (Ro(i, j) > NoiseFreq) Zo(i, j) = 1;
        }
    }
    // Determined Sum of Squared Errors (SSE) between `actual signal power' and `approximated signal power'
    auto estimateSumOfSquaredErr = [=](const std::array<double, 2> &x) -> double {
        double Aobj = x[0];
        double Bobj = x[1];
        mat OBJPower = Aobj * (pow(Ro, Bobj));
        mat SIGPower = elem_mult(OBJPower, otf);
        cmat nNoise = elem_mult(fCent, to_cmat(Zo));
        complex<double> NoisePower = sum(sum(elem_mult(nNoise, conj(nNoise)))) / sum(sum(Zo));
        // Noise free object power computation
        cmat Fpower = elem_mult(fCent, conj(fCent)) - NoisePower;
        mat cent = sqrt(abs(Fpower));
        // SSE computation
        mat Error = cent - SIGPower;
        double Esum = sum(sum(elem_mult(elem_div(pow(Error, 2), Ro), Zloop)));
        return Esum;
    };
    // very time-consuming, need to be optimized
    nelder_mead_result<double, 2> result = nelder_mead<double, 2>(
            estimateSumOfSquaredErr,
            OBJpara0,
            1.0e-25, // the terminating limit for the variance of function values
            step, 1, 500
    );
    std::cout << "fminsearch Found minimum power parameters: " << std::fixed << result.xmin[0] << ' ' << result.xmin[1]
              << std::endl;
    cout << "fminsearch step: " << result.icount << endl;
    return vec(result.xmin.data(), 2);
}

/**
 * obtaining the noisy estimates of three frequency components
 * @param patterns: raw SIM images
 * @param otf: system OTF
 * @param index: phase index
 * @return noisy estimates of separated frequency components; avg. illumination frequency vector
 */
tuple<Vec<cmat>, vec> separatedSIMComponents2D(Vec<mat> patterns, mat otf, int index) {
    cout << "start estimate freq" << endl;
    ThreadPool pool(3);
    vector<future<vec>> results1;
    for (int i = 0; i < 3; i++) {
        results1.emplace_back(pool.enqueue([=] {
            return estimateFreqVector(patterns[index + i], otf);
        }));
    }
    vec freq = zeros(2);
    for (future<vec> &result: results1) {
        freq += result.get();
    }
    freq = freq / 3.0;
    cout << "mean of three order freq: " << freq << endl;
    cout << "start estimate phase" << endl;
    vec phase(3);
    vector<future<double>> results2;
    for (int i = 0; i < 3; i++) {
        results2.emplace_back(pool.enqueue([=] {
            return estimatePhaseShift(patterns[index + i], freq);
        }));
    }
    for (int i = 0; i < 3; ++i) {
        phase[i] = results2[i].get();
    }
    phase = phase * 180 / pi;
    cout << "three order phase: " << phase << endl;
    // computing PSFe for edge tapering SIM images
    mat psfd = pow(otf, 3);
    psfd = fftshift(psfd);
    psfd = ifft2(to_cmat(psfd));
    psfd = fftshift(psfd);
    psfd = psfd / max(max(psfd));
    psfd = psfd / sum(sum(psfd));
    int h = 30;
    int wo = patterns[0].rows() / 2;
    mat PSFe = psfd.get(wo - h, wo + h - 1, wo - h, wo + h - 1);
    // edge tapering raw SIM images
    Vec<cmat> ftNoisyImages(3);
    for (int i = 0; i < 3; ++i) {
        mat noisy_et = edgeTaper(patterns[index + i], PSFe);
        cmat ftNoisy = fft2(noisy_et);
        ftNoisy = fftshift(ftNoisy);
        ftNoisyImages.set(i, ftNoisy);
    }
    int MF = 1.0;
    // Transformation Matrix
    MatrixXcd M(3, 3);
    for (int k = 0; k < 3; ++k) {
        M(k, 0) = 1.0;
        M(k, 1) = 0.5 * MF * exp(-1i * phase[k]);
        M(k, 2) = 0.5 * MF * exp(+1i * phase[k]);
    }
    // Separate the components
    cout << "Separate the components" << endl;
    MatrixXcd Minv = M.inverse();
    Vec<cmat> unmixedFT(3); //  unmixed frequency components of raw SIM images
    for (int i = 0; i < 3; ++i) {
        unmixedFT[i] = Minv(i, 0) * ftNoisyImages[0]
                       + Minv(i, 1) * ftNoisyImages[1]
                       + Minv(i, 2) * ftNoisyImages[2];
    }
    return make_tuple(unmixedFT, freq);
}

/**
 * Determination of modulation factor
 * @param freqComp: off-center frequency component
 * @param freq: illumination frequency vector
 * @param OBJParaA: Object power parameters
 * @param otf: system OTF
 * @return modulation factor
 */
double estimateModulationFactor(cmat freqComp, vec freq, vec OBJParaA, mat otf) {
    int width = otf.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    cmat Cv = (X - wo) + 1i * (Y - wo);
    mat Ro = abs(Cv);
    // magnitude of illumination vector
    double k2 = sqrt(sum(pow(freq, 2)));
    // vector along illumination direction
    complex<double> kv = freq[1] + 1i * freq[0];
    mat Rp = abs(Cv + kv);
    // Object spectrum
    mat OBJp = OBJParaA[0] * pow((Rp + 0), OBJParaA[1]);
    // illumination vector rounded to the nearest pixel
    vec k3 = -round(freq);

    OBJp(wo + k3(1), wo + k3(2)) = 0.25 * OBJp(wo + 1 + k3(1), wo + k3(2))
                                   + 0.25 * OBJp(wo + k3(1), wo + 1 + k3(2))
                                   + 0.25 * OBJp(wo - 1 + k3(1), wo + k3(2))
                                   + 0.25 * OBJp(wo + k3(1), wo - 1 + k3(2));
    // signal spectrum
    mat SIGap = elem_mult(OBJp, otf);
    // OTF cut-off frequency
    double kOtf = otfEdgeF(otf);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = kOtf + 20;
    // NoisePower determination
    mat Zo = zeros(width, width);
    // frequency range over which signal power matching is done to estimate modulation factor
    mat Zmask = zeros(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) > 0.2 * k2;
            bool r2 = Ro(i, j) < 0.8 * k2;
            bool r3 = Rp(i, j) > 0.2 * k2;
            Zmask(i, j) = r1 * r2 * r3;
            if (Ro(i, j) > NoiseFreq) Zo(i, j) = 1;
        }
    }

    cmat nNoise = elem_mult(freqComp, to_cmat(Zo));
    double noisePower = sum(sum(real(elem_mult(nNoise, conj(nNoise))))) / sum(sum(Zo));

    // Noise free object power computation
    mat Fpower = real(elem_mult(freqComp, conj(freqComp))) - noisePower;
    mat fDp = sqrt(abs(Fpower));

    // least square approximation for modulation factor
    double Mm = sum(sum(elem_mult(elem_mult(SIGap, abs(fDp)), Zmask)));
    Mm = Mm / sum(sum(elem_mult(pow(SIGap, 2), Zmask)));
    cout << "estimate modulation factor: " << Mm << endl;
    return Mm;
}

/**
 * Wiener Filtering frequency component
 * @param fiSMao: noisy frequency component
 * @param otf: system OTF
 * @param co: Wiener filter constant [=1, for minimum RMS estimate]
 * @param OBJParaA: object power parameters
 * @param SFo: scaling factor (not significant here, so set to 1)
 * @param isCenter:  Filtering the central or off-center frequency component
 * @return Wiener Filtered estimate of FiSMao; avg. noise power in FiSMao
 */
tuple<cmat, double>
wienerFilterCenter(cmat fiSMao, mat otf, double co, vec OBJParaA, double SFo, bool isCenter, mat OBJsideP) {
    int width = fiSMao.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    mat Ro = sqrt(pow((X - wo), 2) + pow((Y - wo), 2));
    mat otfPower = elem_mult(otf, otf);

    // OTF cut-off frequency
    double kOtf = otfEdgeF(otf);
    // NoisePower determination
    mat Zo = zeros(width, width);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = kOtf + 20;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            if (Ro(i, j) > NoiseFreq) Zo(i, j) = 1;
        }
    }
    cmat nNoise = elem_mult(fiSMao, to_cmat(Zo));
    double noisePower = sum(sum(real(elem_mult(nNoise, conj(nNoise))))) / sum(sum(Zo));

    // Object Power determination
    mat OBJpower;
    if (isCenter) {
        Ro(wo + 1, wo + 1) = 1;
        OBJpower = OBJParaA[0] * pow(Ro, OBJParaA[1]);
    } else {
        OBJpower = OBJsideP;
    }
    OBJpower = pow(OBJpower, 2);

    // Wiener Filtering
    mat temp1 = SFo * otf / noisePower;
    mat temp2 = SFo * SFo * otfPower / noisePower + co / OBJpower;
    cmat FiSMaof = elem_div(elem_mult(fiSMao, to_cmat(temp1)), to_cmat(temp2));
    return make_tuple(FiSMaof, noisePower);
}

/**
 * obtaining Wiener Filtered estimates of noisy frequency components
 * @param component: noisy estimates of separated frequency component
 * @param OBJParaA: object power parameters
 * @param otf: system OTF
 */
tuple<Vec<tuple<cmat, double>>, double> wienerFilter(tuple<Vec<cmat>, vec> component, vec OBJParaA, mat otf) {
    int width = otf.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    cmat Cv = (X - wo) + 1i * (Y - wo);
    mat Ro = abs(Cv);
    double kOtf = otfEdgeF(otf);
    // Wiener Filtering central frequency component
    double SFo = 1;
    double co = 1.0;
    vec kA = get<1>(component);
    tuple<cmat, double> fDof = wienerFilterCenter(get<0>(component)[0], otf, co, OBJParaA, SFo, true, Ro);
    // modulation factor determination
    double Mm = estimateModulationFactor(get<0>(component)[1], kA, OBJParaA, otf);
    // Duplex power (default)
    complex<double> kv = kA[1] + 1i * kA[0]; // vector along illumination direction
    mat Rp = abs(Cv - kv);
    mat Rm = abs(Cv + kv);
    mat OBJp = OBJParaA[0] * pow(Rp, OBJParaA[1]);
    mat OBJm = OBJParaA[0] * pow(Rm, OBJParaA[1]);
    vec k3 = round(kA);
    OBJp(wo + k3(1), wo + k3(2)) = 0.25 * OBJp(wo + 1 + k3(1), wo + k3(2))
                                   + 0.25 * OBJp(wo + k3(1), wo + 1 + k3(2))
                                   + 0.25 * OBJp(wo - 1 + k3(1), wo + k3(2))
                                   + 0.25 * OBJp(wo + k3(1), wo - 1 + k3(2));
    OBJm(wo - k3(1), wo - k3(2)) = 0.25 * OBJm(wo + 1 - k3(1), wo - k3(2))
                                   + 0.25 * OBJm(wo - k3(1), wo + 1 - k3(2))
                                   + 0.25 * OBJm(wo - 1 - k3(1), wo - k3(2))
                                   + 0.25 * OBJm(wo - k3(1), wo - 1 - k3(2));
    // Filtering side lobes (off-center frequency components)
    SFo = Mm;
    tuple<cmat, double> fDpf = wienerFilterCenter(get<0>(component)[1], otf, co, OBJParaA, SFo, false, OBJm);
    tuple<cmat, double> fDmf = wienerFilterCenter(get<0>(component)[2], otf, co, OBJParaA, SFo, false, OBJp);
    // doubling Fourier domain size if necessary
    /* TODO */
    // Shifting the off-center frequency components to their correct location
    cmat fDp1 = fft2(elem_mult(to_cmat(ifft2(get<0>(fDpf))),
                               exp(1i * 2 * pi * (kA(1) / width * (X - wo) + kA(0) / width * (Y - wo)))));
    cmat fDm1 = fft2(elem_mult(to_cmat(ifft2(get<0>(fDmf))),
                               exp(-1i * 2 * pi * (kA(1) / width * (X - wo) + kA(0) / width * (Y - wo)))));
    // Shift induced phase error correction
    double k2 = sqrt(sum(pow(kA, 2)));
    // frequency range over which corrective phase is determined
    mat Zmask = zeros(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) < 0.8 * k2;
            bool r2 = Rp(i, j) < 0.8 * k2;
            Zmask(i, j) = r1 * r2;
        }
    }
    // corrective phase
    cvec Angle(1);
    Angle[0] = sum(sum(elem_mult(elem_mult(get<0>(fDof), conj(fDp1)), to_cmat(Zmask))));
    double Angle0 = angle(Angle)[0];

    // phase correction
    cmat fDp2 = exp(+1i * Angle0) * fDp1;
    cmat fDm2 = exp(-1i * Angle0) * fDm1;
    Vec<tuple<cmat, double>> freqComps(3);
    freqComps[0] = fDof;
    freqComps[1] = make_tuple(fDp2, get<1>(fDpf));
    freqComps[2] = make_tuple(fDm2, get<1>(fDmf));
    return make_tuple(freqComps, Mm);
}

/**
 * 显示SIM矩阵
 * @param window name
 * @param patterns
 * @param w 图像宽度
 * @param isDivMax 是否除最大值
 */
void showPatternImage(string name, Vec<mat> patterns, int w, int isDivMax) {
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
