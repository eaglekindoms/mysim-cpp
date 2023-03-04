//
// Created by eagle on 2023/3/4.
//

#ifndef OPENSIM_ESTIMATE_H
#define OPENSIM_ESTIMATE_H

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
 * illumination frequency vector determination
 * @param ftImage0 FT of raw SIM image
 * @param kOtf OTF cut-off frequency
 * @return  maxK2: illumination frequency vector (approx); Ix,Iy: coordinates of illumination frequency peaks
 */
Vec<ivec> approxFreqDuplex(const cmat &ftImage0, double kOtf) {
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
 *
 * @param ftImage: FT of raw SIM image
 * @param otfFactory: OTF
 * @return uv矩阵, 与otf共轭化后的ftImage
 */
tuple<Vec<mat>, cmat> computeUVMat(const cmat &ftImage, const OtfFactory &otfFactory) {
    mat otf = otfFactory.otf;
    int width = ftImage.rows();
    int wo = width / 2;
    cmat ftImage1 = elem_mult(ftImage, to_cmat(1 - pow(otf, 10)));
    cmat fS1aT = elem_mult(ftImage1, conj(otfFactory.complexOtf));

    bool DoubleMatSize = false;
    if (2.0 * otfFactory.cutOff > wo) {
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
    Vec<mat> result(2);
    result[0] = U - to;
    result[1] = V - to;
    return make_tuple(result, fS1aT);
}

/**
 *  Compute autocorrelation of FT of raw SIM images
 * @param freq: illumination frequency vector
 * @param fS1aT: FT of raw SIM image
 * @param otf: system OTF
 * @param opt: acronym for 'OPTIMIZE'; to be set to 1 when this function is used for optimization, or else to 0
 * @return CCop: autocorrelation of ftImage
 */
double
phaseAutoCorrelationFreqByOpt(vec freq, const cmat &fS1aT, const OtfFactory &otfFactory, bool opt, Vec<mat> uv, int t) {
    mat U = uv[0];
    mat V = uv[1];
    cmat S1aT = exp(std::complex<double>(0, -2 * pi) * (freq[1] / t * U + freq[0] / t * V));
    S1aT = elem_mult(S1aT, ifft2(fS1aT));
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
vec estimateFreqVector(const mat &noisyImage, const OtfFactory &otfFactory) {
    // computing PSFe for edge tapering SIM images
    mat otf = otfFactory.otf;
    int w = otf.rows();
    int wo = w / 2;
    mat psfd = pow(otf, 10);
    psfd = fftshift(psfd);
    cmat cpsfd = to_cmat(psfd);
    psfd = real(ifft2(cpsfd));
    psfd = fftshift(psfd);
    psfd = psfd / max(max(psfd));
    psfd = psfd / sum(sum(psfd));
    int h = 30;
    mat PSFe = psfd.get(wo - h, wo + h - 1, wo - h, wo + h - 1);
    // edge tapering raw SIM image
    mat noisy_et = edgeTaper(noisyImage, PSFe);
    cmat fNoisy_et = fft2(noisy_et);
    fNoisy_et = fftshift(fNoisy_et);
    Vec<ivec> freqVector = approxFreqDuplex(fNoisy_et, otfFactory.cutOff);
    cout << "freqVector: " << freqVector << endl;
    cmat fS1aTnoisy = fft2(noisyImage);
    fS1aTnoisy = fftshift(fS1aTnoisy);
    cout << "==== fminsearch ====" << endl;
    vector<double> first;
    first.push_back(freqVector[0].get(0));
    first.push_back(freqVector[0].get(1));
    tuple<Vec<mat>, cmat> uvs = computeUVMat(fS1aTnoisy, otfFactory);
    Vec<mat> uv = get<0>(uvs);
    cmat fS1aT = get<1>(uvs);
    int t = uv[0].rows();
    auto phaseKai2opt1 = [=](const std::vector<double> &x) -> double {
        vec freq(x.data(), 2);
        return phaseAutoCorrelationFreqByOpt(freq, fS1aT, otfFactory, true, uv, t);
    };
    RealFunctionvalueAtCoordinate result = nelderMead(phaseKai2opt1, first);
    std::cout << "fminsearch Found minimum freq: " << std::fixed << result.coordinate_[0] << ' '
              << result.coordinate_[1] << std::endl;
    return vec(result.coordinate_.data(), 2);
}

/**
 * 估计初相位
 * @param noisyImage 带噪sim空域图像
 * @param freq 光场频域向量
 * @return 估计的相位
 */
double estimatePhaseShift(const mat &noisyImage, vec freq) {
    double phase = 0.0;// 初相位
    int width = noisyImage.rows();
    int wo = width / 2;
    mat X(width, width), Y(width, width);
    vec line = linspace(0, width - 1, width);
    for (int i = 0; i < width; ++i) {
        X.set_row(i, line);
        Y.set_col(i, line);
    }
    vector<double> first;
    first.push_back(phase);
    auto phaseAutoCorrelation1 = [=](const std::vector<double> &x) -> double {
        mat sAo = cos((2 * pi * (freq[1] * (X - wo) + freq[0] * (Y - wo)) / width) + x[0]);
        mat temp = noisyImage - mean(noisyImage);
        double CCop = -sum(sum(elem_mult(temp, sAo)));
        return CCop;
    };
    RealFunctionvalueAtCoordinate result = nelderMead(phaseAutoCorrelation1, first);
    std::cout << "fminsearch Found minimum phase: " << std::fixed << result.coordinate_[0] << std::endl;
    return result.coordinate_[0];
}

/**
 * determination of object power parameters Aobj and Bobj
 * @param fCent: FT of central frequency component
 * @param otf: system OTF
 * @return 功率谱参数
 */
vec estimateObjectPowerParameters(const cmat &fCent, const OtfFactory &otfFactory) {
    mat otf = otfFactory.otf;
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
    mat Zm = zeros(width, width);
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) > 0.3 * otfFactory.cutOff;
            bool r2 = Ro(i, j) < 0.4 * otfFactory.cutOff;
            Zm(i, j) = r1 * r2;
        }
    }
    double ObjA = sum(sum(abs(elem_mult(fCent, to_cmat(Zm))))) / sum(sum(Zm));
    double ObjB = -0.5;
    cout << "estimate object power parameters" << endl;
    Ro(wo, wo) = 1;// to avoid nan
    // range of frequency over which SSE is computed
    mat Zloop = zeros(width, width);
    // NoisePower determination
    mat Zo = zeros(width, width);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = otfFactory.cutOff + 20;
    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            bool r1 = Ro(i, j) < 0.75 * otfFactory.cutOff;
            bool r2 = Ro(i, j) > 0.25 * otfFactory.cutOff;
            Zloop(i, j) = r1 * r2;
            if (Ro(i, j) > NoiseFreq) Zo(i, j) = 1;
        }
    }
    cmat nNoise = elem_mult(fCent, to_cmat(Zo));
    complex<double> NoisePower = sum(sum(elem_mult(nNoise, conj(nNoise)))) / sum(sum(Zo));
    // Noise free object power computation
    cmat Fpower = elem_mult(fCent, conj(fCent)) - NoisePower;
    mat cent = sqrt(abs(Fpower));
    // Determined Sum of Squared Errors (SSE) between `actual signal power' and `approximated signal power'
    auto estimateSumOfSquaredErr = [=](const std::vector<double> &x) -> double {
        double Aobj = x[0];
        double Bobj = x[1];
        mat OBJPower = Aobj * (pow(Ro, Bobj));
        mat SIGPower = elem_mult(OBJPower, otf);
        // SSE computation
        mat Error = cent - SIGPower;
        double Esum = sum(sum(elem_mult(elem_div(pow(Error, 2), Ro), Zloop)));
        return Esum;
    };
    std::vector<double> OBJpara0 = {ObjA, ObjB};
    RealFunctionvalueAtCoordinate result = nelderMead(estimateSumOfSquaredErr, OBJpara0);
    std::cout << "fminsearch Found minimum power parameters: " << std::fixed << result.coordinate_[0] << ' '
              << result.coordinate_[1] << std::endl;
    return vec(result.coordinate_.data(), 2);
}


/**
 * Determination of modulation factor
 * @param freqComp: off-center frequency component
 * @param freq: illumination frequency vector
 * @param OBJPara: Object power parameters
 * @param otf: system OTF
 * @return modulation factor
 */
double estimateModulationFactor(const cmat &freqComp, vec freq, vec OBJPara, const OtfFactory &otfFactory) {
    mat otf = otfFactory.otf;
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
    mat OBJp = OBJPara[0] * pow((Rp + 0), OBJPara[1]);
    // illumination vector rounded to the nearest pixel
    vec k3 = -round(freq);

    OBJp(wo + k3(0), wo + k3(1)) = 0.25 * OBJp(wo + 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo + 1 + k3(1))
                                   + 0.25 * OBJp(wo - 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo - 1 + k3(1));
    // signal spectrum
    mat SIGap = elem_mult(OBJp, otf);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = otfFactory.cutOff + 20;
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
 * estimate freq vector and phase of three frequency components
 * @param patterns: raw sim images
 * @param otf: system otf
 * @param index: orientation index
 * @return: avg.freq and three phase shift
 */
Orientation estimateSIMParameters(const Vec<mat> &patterns, const OtfFactory &otfFactory, int index) {
    cout << "start estimate SIM parameter" << endl;
    ThreadPool pool(3);
    vector<future<vec>> results1;
    for (int i = 0; i < 3; i++) {
        results1.emplace_back(pool.enqueue([=] {
            return estimateFreqVector(patterns[index + i], otfFactory);
        }));
    }
    vec freq = zeros(2);
    for (future<vec> &result: results1) {
        freq += result.get();
    }
    freq = freq / 3.0;
    Orientation ori(index);
    ori.freq = freq;
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
        ori.phaseShift[i] = phase[i];
    }
    cout << "three order phase: " << phase * 180 / pi << endl;
    return ori;
}

#endif //OPENSIM_ESTIMATE_H
