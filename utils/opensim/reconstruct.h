//
// Created by eagle on 2023/3/4.
//

#ifndef OPENSIM_RECONSTRUCT_H
#define OPENSIM_RECONSTRUCT_H

#include <itpp/itbase.h>
#include <itpp/signal/transforms.h>
#include <itpp/stat/misc_stat.h>
#include <utils/sim_parameter.h>
#include <utils/OtfFactory.h>

using namespace std;
using namespace itpp;

/**
 * Wiener Filtering frequency component
 * @param fiSMao: noisy frequency component
 * @param otf: system OTF
 * @param co: Wiener filter constant [=1, for minimum RMS estimate]
 * @param OBJPara: object power parameters
 * @param SFo: scaling factor (not significant here, so set to 1)
 * @param isCenter:  Filtering the central or off-center frequency component
 * @return Wiener Filtered estimate of FiSMao; avg. noise power in FiSMao
 */
cmat
wienerFilterCenter(const cmat &fiSMao, const OtfFactory &otfFactory, double co, vec OBJPara, double SFo, bool isCenter,
                   const mat &OBJSideP, Orientation &ori, int phaseIndex) {
    mat otf = otfFactory.otf;
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
    // NoisePower determination
    mat Zo = zeros(width, width);
    // frequency beyond which NoisePower estimate to be computed
    double NoiseFreq = otfFactory.cutOff + 20;
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
        OBJpower = OBJPara[0] * pow(Ro, OBJPara[1]);
    } else {
        OBJpower = OBJSideP;
    }
    OBJpower = pow(OBJpower, 2);

    // Wiener Filtering
    mat temp1 = SFo * otf / noisePower;
    mat temp2 = SFo * SFo * otfPower / noisePower + co / OBJpower;
    cmat FiSMaof = elem_div(elem_mult(fiSMao, to_cmat(temp1)), to_cmat(temp2));
    ori.noiseComp[phaseIndex] = noisePower;
    return FiSMaof;
}

/**
 * obtaining Wiener Filtered estimates of noisy frequency components
 * @param component: noisy estimates of separated frequency component
 * @param OBJPara: object power parameters
 * @param otf: system OTF
 * @return  Wiener Filtered estimates of components; avg. noise power; modulation factor
 */
Vec<cmat> wienerFilter(Vec<cmat> component, SIMParam &simParam, const OtfFactory &otfFactory, int index) {
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
    // Wiener Filtering central frequency component
    double SFo = 1;
    double co = 1.0;
    // object power parameters
    vec OBJPara = simParam.objPara;
    vec kA = simParam.orientations[index].freq;
    cmat fDof = wienerFilterCenter(component[0], otfFactory,
                                   co, OBJPara, SFo, true, Ro,
                                   simParam.orientations[index], 0);
    // modulation factor determination
    double Mm = simParam.orientations[index].modulationFactor;
//    estimateModulationFactor(component[1], kA, OBJPara, otfFactory);
//    simParam.orientations[index].modulationFactor = Mm;
    // Duplex power (default)
    complex<double> kv = kA[1] + 1i * kA[0]; // vector along illumination direction
    mat Rp = abs(Cv - kv);
    mat Rm = abs(Cv + kv);
    mat OBJp = OBJPara[0] * pow(Rp, OBJPara[1]);
    mat OBJm = OBJPara[0] * pow(Rm, OBJPara[1]);
    vec k3 = round(kA);
    OBJp(wo + k3(0), wo + k3(1)) = 0.25 * OBJp(wo + 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo + 1 + k3(1))
                                   + 0.25 * OBJp(wo - 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo - 1 + k3(1));
    OBJm(wo - k3(0), wo - k3(1)) = 0.25 * OBJm(wo + 1 - k3(0), wo - k3(1))
                                   + 0.25 * OBJm(wo - k3(0), wo + 1 - k3(1))
                                   + 0.25 * OBJm(wo - 1 - k3(0), wo - k3(1))
                                   + 0.25 * OBJm(wo - k3(0), wo - 1 - k3(1));
    // Filtering side lobes (off-center frequency components)
    SFo = Mm;
    cmat fDpf = wienerFilterCenter(component[1], otfFactory,
                                   co, OBJPara, SFo, false, OBJm,
                                   simParam.orientations[index], 1);
    cmat fDmf = wienerFilterCenter(component[2], otfFactory,
                                   co, OBJPara, SFo, false, OBJp,
                                   simParam.orientations[index], 2);
    // doubling Fourier domain size if necessary
    /* TODO */
    // Shifting the off-center frequency components to their correct location
    cmat shiftMat = exp(1i * 2 * pi * (kA(1) / width * (X - wo) + kA(0) / width * (Y - wo)));
    cmat fDp1 = fft2(elem_mult(ifft2(fDpf), shiftMat));
    cmat fDm1 = fft2(elem_mult(ifft2(fDmf), conj(shiftMat)));
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
    Angle[0] = sum(sum(elem_mult(elem_mult(fDof, conj(fDp1)), to_cmat(Zmask))));
    double Angle0 = angle(Angle)[0];

    // phase correction
    cmat fDp2 = exp(+1i * Angle0) * fDp1;
    cmat fDm2 = exp(-1i * Angle0) * fDm1;
    Vec<cmat> freqComps(3);
    freqComps[0] = fDof;
    freqComps[1] = fDp2;
    freqComps[2] = fDm2;
    return freqComps;
}

/**
 * To obtain signal spectrums corresponding to central and off-center frequency components
 * @param OBJPara: object power parameters
 * @param k2fa: illumination frequency vector
 * @param otf: system OTF
 * @param fDIp: one of the off-center frequency component (utilized here only for visual verification of computation)
 * @return signal spectrum corresponding to frequency component
 */
Vec<mat> tripletSNR0(vec OBJPara, vec k2fa, const mat &otf, const cmat &fDIp) {
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
    complex<double> kv = k2fa[1] + 1i * k2fa[0]; // vector along illumination direction
    mat Rp = abs(Cv - kv);
    mat Rm = abs(Cv + kv);
    mat OBJo = OBJPara[0] * pow(Ro, OBJPara[1]);
    mat OBJp = OBJPara[0] * pow(Rp, OBJPara[1]);
    mat OBJm = OBJPara[0] * pow(Rm, OBJPara[1]);
    vec k3 = round(k2fa);
    OBJo(wo, wo) = 0.25 * OBJo(wo + 1, wo) + 0.25 * OBJo(wo, wo + 1)
                   + 0.25 * OBJo(wo - 1, wo) + 0.25 * OBJo(wo, wo - 1);
    OBJp(wo + k3(0), wo + k3(1)) = 0.25 * OBJp(wo + 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo + 1 + k3(1))
                                   + 0.25 * OBJp(wo - 1 + k3(0), wo + k3(1))
                                   + 0.25 * OBJp(wo + k3(0), wo - 1 + k3(1));
    OBJm(wo - k3(0), wo - k3(1)) = 0.25 * OBJm(wo + 1 - k3(0), wo - k3(1))
                                   + 0.25 * OBJm(wo - k3(0), wo + 1 - k3(1))
                                   + 0.25 * OBJm(wo - 1 - k3(0), wo - k3(1))
                                   + 0.25 * OBJm(wo - k3(0), wo - 1 - k3(1));
    // signal spectrum
    mat SIGao = elem_mult(OBJo, otf);
    mat SIGap = elem_mult(OBJp, otf);
    mat SIGam = elem_mult(OBJm, otf);
    SIGap = circShift(SIGap, -k3);
    SIGam = circShift(SIGam, k3);
//    SIGap.
    Vec<mat> result(3);
    result(0) = SIGao;
    result(1) = SIGap;
    result(2) = SIGam;
    return result;
}

#endif //OPENSIM_RECONSTRUCT_H
