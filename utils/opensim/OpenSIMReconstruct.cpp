#include <utils/SIMReconstruct.h>
#include "estimate.h"
#include "reconstruct.h"

class OpenSIMReconstruct : public SIMReconstruct {
public:

    SIMParam
    estimateParameters(const Vec<mat> &patterns, const OtfFactory &otfFactory) {
        SIMParam simParam;
        ThreadPool pool(3);
        vector<future<Orientation>> poolResult;
        for (int i = 0; i < 3; ++i) {
            poolResult.emplace_back(pool.enqueue([=] {
                cout << "estimates frequency components, index: " << i << endl;
                return estimateSIMParameters(patterns, otfFactory, i * 3);
            }));
        }
        for (int i = 0; i < 3; ++i) {
            simParam.orientations[i] = poolResult[i].get();
        }
        return simParam;
    }

    Vec<cmat>
    separatedSIMComponents(const Vec<mat> &patterns, SIMParam &simParam, const OtfFactory &otfFactory) {
        Vec<Vec<cmat>> components(3);
        for (int i = 0; i < 3; ++i) {
            components[i] = separatedSIMComponents2D(patterns, simParam.orientations[i], otfFactory, i * 3);
        }
        // averaging the central frequency components
        cmat fCent = (components[0][0] + components[1][0] + components[2][0]) / 3;
        // Object power parameters determination
        simParam.objPara = estimateObjectPowerParameters(fCent, otfFactory);//"273624.7852070, -1.039610";
        // Wiener Filtering the noisy frequency components
//        Vec<mat> filterComps(9);
        Vec<cmat> freqComp(9);
        for (int i = 0; i < 3; ++i) {
            simParam.orientations[i].modulationFactor =
                    estimateModulationFactor(components[i][1],
                                             simParam.orientations[i].freq,
                                             simParam.objPara, otfFactory);
            Vec<cmat> fComp = wienerFilter(components[i], simParam, otfFactory, i);
            for (int j = 0; j < 3; ++j) {
                freqComp[i * 3 + j] = fComp[j];
//                filterComps[i * 3 + j] = real(fComp[j]);
            }
        }
//        showPatternImage("filtered sim images", filterComps, obj.rows(), 0);
        return freqComp;
    }

    mat reconstruct(const Vec<cmat> &freqComp, const SIMParam &simParam, const OtfFactory &otfFactory) {
        Vec<cmat> results = mergeSIMImages(freqComp, simParam, otfFactory.otf);
        Vec<mat> reconstructImages(6);
        for (int i = 0; i < 3; ++i) {
            reconstructImages[i] = real(ifft2(fftshift(results[i])));
            double rMax = max(max(reconstructImages[i], 1));
            reconstructImages[i] = reconstructImages[i] / rMax;
            reconstructImages[i + 3] = abs(results[i]);
        }
        showPatternImage("reconstruction sim images", reconstructImages, reconstructImages[0].rows(), 0);
        // show raw image
        return reconstructImages[0];
    }

private:

    /**
     * obtaining the noisy estimates of three frequency components
     * @param patterns: raw SIM images
     * @param simParam: sim param needed to be updated
     * @param otf: system OTF
     * @param index: phase index
     * @return noisy estimates of separated frequency components, and Orientation param
     */
    Vec<cmat>
    separatedSIMComponents2D(Vec<mat> patterns, const Orientation &ori, const OtfFactory &otfFactory, int index) {
        mat otf = otfFactory.otf;
        // computing PSFe for edge tapering SIM images
        mat psfd = pow(otf, 3);
        psfd = fftshift(psfd);
        psfd = real(ifft2(to_cmat(psfd)));
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
        double MF = 1.0;
        // Transformation Matrix
        MatrixXcd M(3, 3);
        for (int k = 0; k < 3; ++k) {
            M(k, 0) = 1.0;
            M(k, 1) = 0.5 * MF * exp(-1i * ori.phaseShift[k]);
            M(k, 2) = 0.5 * MF * exp(+1i * ori.phaseShift[k]);
        }
        M = 0.5 * M;
        // Separate the components
        cout << "Separate the components" << endl;
        MatrixXcd Minv = M.inverse();
        cout << "Separate Matrix = " << Minv << endl;
        Vec<cmat> unmixedFT(3); //  unmixed frequency components of raw SIM images
        for (int i = 0; i < 3; ++i) {
            unmixedFT[i] = Minv(i, 0) * ftNoisyImages[0]
                           + Minv(i, 1) * ftNoisyImages[1]
                           + Minv(i, 2) * ftNoisyImages[2];
        }
        return unmixedFT;
    }

    /**
     * To merge all 9 frequency components into one using generalized Wiener Filter
     * @param freqComp: nine frequency components
     * @param noiseComp: noise powers corresponding to nine frequency components
     * @param modFactors: modulation factors for the three illumination orientations
     * @param freqVectors: illumination frequency vectors for the three illumination orientations
     * @param otf: system OTF
     * @return Fsum: all nine frequency components merged into one using generalised Wiener Filter;\n
     * Fperi: six off-center frequency components merged into one using generalised Wiener Filter;\n
     * Fcent: averaged of the three central frequency components
     */
    Vec<cmat> mergeSIMImages(Vec<cmat> freqComp, SIMParam simParam, const mat &otf) {
        Vec<mat> sigComp(9);
        vec noiseComp(9);
        vec OBJPara = simParam.objPara;
        for (int i = 0; i < 3; ++i) {
            sigComp.set_subvector(i * 3,
                                  tripletSNR0(OBJPara, simParam.orientations[i].freq, otf, freqComp[i * 3 + 2]));
        }
        for (int i = 0; i < 3; ++i) {
            sigComp[i * 3 + 1] = simParam.orientations[i].modulationFactor * sigComp[i * 3 + 1];
            sigComp[i * 3 + 2] = simParam.orientations[i].modulationFactor * sigComp[i * 3 + 2];
            for (int j = 0; j < 3; ++j) {
                noiseComp[i * 3 + j] = simParam.orientations[i].noiseComp[j];
            }
        }
        // Generalized Wiener-Filter computation
        Vec<mat> snrComps(9);
        mat ComDeno(snrComps[0].rows(), snrComps[0].cols());
        mat ComPeri(snrComps[0].rows(), snrComps[0].cols());
        // all nine frequency components merged into one using generalised Wiener Filter
        cmat FSum(snrComps[0].rows(), snrComps[0].cols());
        // six off-center frequency components merged into one using generalised Wiener Filter
        cmat Fperi(snrComps[0].rows(), snrComps[0].cols());
        for (int i = 0; i < 9; ++i) {
            snrComps[i] = pow(sigComp[i], 2) / noiseComp[i];
            ComDeno += snrComps[i];
            FSum += elem_mult(freqComp[i], to_cmat(snrComps[i]));
            // 非低频区域累加
            if (i != 0 || i != 3 || i != 6) {
                ComPeri += snrComps[i];
                Fperi += elem_mult(freqComp[i], to_cmat(snrComps[i]));
            }
        }
        ComDeno = 0.01 + ComDeno;
        ComPeri = 0.01 + ComPeri;
        FSum = elem_div(FSum, to_cmat(ComDeno));
        Fperi = elem_div(Fperi, to_cmat(ComPeri));
        // averaged central frequency component
        cmat FCent = (freqComp[0] + freqComp[3] + freqComp[6]) / 3;
        Vec<cmat> result(3);
        result[0] = FSum;
        result[1] = Fperi;
        result[2] = FCent;
        return result;
    }
};