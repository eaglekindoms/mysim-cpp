//
// Created by eagle on 2023/3/21.
//

#ifndef OTFPROVIDER_H
#define OTFPROVIDER_H

#include <itpp/itbase.h>
#include <iostream>

using namespace itpp;

class OtfProvider {
public:
    double cyclesPerMicron;
    double na, lambda, cutOff;
    int samplesLateral = 512;
    double vecCyclesPerMicron = -1;
    double attStrength = .99, attFWHM = 1.2;
    bool useAttenuation;

    cmat vals;
    cmat valsAtt;
    mat valsOnlyAtt;

    // was the OTF set from estimate?
    // Are there different bands? If so, how many?
    bool isEstimate, isMultiBand;
    int maxBand = 1;
    double estimateAValue = .3;

    /**
    * For [0..cutoff] normalized to [0..1], return the ideal OTF. OTF of an ideal,
    * i.e. aberration-free lens system, resolution-limited by a circular pupil in
    * the Fourier place,
    * "Joseph W. Goodman, Introduction to Fourier Optics, 3. edition, page 145".
    *
    * @param dist Distance to cutoff, in normalized range 0..1. Values outsize of 0..1 return 0.
    */
    static double valIdealOTF(double dist) {
        if ((dist < 0) || (dist > 1))
            return 0;
        return (2 / pi) * (acos(dist) - dist * sqrt(1 - dist * dist));
    }

    /**
     * Returns the attenuation value at dist.
     *
     * @param dist Distance to center in cycles/micron
     * @param str  Strength of the attenuation
     * @param fwhm FWHM of Attenuation, in cycles/micron
     */
    static double valAttenuation(const double dist, const double str, const double fwhm) {
        return (1 - str * exp(-(pow(dist, 2)) / (2 * pow(fwhm / 2.355, 2))));
    }

    /**
     * Create a new OTF from a (very basic) estimate.
     * The curvature factor account for deviation of
     * real-world OTFs from the theoretical optimum.
     * a=1 yields an ideal OTF, a=0.2 .. 0.4 are more realistic,
     * a has no effect on cutoff.
     *
     * @param na     Objectives NA
     * @param lambda Emission wavelength (nm)
     * @param a      curvature factor, a = [0..1]
     */
    static OtfProvider fromEstimate(double na, double lambda, double a) {
        if ((a < 0) || (a > 1) || (na < 0.3) || (na > 2.2) || (lambda < 300) || (lambda > 1500))
            throw "unphysical input parameters";
        OtfProvider ret;
        ret.na = na;
        ret.lambda = lambda;
        ret.cutOff = 1000 / (lambda / na / 2);
        ret.cyclesPerMicron = ret.cutOff / ret.samplesLateral;
        ret.vals = cmat(1, ret.samplesLateral);
        ret.valsAtt = cmat(1, ret.samplesLateral);
        ret.valsOnlyAtt = mat(1, ret.samplesLateral);

        ret.isMultiBand = false;
        ret.isEstimate = true;
        ret.estimateAValue = a;

        // sample some values up to cutoff
        for (int i = 0; i < ret.samplesLateral; ++i) {
            // v: normalize [0..cutoff] -> [0..1]
            double v = i / (double) ret.samplesLateral;
            // get OTF at v, multiply empirical correction for curvature
            double r = valIdealOTF(v) * pow(a, v);
            ret.vals(0, i) = std::complex<double>(r, 0);
        }
        // initialize attenuation cache
        ret.setAttenuation(ret.attStrength, ret.attFWHM);
        ret.useAttenuation = false;
        return ret;
    }

    /**
     * Sets the OTF attenuation parameters.
     * Attenuation is also swichted on by this function.
     *
     * @param strength Strength of attenuation, 0..1, usually 0.9 .. 0.99
     * @param fwhm     FWHM of the attenuation, in cycles / micron
     */
    void setAttenuation(double strength, double fwhm) {

        this->attStrength = strength;
        this->attFWHM = fwhm;

        // update cached attenuation values
        for (int b = 0; b < vals.rows(); b++)
            for (int v = 0; v < vals.get_row(b).length(); v++) {
                double dist = v * cyclesPerMicron;
                valsOnlyAtt(b, v) = valAttenuation(dist, attStrength, attFWHM);
                valsAtt(b, v) = vals(b, v) * valsOnlyAtt(b, v);
            }
    }

    /**
     * Multiplies / outputs OTF to a vector. Quite general function,
     * some wrappers are provided for conveniece.
     *
     * @param vec    Vector to write / multiply to
     * @param band   OTF band
     * @param kx     OTF center position offset x
     * @param ky     OTF center position offset y
     * @param useAtt if to use attenuation (independent of how {@link #switchAttenuation} is set)
     * @param write  if set, vector is overridden instead of multiplied
     */
    void otfToVector(cmat &vec, const int band,
                     const double kx, const double ky,
                     const bool useAtt, const bool write) {

        // parameters
        if (vecCyclesPerMicron <= 0)
            throw "Vector pixel size not initialized";
        int w = vec.cols(), h = vec.rows();

        // loop output vector
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; x++) {
                // wrap to coordinates: x in [-w/2,w/2], y in [-h/2, h/2]
                double xh = (x < w / 2) ? (x) : (x - w);
                double yh = (y < h / 2) ? (-y) : (h - y);
                // from these, calculate distance to kx,ky, convert to cycl/microns
                double rad = sqrt((xh - kx) * (xh - kx) + (yh - ky) * (yh - ky));
                double cycl = rad * vecCyclesPerMicron;
                // over cutoff? just set zero
                if (cycl > cutOff) {
                    vec.set(x, y, std::complex<double>(0, 0));
                }
                // within cutoff?
                if (cycl <= cutOff) {
                    // get the OTF value
                    std::complex<double> val = getOtfVal(band, cycl, useAtt);
                    // multiply to vector or write to vector
                    if (!write) {
                        vec.set(x, y, vec.get(x, y) * (conj(val)));
                    } else {
                        vec.set(x, y, val);
                    }
                }
            }
        }
    }

    /**
     * Get the OTF value at 'cycl'.
     *
     * @param band OTF band
     * @param cycl Position in cycles/micron
     * @param att-> If true, return attenuated value (see {@link #setAttenuation})
     */
    std::complex<double> getOtfVal(int band, double cycl, bool att) {
        // checks
        if (!this->isMultiBand)
            band = 0;
        if ((band >= maxBand) || (band < 0))
            throw "band idx too high or <0";
        if (cycl < 0)
            throw "cylc negative!";

        double pos = cycl / cyclesPerMicron;
        // for now, linear interpolation, could be better with a nice cspline
        int lPos = (int) floor(pos);
        float f = (float) (pos - lPos);
        // out of support, return 0
        if (cycl >= cutOff || ceil(pos) >= samplesLateral) {
            return std::complex<double>(0, 0);
        } else if (att) {
            std::complex<double> retl = valsAtt(band, lPos) * (1 - f);
            std::complex<double> reth = valsAtt(band, lPos) * (f);
            return retl + reth;
        } else {
            std::complex<double> retl = vals(band, lPos) * (1 - f);
            std::complex<double> reth = vals(band, lPos) * (f);
            return retl + reth;
        }

    }

    /**
        * Sets pixel size, for output to vectors
        *
        * @param cyclesPerMicron Pixel size of output vector, in cycles/micron
        */
    void setPixelSize(double cyclesPerMicron) {
        if (cyclesPerMicron <= 0)
            throw "pxl size must be positive";
        vecCyclesPerMicron = cyclesPerMicron;
    }

private:
    OtfProvider() {}
};

#endif //OTFPROVIDER_H
