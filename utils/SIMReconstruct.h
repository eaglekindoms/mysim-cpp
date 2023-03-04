#ifndef ESTIMATE_PARAM_H
#define ESTIMATE_PARAM_H

#include <itpp/itbase.h>
#include <utils/OtfFactory.h>
#include <utils/sim_parameter.h>

using namespace std;
using namespace itpp;

class SIMReconstruct {
public:
    /**
     * estimate sim param
     * @param patterns: raw sim images
     * @param OtfFactory: system otf
     * @return sim param
     */
    virtual SIMParam
    estimateParameters(const Vec<mat> &patterns, const OtfFactory &otfFactory) = 0;

    /**
     * separated frequency components
     * @param patterns: raw sim images
     * @param simParam: sim param
     * @param otfFactory: system otf
     * @return noisy estimates of separated frequency components
     */
    virtual Vec<cmat>
    separatedSIMComponents(const Vec<mat> &patterns, SIMParam &simParam, const OtfFactory &otfFactory) = 0;

    /**
     * reconstruct sim images
     * @param freqComp: nine frequency components
     * @param simParam: sim parameters
     * @param OtfFactory: system OTF
     * @return all nine frequency components merged into one
     */
    virtual mat reconstruct(const Vec<cmat> &freqComp, const SIMParam &simParam, const OtfFactory &otfFactory) = 0;
};

#endif // ESTIMATE_PARAM_H
