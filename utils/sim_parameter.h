//
// Created by eagle on 2023/2/21.
//

#ifndef SIM_PARAMETER_H
#define SIM_PARAMETER_H

#include <itpp/itbase.h>
#include <vector>
#include <iostream>
#include <utils/itpp_mat_utils.h>

using namespace std;


class Orientation {
public:
    int orientationIndex;
    vec freq = zeros(2);
    double phaseShift[3] = {0};
    double modulationFactor = 0;
    double noiseComp[3] = {0};

    Orientation() {}

    Orientation(int index) {
        orientationIndex = index;
        cout << "init each orientation's param" << endl;
    }

    void setNoiseComp(double n1, double n2, double n3) {
        noiseComp[0] = n1;
        noiseComp[1] = n2;
        noiseComp[2] = n3;
    }

    void setPhaseShift(double p1, double p2, double p3) {
        phaseShift[0] = p1;
        phaseShift[1] = p2;
        phaseShift[2] = p3;
    }
};


class SIMParam {
public:
    vector<Orientation> orientations;

    SIMParam() {
        orientations.resize(3, Orientation(0));
        for (int i = 0; i < 3; ++i) {
            orientations[i].orientationIndex = i;
        }
    }

    void matlabParam() {
        orientations[0].freq = "-0.013343 75.217859";
        orientations[1].freq = "65.150523 37.615985";
        orientations[2].freq = "-65.134815 37.613092";
        orientations[0].setPhaseShift(-2.5559, 121.4715, -122.1179);
        orientations[1].setPhaseShift(3.2632, 115.4465, -118.9854);
        orientations[2].setPhaseShift(-0.7789, -122.2549, 115.8610);
        orientations[0].modulationFactor = 0.7323;
        orientations[1].modulationFactor = 0.6685;
        orientations[2].modulationFactor = 0.6329;
        orientations[0].setNoiseComp(2.6689496e+06, 1.0689196e+07, 1.0689196e+07);
        orientations[1].setNoiseComp(2.6433418e+06, 1.0528055e+07, 1.0528055e+07);
        orientations[2].setNoiseComp(2.6051157e+06, 1.0437245e+07, 1.0437245e+07);
    }
};


std::ostream &operator<<(std::ostream &os, const Orientation &ori) {
    os << '[' << "orientationIndex = " << ori.orientationIndex;
    os << ", freqX = " << ori.freq[0];
    os << ", freqY = " << ori.freq[1];
    os << ", phaseShift = " << ori.phaseShift[0] << ", " << ori.phaseShift[1] << ", " << ori.phaseShift[2];
    os << ", modulationFactor = " << ori.modulationFactor << ']';
    return os;
}

#endif //SIM_PARAMETER_H
