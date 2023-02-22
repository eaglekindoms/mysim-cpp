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
