/*!
 * \file
 * \brief Newton search test program
 * \author Tony Ottosson
 *
 * -------------------------------------------------------------------------
 *
 * Copyright (C) 1995-2010  (see AUTHORS file for a list of contributors)
 *
 * This file is part of IT++ - a C++ library of mathematical, signal
 * processing, speech processing, and communications classes and functions.
 *
 * IT++ is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any
 * later version.
 *
 * IT++ is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along
 * with IT++.  If not, see <http://www.gnu.org/licenses/>.
 *
 * -------------------------------------------------------------------------
 */

#include <iostream>
#include <utils/optimizer.h>
#include <utils/neldermead.h>
#include <itpp/itoptim.h>

using namespace std;
using namespace itpp;

double rosenbrock(const std::array<double, 2> &x) {
//    double f1 = x[1] - sqr(x[0]), f2 = 1 - x[0];
//    double F = 50 * sqr(f1) + 0.5 * sqr(f2) + 0.0;
    double F = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (3 - x[0]) * (3 - x[0]);
    return F;
}

double rosen(const vec &x) {
//    double f1 = x[1] - sqr(x[0]), f2 = 1 - x[0];
//    double F = 50 * sqr(f1) + 0.5 * sqr(f2) + 0.0;
    double F = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (3 - x[0]) * (3 - x[0]);
    return F;
}

double rosenbrock1(const std::vector<double> &x) {
//    double f1 = x[1] - sqr(x[0]), f2 = 1 - x[0];
//    double F = 50 * sqr(f1) + 0.5 * sqr(f2) + 0.0;
    double F = 100.0 * (x[1] - x[0] * x[0]) * (x[1] - x[0] * x[0]) + (3 - x[0]) * (3 - x[0]);
    return F;
}
// gradient 是func的导数函数


int main(void) {

    cout << "=====================================" << endl;
    cout << "    Test of Numerical optimization " << endl;
    cout << "=====================================" << endl;
    std::array<double, 2> start = {0.5, 0.5};
    std::array<double, 2> step = {0.1, 0.1};

    nelder_mead_result<double, 2> result = nelder_mead<double, 2>(
            rosenbrock,
            start,
            1.0e-25, // the terminating limit for the variance of function values
            step
    );
    std::cout << "Found minimum: " << std::fixed << result.xmin[0] << ' ' << result.xmin[1] << std::endl;
    std::vector<double> start1;
    start1.push_back(0.5);
    start1.push_back(0.5);
//    NelderMeadSimplex simplex(rosenbrock1, start1);
//    std::cout << simplex.bestVertex().coordinate_[0] <<", "<< simplex.bestVertex().coordinate_[0] << std::endl;
//    std::cout << simplex.bestVertex().value_;
    RealFunctionvalueAtCoordinate bestVertex = nelderMead(rosenbrock1, start1);
    std::cout << bestVertex.coordinate_[0] << ", " << bestVertex.coordinate_[1] << std::endl;
    std::cout << bestVertex.value_;
//    cout << "step:" << result.icount << endl;
//    Optimset optimset;
//    optimset.tolX = 0.0001;
//    optimset.tolF = 0.0001;
//    optimset.max_iter = 1000;
//    optimset.max_eval = 1000;
//    optimset.verbose = 0;
//    vec s1 = "0.0, 0.0";
//    auto x = nelder_mead(2, s1, rosen, optimset);
//    printf("argmin found at point: ");
//    cout << "result x: " << x.x << ", fx: " << x.fx << endl;
    return 0;
}
//    Optimset optimset;
//    optimset.tolX = 0.0001;
//    optimset.tolF = 0.0001;
//    optimset.max_iter = 500;
//    optimset.max_eval = 500;
//    optimset.verbose = 0;
//    auto phaseKai2opt1 = [&fS1aTnoisy, &otf](const vec &freq) -> double {
//        return phaseAutoCorrelationFreqByOpt(freq, fS1aTnoisy, otf, true);
//    };
//    auto x = nelder_mead(2, reinterpret_cast<vec &>(freqVector[0]), phaseKai2opt1, optimset);
//    printf("argmin found at point: ");
//    cout << "result x: " << x.x << ", fx: " << x.fx << endl;