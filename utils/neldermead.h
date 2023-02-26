//
// Created by eagle on 2023/2/24.
//

#ifndef NELDERMEAD_H
#define NELDERMEAD_H

#include <functional>
#include <list>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <chrono>


/**
 * 获取操作系统当前时间点，精确到ms
 * @return
 */
long long get_cur_time() {
    // 获取操作系统当前时间点（精确到微秒）
    std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tpMicro
            = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now());
    // (微秒精度的)时间点 => (微秒精度的)时间戳
    time_t totalMicroSeconds = tpMicro.time_since_epoch().count();

    long long currentTime = ((long long) totalMicroSeconds) / 1000;

    return currentTime;
}

/*! \internal
 *  \brief The parameters for a Nelder-Mead optimisation.
 */
struct NelderMeadParameters {
    //! Factor to evaluate the reflection point
    double alpha_ = 1;
    //! Factor to evaluate the expansion point
    double gamma_ = 2;
    //! Factor to evaluate the contraction point
    double rho_ = 0.5;
    //! Factor to evaluate the simplex shrinkage
    double sigma_ = 0.5;
};

constexpr NelderMeadParameters defaultNelderMeadParameters = {1, 2, 0.5, 0.5};

/*! \internal
 * \brief Tie together coordinate and function value at this coordinate.
 */
struct RealFunctionvalueAtCoordinate {
    //! Vertex coordinate
    std::vector<double> coordinate_;
    //! Function value at this coordinate
    double value_;
};

/*! \brief Evaluate the linear combination of two vectors a and b.
 *
 * \param[in] alpha scaling factor for a
 * \param[in] a vector to be scaled
 * \param[in] beta scaling factor for b
 * \param[in] b vector to be scaled
 *
 * \returns alpha * a + beta * b.
 */
std::vector<double> linearCombination(double alpha, std::vector<double> a, double beta, std::vector<double> b) {
    if (a.size() != b.size()) {
        std::cout << "Input vectors have to have the same size to evaluate their linear combination." << std::endl;
        exit(-1);
    }
    std::vector<double> result(a.size());
    std::transform(
            std::begin(a), std::end(a), std::begin(b), std::begin(result), [alpha, beta](auto a, auto b) {
                return alpha * a + beta * b;
            });
    return result;
}

/*! \internal
 * \brief The simplex for the Nelder-Mead algorithm.
 *
 * Contains the N+1 simplex N-dimensional coordinates and its function values.
 * Allows for simplex manipulations as needed for the Nelder-Mead algorithm.
 *
 * \note Keeps the simplex sorted according to function values with the simplex
 *       at the lowest function value first.
 */
class NelderMeadSimplex {
public:

    int reflectStep = 0;
    int expanseStep = 0;
    int contractStep = 0;
    int shrinkStep = 0;

    /*! \brief Set up Nelder-Mead simplex from an initial guess.
     *
     * \note Triggers N+1 function evaluations at all simplex points.
     *
     * \param[in] f the function to be evaluated
     * \param[in] initalGuess initial guess of the coordinates.
     *
     */
    NelderMeadSimplex(const std::function<double(const std::vector<double> &)> &fn,
                      std::vector<double> start) {
        // initial simplex contains the initally guessed vertex
        std::vector<double> initalVertex;
        initalVertex.insert(initalVertex.end(), start.begin(), start.end());
        simplex_.push_back({initalVertex, fn(initalVertex)});
        // create the missing verticies by moving 0.05 or 0.0025 if null
        // from the initial vertex dimension after dimension
        for (auto &v: initalVertex) {
            const auto oldValue = v;
            if (v == 0) {
                v = 0.0025;
            } else {
                v += 0.05;
            }
            simplex_.push_back({initalVertex, fn(initalVertex)});
            v = oldValue;
        }
        simplex_.sort([](const RealFunctionvalueAtCoordinate &lhs,
                         const RealFunctionvalueAtCoordinate &rhs) { return lhs.value_ < rhs.value_; });
        updateCentroidAndReflectionPoint();
    }

    //! Return the vertex with the lowest function value at any of the simplex vertices.
    const RealFunctionvalueAtCoordinate &bestVertex() const {
        return simplex_.front();
    }

    //! Return the vertex of the simplex with the highest (worst) function value.
    const RealFunctionvalueAtCoordinate &worstVertex() const {
        return simplex_.back();
    }

    //! Return the second largest function value at any of the simplex vertices.
    double secondWorstValue() const {
        // go backwards one step from the end of the sorted simplex list
        // and look at the vertex value
        return std::next(std::rbegin(simplex_))->value_;
    }

    //! Return the reflection point and the evaluated function value at this point.
    RealFunctionvalueAtCoordinate
    evaluateReflectionPoint(const std::function<double(const std::vector<double> &)> &fn) const {
        return {reflectionPointCoordinates_, fn(reflectionPointCoordinates_)};
    }

    //! Evaluate and return the expansion point and function value.
    RealFunctionvalueAtCoordinate
    evaluateExpansionPoint(const std::function<double(const std::vector<double> &)> &fn) const {
        const std::vector<double> expansionPointCoordinate =
                linearCombination(1 - defaultNelderMeadParameters.gamma_,
                                  centroidWithoutWorstPoint_,
                                  defaultNelderMeadParameters.gamma_,
                                  reflectionPointCoordinates_);
        return {expansionPointCoordinate, fn(expansionPointCoordinate)};
    }

    //! Evaluate and return the contraction point and function value.
    RealFunctionvalueAtCoordinate
    evaluateContractionPoint(const std::function<double(const std::vector<double> &)> &fn) const {
        std::vector<double> contractionPoint = linearCombination(1 - defaultNelderMeadParameters.rho_,
                                                                 centroidWithoutWorstPoint_,
                                                                 defaultNelderMeadParameters.rho_,
                                                                 worstVertex().coordinate_);
        return {contractionPoint, fn(contractionPoint)};
    }

    /*! \brief Replace the simplex vertex with the largest function value.
     *
     * \param[in] newVertex to replace the worst vertex with
     * \note keeps the simplex list sorted and reevaluates the reflection point
     */
    void swapOutWorst(const RealFunctionvalueAtCoordinate &newVertex) {
        // drop the worst point - we know it's at the back of the simplex list, because
        // we kept the list sorted
        simplex_.pop_back();
        // find the point to insert the new vertex, so that the simplex vertices
        // keep being sorted according to function value
        const auto insertionPoint = std::lower_bound(
                std::begin(simplex_),
                std::end(simplex_),
                newVertex.value_,
                [](const RealFunctionvalueAtCoordinate &lhs, double value) { return lhs.value_ < value; });
        simplex_.insert(insertionPoint, newVertex);
        // now that the simplex has changed, it has a new centroid and reflection point
        updateCentroidAndReflectionPoint();
    }

    /*! \brief Shrink the simplex.
     *
     * All points move closer to the best point by a factor \f$\sigma\f$.
     *
     * Replace all point coordinates, except the best, with
     * \f$x_i = x_{\mathrm{best}} + \sigma (x_i - x_{\mathrm{best}})\f$
     */
    void shrinkSimplexPointsExceptBest(const std::function<double(const std::vector<double> &)> &fn) {
        std::vector<double> bestPointCoordinate = simplex_.front().coordinate_;
        // skipping over the first simplex vertex, pull points closer to the best
        // vertex
        std::transform(std::next(std::begin(simplex_)),
                       std::end(simplex_),
                       std::next(std::begin(simplex_)),
                       [bestPointCoordinate, fn](
                               const RealFunctionvalueAtCoordinate &d) -> RealFunctionvalueAtCoordinate {
                           const std::vector<double> shrinkPoint =
                                   linearCombination(defaultNelderMeadParameters.sigma_,
                                                     d.coordinate_,
                                                     1 - defaultNelderMeadParameters.sigma_,
                                                     bestPointCoordinate);
                           return {shrinkPoint, fn(shrinkPoint)};
                       });

        simplex_.sort([](const RealFunctionvalueAtCoordinate &lhs,
                         const RealFunctionvalueAtCoordinate &rhs) { return lhs.value_ < rhs.value_; });

        // now that the simplex has changed, it has a new centroid and reflection point
        updateCentroidAndReflectionPoint();
    }

    /*! \brief The oriented length of the vertex.
     *
     * The oriented length of the simplex is defined as the largest distance
     * between the first simplex vertex coordinate (with the lowest, best function
     * value) and any other simplex coordinate.
     *
     * The oriented length is used as a computationally fast and simple
     * convergence criterion because it is proven that
     * orientedLegnth < simplex_diameter < 2 * orientedLength
     *
     */
    double orientedLength() const {
        double result = 0;
        const std::vector<double> firstSimplexVertexCoordinate = simplex_.front().coordinate_;
        // find out which vertex coordinate has the largest distance to the first simplex vertex.
        for (const auto &simplexVertex: simplex_) {
            const std::vector<double> differenceVector =
                    linearCombination(1, firstSimplexVertexCoordinate, -1, simplexVertex.coordinate_);
            const double thisLength = std::accumulate(
                    std::begin(differenceVector), std::end(differenceVector), 0., [](double sum, double value) {
                        return sum + value * value;
                    });
            result = std::max(result, thisLength);
        }
        return sqrt(result);
    }

private:
    /*! \brief Update centroid and reflection point.
     *
     * The arithmetic mean of all vertex coordinates expect the one at the
     * highest (worst) function value.
     *
     */
    void updateCentroidAndReflectionPoint() {
        // intialize with first vertex, then add up all other vertex coordinates
        // expect last one
        centroidWithoutWorstPoint_ =
                std::accumulate(std::next(std::begin(simplex_)),
                                std::prev(std::end(simplex_)),
                                simplex_.front().coordinate_,
                                [](std::vector<double> sum, const RealFunctionvalueAtCoordinate &x) {
                                    std::transform(std::begin(sum),
                                                   std::end(sum),
                                                   std::begin(x.coordinate_),
                                                   std::begin(sum),
                                                   std::plus<>());
                                    return sum;
                                });

        // divide the summed up coordinates by N (the simplex has N+1 vertices)
        std::transform(std::begin(centroidWithoutWorstPoint_),
                       std::end(centroidWithoutWorstPoint_),
                       std::begin(centroidWithoutWorstPoint_),
                       [n = simplex_.size() - 1](const auto &x) { return x / n; });

        // now, that we have evaluated the centroid, update the reflection points
        reflectionPointCoordinates_ = linearCombination(
                defaultNelderMeadParameters.alpha_ + 1, centroidWithoutWorstPoint_, -1, worstVertex().coordinate_);
    }

    /*! \brief The points of the simplex with the function values.
     * \note This list stays sorted according to function value during the
     *       life-time of this object.
     */
    std::list<RealFunctionvalueAtCoordinate> simplex_;

    //! The centroid of the simplex, skipping the worst point is updated once the simplex changes
    std::vector<double> centroidWithoutWorstPoint_;

    //! The reflection point and its function value is updated once the simplex changes
    std::vector<double> reflectionPointCoordinates_;
};

RealFunctionvalueAtCoordinate nelderMead(const std::function<double(const std::vector<double> &)> &functionToMinimize,
                                         std::vector<double> initalGuess,
                                         double minimumRelativeSimplexLength = 1e-4,
                                         int maxSteps = 500) {
    // 计时器
    long long t1 = get_cur_time();
    // Set up the initial simplex, sorting vertices according to function value
    NelderMeadSimplex nelderMeadSimplex(functionToMinimize, initalGuess);
    // Run until maximum step size reached or algorithm is converged, e.g.,
    // the oriented simplex length is smaller or equal a given number.
    const double minimumSimplexLength = minimumRelativeSimplexLength * nelderMeadSimplex.orientedLength();
    int currentStep = 0;
    for (; nelderMeadSimplex.orientedLength() > minimumSimplexLength && currentStep < maxSteps;
           ++currentStep) {

        // see if simplex can by improved by reflecing the worst vertex at the centroid
        const RealFunctionvalueAtCoordinate &reflectionPoint =
                nelderMeadSimplex.evaluateReflectionPoint(functionToMinimize);
        nelderMeadSimplex.reflectStep++;

        // Reflection point is not better than best simplex vertex so far
        // but better than second worst
        if ((nelderMeadSimplex.bestVertex().value_ <= reflectionPoint.value_)
            && (reflectionPoint.value_ < nelderMeadSimplex.secondWorstValue())) {
            nelderMeadSimplex.swapOutWorst(reflectionPoint);
            continue;
        }

        // If the reflection point is better than the best one see if simplex
        // can be further improved by continuing going in that direction
        if (reflectionPoint.value_ < nelderMeadSimplex.bestVertex().value_) {
            RealFunctionvalueAtCoordinate expansionPoint =
                    nelderMeadSimplex.evaluateExpansionPoint(functionToMinimize);
            nelderMeadSimplex.expanseStep++;
            if (expansionPoint.value_ < reflectionPoint.value_) {
                nelderMeadSimplex.swapOutWorst(expansionPoint);
            } else {
                nelderMeadSimplex.swapOutWorst(reflectionPoint);
            }
            continue;
        }

        // The reflection point was a poor choice, try contracting the
        // worst point coordinates using the centroid instead
        RealFunctionvalueAtCoordinate contractionPoint =
                nelderMeadSimplex.evaluateContractionPoint(functionToMinimize);
        nelderMeadSimplex.contractStep++;
        if (contractionPoint.value_ < nelderMeadSimplex.worstVertex().value_) {
            nelderMeadSimplex.swapOutWorst(contractionPoint);
            continue;
        }

        // If neither expansion nor contraction of the worst point give a
        // good result shrink the whole simplex
        nelderMeadSimplex.shrinkSimplexPointsExceptBest(functionToMinimize);
        nelderMeadSimplex.shrinkStep++;
    }
    long long t2 = get_cur_time();
    std::cout << "nelder-mead use: " << t2 - t1 << "ms" << ", totalStep: " << currentStep;
    std::cout << ", reflectStep: " << nelderMeadSimplex.reflectStep;
    std::cout << ", expanseStep: " << nelderMeadSimplex.expanseStep;
    std::cout << ", contractStep: " << nelderMeadSimplex.contractStep;
    std::cout << ", shrinkStep: " << nelderMeadSimplex.shrinkStep << std::endl;
    return {nelderMeadSimplex.bestVertex().coordinate_, nelderMeadSimplex.bestVertex().value_};
}


#endif //NELDERMEAD_H
