#include "solver.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include <algorithm>
#include <boost/numeric/ublas/matrix_expression.hpp>
#include <cmath>
#include <math.h>

using namespace boost::numeric;

const size_t SIZE = 30000;
const double EPSILON = 1e-8;

Solver::Solver(size_t size, ublas::matrix<double> &coeffs,
               ublas::vector<double> &free_coeffs)
    : size(size) {
  this->coeffs = ublas::matrix<double>(coeffs);
  this->free_coeffs = ublas::vector<double>(free_coeffs);
}

ublas::vector<double> Solver::FindSolution() {

  ublas::vector<double> result(size);

  double tau = double(1) / this->size;
  double metric = std::numeric_limits<double>::max();
  double prev = metric;

  while (metric >= EPSILON) {
    result = this->iterate_solution(result, tau);
    prev = metric;
    metric = this->calc_metric(result);

    if (prev < metric) {
      tau *= -1;
    }
  }

  return result;
}

ublas::vector<double> Solver::iterate_solution(ublas::vector<double> &sol,
                                               double tau) {
  ublas::vector<double> tmp = ublas::prod(this->coeffs, sol);
  tmp -= this->free_coeffs;
  tmp *= tau;
  return sol - tmp;
}

double Solver::calc_metric(ublas::vector<double> &sol) {

  ublas::vector<double> tmp = ublas::prod(this->coeffs, sol);
  tmp -= this->free_coeffs;

  double numerator = 0;
  double denominator = 0;

  std::for_each(tmp.begin(), tmp.end(),
                [&numerator](double &n) { numerator += n * n; });

  std::for_each(this->free_coeffs.begin(), this->free_coeffs.end(),
                [&denominator](double &n) { denominator += n * n; });

  return sqrt(numerator) / sqrt(denominator);
}
