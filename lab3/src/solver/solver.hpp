
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
using namespace boost::numeric;

#pragma once

class Solver {
public:
  Solver(size_t size, ublas::matrix<double> &coeffs,
         ublas::vector<double> &free_coeffs);
  ublas::vector<double> FindSolution();

private:
  ublas::vector<double> iterate_solution(ublas::vector<double> &sol,
                                         double tau);
  double calc_metric(ublas::vector<double> &sol);
  ublas::matrix<double> coeffs;
  ublas::vector<double> free_coeffs;
  size_t size;
};
