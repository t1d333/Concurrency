#include "solver.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <cmath>
#include <omp.h>

using namespace boost::numeric;

const double EPSILON = 10e-4;

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
    prev = metric;

    result = this->iterate_solution(result, tau);

    metric = this->calc_metric(result);

    if (prev < metric) {
      tau *= -1;
    }
  }

  return result;
}

ublas::vector<double> Solver::iterate_solution(ublas::vector<double> &sol,
                                               double tau) {

  ublas::vector<double> result(this->size);
#pragma omp parallel shared(result)
  {
    ublas::vector<double> tmp_free = ublas::subslice(
        this->free_coeffs,
        omp_get_thread_num() * (this->size / omp_get_max_threads()), 1,
        this->size / omp_get_max_threads());

    ublas::matrix<double> tmp_coeffs = ublas::subslice(
        this->coeffs,
        omp_get_thread_num() * (this->size / omp_get_max_threads()), 1,
        this->size / omp_get_max_threads(), 0, 1, this->size);
    ublas::vector<double> tmp = ublas::prod(tmp_coeffs, sol);
    tmp -= tmp_free;
    tmp *= tau;

    size_t thread_num = omp_get_thread_num();
    size_t chunk_size = this->size / omp_get_max_threads();
    size_t start_index = thread_num * chunk_size;
    size_t end_index = (thread_num == omp_get_max_threads() - 1)
                           ? this->size
                           : start_index + chunk_size;

#pragma omp barrier
    for (size_t j = start_index; j < end_index; ++j) {
      result(j) = tmp(j - start_index);
    }
#pragma omp barrier
  }

  return sol - result;
}

double Solver::calc_metric(ublas::vector<double> &sol) {

  double numerator = 0;
#pragma omp parallel shared(numerator)
  {
    ublas::vector<double> tmp_free = ublas::subslice(
        this->free_coeffs,
        omp_get_thread_num() * (this->size / omp_get_max_threads()), 1,
        this->size / omp_get_max_threads());

    ublas::matrix<double> tmp_coeffs = ublas::subslice(
        this->coeffs,
        omp_get_thread_num() * (this->size / omp_get_max_threads()), 1,
        this->size / omp_get_max_threads(), 0, 1, this->size);
    ublas::vector<double> tmp = ublas::prod(tmp_coeffs, sol);
    tmp -= tmp_free;

    double sum = 0;
    std::for_each(tmp.begin(), tmp.end(), [&sum](double d) { sum += d * d; });
#pragma omp atomic
    numerator += sum;
  }

  return std::sqrt(numerator) / ublas::norm_2(this->free_coeffs);
}
