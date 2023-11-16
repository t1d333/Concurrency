#include "./solver/solver.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "omp.h"
#include <cstddef>
#include <iostream>

using namespace boost::numeric;

const size_t SIZE = 10;

ublas::matrix<double> get_coeffs(size_t size) {
  ublas::matrix<double> result(size, size);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result(i, j) = i == j ? 2.0 : 1.0;
    }
  }

  return result;
}

ublas::vector<double> get_free_coeffs(size_t size) {
  ublas::vector<double> result(size, size + 1);
  return result;
}

ublas::matrix<double> generate_random_matrix(size_t size) {}

int main() {
  ublas::matrix<double> coeffs = get_coeffs(SIZE);
  ublas::vector<double> free_coeffs = get_free_coeffs(SIZE);

  Solver s = Solver(SIZE, coeffs, free_coeffs);

  ublas::vector<double> result = s.FindSolution();

  for (size_t i = 0; i < SIZE; i++) {
    std::cout << *result.find_element(i) << " ";
  }
}
