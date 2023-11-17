#include "./solver/solver.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "omp.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace boost::numeric;

const size_t SIZE = 512;

ublas::matrix<double> get_coeffs(size_t size) {
  ublas::matrix<double> result(size, size);

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      result(i, j) = i == j ? 2.0 : 1.0;
    }
  }

  return result;
}

ublas::vector<double> gen_random_u(size_t size) {
  ublas::vector<double> result(size);

  for (size_t i = 0; i < size; i++) {
    result(i) = rand() % size;
  }
  return result;
}

int main() {
  std::cout << omp_get_max_threads() << std::endl;

  ublas::matrix<double> coeffs = get_coeffs(SIZE);
  ublas::vector<double> sol = gen_random_u(SIZE);
  ublas::vector<double> free_coeffs = ublas::prod(coeffs, sol);

  Solver s = Solver(SIZE, coeffs, free_coeffs);

  auto start = std::chrono::high_resolution_clock::now();

  ublas::vector<double> result = s.FindSolution();

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> duration = end - start;

  std::cout << "Execution time: " << duration.count() << " seconds."
            << std::endl;
}
