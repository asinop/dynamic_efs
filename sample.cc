#include <algorithm>
#include <cstdint>
#include <iostream>
#include <utility>
#include <vector>

#include <Eigen/Sparse>


// Sparse matrix type, where each value is a `double` and each row/column is
// indexed by an `int`.
using SparseMatrix = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
// Each entry of this sparse matrix is a triplet of the form {row, column, value}.
using SparseMatrixEntry = Eigen::Triplet<double, int>;
// Dense matrix type over `double` values.
using DenseMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

int main() {
  /* Creates the following sparse PD matrix (which corresponds to the
     Laplacian matrix of a 4-path graph with last row/column removed):

         [+1 -1  0]
     L = [-1 +2 -1]
         [0  -1 +2]
     Then solves the following system:
         [3]
     d = [2]
         [1]

     L x = d, prints out x.
  */
  std::vector<SparseMatrixEntry> entries;
  entries.push_back({/*row=*/0, /*column=*/0, /*value=*/+1.0});
  entries.push_back({/*row=*/0, /*column=*/1, /*value=*/-1.0});
  entries.push_back({/*row=*/1, /*column=*/0, /*value=*/-1.0});
  entries.push_back({/*row=*/1, /*column=*/1, /*value=*/+2.0});
  entries.push_back({/*row=*/1, /*column=*/2, /*value=*/-1.0});
  entries.push_back({/*row=*/2, /*column=*/1, /*value=*/-1.0});
  entries.push_back({/*row=*/2, /*column=*/2, /*value=*/+2.0});

  SparseMatrix laplacian_matrix(/*number_of_rows=*/ 3,
                                /*number_of_columns=*/ 3);
  laplacian_matrix.setFromTriplets(entries.begin(), entries.end());

  DenseMatrix rhs = DenseMatrix::Zero(/*number_of_rows=*/3,
                                       /*number_of_columns=*/1);
  rhs(0, 0) = +3.0;
  rhs(1, 0) = +2.0;
  rhs(2, 0) = +1.0;

  // Setup the preconditioned conjugate gradient solver with incomplete
  // cholesky preconditioner.
  Eigen::ConjugateGradient<
      SparseMatrix, Eigen::Lower | Eigen::Upper,
      Eigen::IncompleteCholesky<double, Eigen::Lower | Eigen::Upper>>
      conj_gradient;
  conj_gradient.compute (laplacian_matrix);
  DenseMatrix solution = conj_gradient.solve(rhs);


  std::cout << laplacian_matrix.toDense() << std::endl;
  std::cout << " * \n" << solution << std::endl;
  std::cout << " = \n" << rhs << std::endl;


  return 0;
}
