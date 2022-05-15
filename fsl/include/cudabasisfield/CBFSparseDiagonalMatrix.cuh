//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Stores a mxn sparse matrix with p non-zero diagonals
/// \details This matrix makes no assumptions regarding symmetry or order of diagonals, but
///          these are important factors to consider when using the raw underlying values in
///          cuda kernels.
///
///          Example of how storage works for 3x3 matrix:
///
///               [1 2 3]    0 0[1 2 3]       [0 0 1 2 3]
///           A = [4 5 6] =>   0[4 5 6]0   => [0 4 5 6 0] + [-2 -1 0 1 2]
///               [7 8 9]       [7 8 9]0 0    [7 8 9 0 0]
///
///          I.e. the underlying data storage contains an mxp matrix where each column
///          contains the values of one of the diagonals, and the p-length vector contains
///          the offset of the diagonal into the original matrix, with 0 representing the main
///          diagonal, -ve for lower diagonals, and +ve for upper diagonals.
/// \author Frederik Lange
/// \date March 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CBF_SPARSE_DIAGONAL_MATRIX_CUH
#define CBF_SPARSE_DIAGONAL_MATRIX_CUH

#include <vector>
#include <memory>
#include <string>

#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include "miscmaths/bfmatrix.h"

namespace CBF
{
  class SparseDiagonalMatrix
  {
    public:
      // CONSTRUCTORS //
      /// Basic constructor
      SparseDiagonalMatrix (unsigned int n_rows, unsigned int n_cols,
                           const std::vector<int>& offsets);
      /// Copy constructor
      SparseDiagonalMatrix (const SparseDiagonalMatrix &obj);

      ///MEMBER FUNCTIONS //
      /// Get a pointer to the matrix values
      float *get_raw_pointer();
      /// Return vector of diagonal offsets
      std::vector<int> get_offsets() const;
      /// Save Matrix to file
      void save_matrix_to_text_file(std::string file_name);
      /// Convert to MISCMATHS::SparseBFMatrix csc format
      std::shared_ptr<MISCMATHS::BFMatrix> convert_to_sparse_bf_matrix(
          MISCMATHS::BFMatrixPrecisionType prec);
    protected:

    private:
      // DATAMEMBERS //
      unsigned int n_rows_;
      unsigned int n_cols_;
      /// The starting row for each diagonal
      std::vector<int> offsets_;
      /// The values of the matrix will be stored in a device_vector
      thrust::device_vector<float> d_values_;
  };
} ///namespace CBF
#endif // CBF_SPARSE_DIAGONAL_MATRIX_CUH
