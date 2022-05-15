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

#include <memory>
#include <vector>
#include <numeric>
#include <iostream>
#include <string>
#include <fstream>
#include <functional>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/count.h>

#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "miscmaths/bfmatrix.h"

#include "CBFSparseDiagonalMatrix.cuh"
#include "CBFKernels.cuh"
#pragma pop


namespace CBF
{
    // Basic constructor
    SparseDiagonalMatrix::SparseDiagonalMatrix(unsigned int n_rows, unsigned int n_cols,
                                               const std::vector<int>& offsets)
      :n_rows_(n_rows)
      ,n_cols_(n_cols)
      ,offsets_(offsets)
      ,d_values_(offsets_.size()*n_rows_,0.0)
    {}

    // Copy constructor
    SparseDiagonalMatrix::SparseDiagonalMatrix(const SparseDiagonalMatrix &obj)
      :n_rows_(obj.n_rows_)
      ,n_cols_(obj.n_cols_)
      ,offsets_(obj.offsets_)
      ,d_values_(obj.d_values_)
    {}

    // Get a pointer to the matrix values
    float *SparseDiagonalMatrix::get_raw_pointer()
    {
        return thrust::raw_pointer_cast(d_values_.data());
    } // get_raw_pointer

    std::vector<int> SparseDiagonalMatrix::get_offsets() const
    {
      return offsets_;
    } // get_offsets

    // Save Matrix to file
    void SparseDiagonalMatrix::save_matrix_to_text_file(std::string file_name)
    {
        // Attempt to create output stream
        std::ofstream matrix_file;
        matrix_file.open(file_name);
        // If successful
        if (matrix_file.is_open())
        {
            // Copy vector values from device to host
            thrust::host_vector<float> h_values = d_values_;
            // Copy values to file object
            std::ostream_iterator<float> file_iter(matrix_file,"\n");
            thrust::copy(h_values.begin(),h_values.end(),file_iter);
            matrix_file.close();
        }

        else
        {
            std::cout << "Failed to open file" << std::endl;
        }
        // Save diagonal numbers
        std::string diagonal_file_name = file_name + "_diagonals";
        std::ofstream diagonal_file;
        diagonal_file.open(diagonal_file_name);
        // If successful
        if (diagonal_file.is_open())
        {
            // Copy values to file object
            std::ostream_iterator<unsigned int> file_iter(diagonal_file,"\n");
            std::copy(offsets_.begin(),offsets_.end(),file_iter);
            diagonal_file.close();
        }
        else
        {
            std::cout << "Failed to open file" << std::endl;
        }
    } // save_matrix_to_text_file

    // Convert to MISCMATHS::SparseBFMatrix csc format
    std::shared_ptr<MISCMATHS::BFMatrix> SparseDiagonalMatrix::convert_to_sparse_bf_matrix(
        MISCMATHS::BFMatrixPrecisionType prec)
    {
      // Copy device data to host
      thrust::host_vector<float> d_values = d_values_;
      // Row, column, value triplets.
      auto col_ptrs = std::vector<unsigned int>(n_cols_ + 1);
      auto row_inds = std::vector<unsigned int>(d_values.size());
      auto mat_vals = std::vector<double>(d_values.size());
      // Initialise row pointers
      col_ptrs[0] = 0;
      // Loop through cols
      int val_count = 0;
      for (int c_i = 0; c_i < n_cols_; ++c_i){
        // Cumulative sum
        col_ptrs[c_i+1] = col_ptrs[c_i];
        // Loop through diagonals BACKWARDS
        for (int d_i = offsets_.size() - 1, d_i_end = 0; d_i >= d_i_end; --d_i){
          // Row index for this value in this diagonal
          auto r_i = c_i - offsets_[d_i];
          // If valid row
          if (r_i >= 0 && r_i < n_rows_){
            // Increment col ptr
            col_ptrs[c_i+1] += 1;
            // Store row index
            row_inds[val_count] = r_i;
            // Store Value
            mat_vals[val_count] = d_values[d_i*n_rows_ + r_i];
            ++val_count;
          }
        }
      }
      // Actually create the matrix
      std::shared_ptr<MISCMATHS::BFMatrix> csc_sbf;
      if (prec == MISCMATHS::BFMatrixFloatPrecision){
        csc_sbf = std::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<float>(
                n_rows_,
                n_cols_,
                row_inds.data(),
                col_ptrs.data(),
                mat_vals.data()));
      }
      else{
        csc_sbf = std::shared_ptr<MISCMATHS::BFMatrix>(
            new MISCMATHS::SparseBFMatrix<double>(
              n_rows_,
              n_cols_,
              row_inds.data(),
              col_ptrs.data(),
              mat_vals.data()));
      }
      return csc_sbf;
    } // convert_to_sparse_bf_matrix
} // namespace CBF
