//////////////////////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Utility functions for CUDA code
/// \details Mostly a number of functions for converting between linear and volumetric
///          indices.
/// \author Frederik Lange
/// \date February 2018
/// \copyright Copyright (C) 2018 University of Oxford
//////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CBF_KERNEL_HELPERS_CUH
#define CBF_KERNEL_HELPERS_CUH

#include <vector>
#include <cuda.h>

namespace CBF
{
    /// Convert from volume coordinates to linear coordinates
    __host__ __device__ void index_vol_to_lin(
                                // Input
                                // Vol coords
                                unsigned int xind, unsigned int yind, unsigned int zind,
                                // Vol dims
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                // Output
                                // Index into linear array
                                unsigned int *lind);

    /// Convert from linear coordinates to volume coordinates
    __host__ __device__ void index_lin_to_vol(
                                // Input
                                // Index into linear array
                                unsigned int lind,
                                // Vol dims
                                unsigned int szx, unsigned int szy, unsigned int szz,
                                 // Output
                                unsigned int *xind, unsigned int *yind, unsigned int *zind);

    /// Convert from linear coordinates back to linear coordinates of different spaces
    __host__ __device__ void index_lin_to_lin(
                                // Input
                                // Original linear index
                                unsigned int lind_in,
                                // size of original volume space
                                unsigned int lv_szx, unsigned int lv_szy, unsigned int lv_szz,
                                // size of new volume space
                                unsigned int vl_szx, unsigned int vl_szy, unsigned int vl_szz,
                                // Output
                                unsigned int *lind_out);

    /// Calculate the start and end row & column for each diagonal in the Hessian
    __host__ __device__ void identify_diagonal(
                                // Input
                                // Index of current diagonal
                                unsigned int diag_ind,
                                // No. of overlapping splines in each direction
                                unsigned int rep_x, unsigned int rep_y, unsigned int rep_z,
                                // Total no. of splines in each direction
                                unsigned int spl_x, unsigned int spl_y, unsigned int spl_z,
                                // Output
                                unsigned int *first_row, unsigned int *last_row,
                                unsigned int *first_column, unsigned int *last_column);
    /// Get the dimensions of the spline field coeffients based on warp parameterisaton and
    /// image size
    __host__ std::vector<unsigned int> get_spl_coef_dim(const std::vector<unsigned int>& ksp,
                                                        const std::vector<unsigned int>& isz);
} // namespace CBF
#endif // CBF_KERNEL_HELPERS_CUH
