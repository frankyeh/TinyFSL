//
//  Declarations for functions generating specialised
//  sparse matrices of type SpMat
//
//  SpMatMatrices.h
//
//  Declares global functions for generating specialised
//  sparse matrices of type SpMat
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2019 University of Oxford
//
/*  CCOPYRIGHT  */

#ifndef SpMatMatrices_h
#define SpMatMatrices_h

#include <vector>
#include "armawrap/newmat.h"
#include "SpMat.h"

namespace MISCMATHS {

enum BoundaryCondition { MIRROR, PERIODIC };

/// Generates Symmetric Toeplitz matrix with first column given by col
SpMat<float> SparseSymmetricToeplitz(const NEWMAT::ColumnVector& col);

/// Generates Hessian for Bending Energy on a regular 3D grid
SpMat<float> Sparse3DBendingEnergyHessian(const std::vector<unsigned int>& isz,  // Matrix size
					  const std::vector<double>&       vxs,  // Voxel size
					  MISCMATHS::BoundaryCondition     bc);  // PERIODIC or MIRROR
} // End namespace MISCMATHS

#endif // End #ifndef SpMatMatrices_h
