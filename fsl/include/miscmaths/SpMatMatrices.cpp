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

#include <vector>
#include "armawrap/newmat.h"
#include "SpMat.h"
#include "SpMatMatrices.h"

namespace MISCMATHS {

/*!
 * Global function that creates and returns a symmetric
 * Toeplitz matrix with dimensions col.Nrows() x col.Nrows() and where the
 * first column is given by col and all subsequent columns are translated
 * and shifted versions of that column.
 * \return A sparse symmetric Toeplitz matrix
 * \param[in] col First column of matrix
 */
MISCMATHS::SpMat<float> SparseSymmetricToeplitz(const NEWMAT::ColumnVector& col)
{
  unsigned int mn = static_cast<unsigned int>(col.Nrows());
  unsigned int nnz = 0; // No of non-zeros per column
  for (unsigned int i=0; i<mn; i++) nnz += (col(i+1) == 0) ? 0 : 1;
  std::vector<unsigned int> indx(nnz);
  std::vector<float> val(nnz);
  {
    unsigned int i = 0; unsigned int j = 0;
    for (i=0, j=0; i<mn; i++) if (col(i+1) != 0) { indx[j] = i; val[j++] = static_cast<float>(col(i+1)); }
  }
  unsigned int *irp = new unsigned int[nnz*mn];
  unsigned int *jcp = new unsigned int[mn+1];
  double *sp = new double[nnz*mn];
  unsigned int irp_cntr = 0;
  for (unsigned int col=0; col<mn; col++) {
    jcp[col] = irp_cntr;
    for (unsigned int r=0; r<nnz; r++) {
      irp[irp_cntr] = indx[r];
      sp[irp_cntr++] = val[r];
      indx[r] = (indx[r] == mn-1) ? 0 : indx[r]+1;
    }
  }
  jcp[mn] = irp_cntr;
  MISCMATHS::SpMat<float> tpmat(mn,mn,irp,jcp,sp);
  delete [] irp; delete [] jcp; delete [] sp;
  return(tpmat);
}

/*!
 * Global function that creates and returns a symmetric matrix with dimensions
 * prod(isz) x prod(isz) and which represent an approximate Hessian for
 * Bending energy. It is approximate because it only considers the straight
 * second derivatives.
 * \return A sparse symmetric Hessian of Bending Energy
 * \param[in] isz 3 element vector specifying matrix size of image
 * \param[in] vxs 3 element vector with voxel size in mm
 * \param[in] bc Boundary condition (PERIODIC or MIRROR)
 */
MISCMATHS::SpMat<float> Sparse3DBendingEnergyHessian(const std::vector<unsigned int>& isz,
						     const std::vector<double>&       vxs,
						     MISCMATHS::BoundaryCondition     bc)
{
  unsigned int mn = isz[0]*isz[1]*isz[2];
  unsigned int *irp = new unsigned int[3*mn]; // Worst case, might be slightly smaller
  unsigned int *jcp = new unsigned int[mn+1];
  double *sp = new double[3*mn];
  // x-direction
  unsigned int irp_cntr = 0;
  double sf = 1 / vxs[0]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (i==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + isz[0]-1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	  else if (i==isz[0]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (i==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0];
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + 1;
	    sp[irp_cntr++] = -2.0 * sf;
	  }
	  else if (i==isz[0]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i - 1;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i + 1;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  MISCMATHS::SpMat<float> At(mn,mn,irp,jcp,sp);
  MISCMATHS::SpMat<float> AtA = At * At.t();

  // y-direction
  irp_cntr = 0;
  sf = 1 / vxs[1]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (j==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (isz[1]-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	  }
	  else if (j==isz[1]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0;
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j+1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (j==0) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + isz[0] + i;
	    sp[irp_cntr++] = -2.0;
	  }
	  else if (j==isz[1]-1) {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -2.0;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0;
	  }
	  else {
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j-1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + (j+1)*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  At = MISCMATHS::SpMat<float>(mn,mn,irp,jcp,sp);
  AtA += At * At.t();

  // z-direction
  irp_cntr = 0;
  sf = 1 / vxs[2]; // Let us scale all directions to mm^{-1}
  for (unsigned int k=0; k<isz[2]; k++) {
    for (unsigned int j=0; j<isz[1]; j++) {
      for (unsigned int i=0; i<isz[0]; i++) {
	jcp[k*isz[0]*isz[1] + j*isz[0] + i] = irp_cntr;
	if (bc == MISCMATHS::PERIODIC) {
	  if (k==0) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = (isz[2]-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	  else if (k==isz[2]-1) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	  }
	  else {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = (k+1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
	else if (bc == MISCMATHS::MIRROR) {
	  if (k==0) {
	    irp[irp_cntr] = j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -2.0 * sf;
	  }
	  else if (k==isz[2]-1) {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -2.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	  }
	  else {
	    irp[irp_cntr] = (k-1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	    irp[irp_cntr] = k*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = 2.0 * sf;
	    irp[irp_cntr] = (k+1)*isz[0]*isz[1] + j*isz[0] + i;
	    sp[irp_cntr++] = -1.0 * sf;
	  }
	}
      }
    }
  }
  jcp[mn] = irp_cntr;
  At = MISCMATHS::SpMat<float>(mn,mn,irp,jcp,sp);
  AtA += At * At.t();
  delete [] irp; delete [] jcp; delete [] sp;
  return(AtA);
}

} // End namespace MISCMATHS
