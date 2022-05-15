/*! \file EddyMatrixKernels.h
    \brief Contains declarations of kernels used for LS resampling in Eddy project

    \author Jesper Andersson
    \version 1.0b, July, 2013
*/


#ifndef EddyMatrixKernels_h
#define EddyMatrixKernels_h

#include <cuda.h>

namespace EddyMatrixKernels {

__global__ void QR(// Input
		   const float  *K,     // Row-first matrices to decompose
		   unsigned int m,      // Number of rows of K
		   unsigned int n,      // Number of columns of K
		   unsigned int nmat,   // Number of matrices
		   // Output
		   float        *Qt,    // nmat mxm Q matrices
		   float        *R);    // nmat mxn R matrices

__global__ void Solve(// Input
		      const float *Qt,     // Orthogonal mxm matrices
		      const float *R,      // Upper diagonal mxn matrix
		      const float *y,      // mx1 column vectors
		      unsigned int m,      // No. of rows of Q and R
		      unsigned int n,      // No. of columns of R
		      unsigned int nmat,   // No. of matrices and vectors
		      // Output
		      float       *y_hat); // Solution b to y = Q*R*b

__global__ void KtK(// Input
		    const float  *K,     // Row/Column first input matrix
		    unsigned int m,      // No of rows of K
		    unsigned int n,      // No of columns of K
		    unsigned int nmat,   // Number of matrices
		    const float  *StS,   // Matrix to add to K'*K, Row/Column first
		    float        lambda, // Weight of StS
		    bool         rf,     // If true, K and StS are row-first
		    // Output
		    float        *KtK);  // K'*K + lambda*StS, row first

__global__ void Kty(// Input
                    const float   *K,      // Column first K matrices
		    const float   *y,      // y-vectors
		    unsigned int  m,       // No of rows of K and y
		    unsigned int  n,       // No of columns of K
		    unsigned int  nmat,    // Number of matrices
		    // Output
		    float         *Kty);   // K'*y, nx1

__global__ void Wir(// Input
                    const float   *zcoord,  // Volume of z-coordinates
		    unsigned int  xsz,      // x-size of vol with z-coords
		    unsigned int  ysz,      // y-size of vol with z-coords
		    unsigned int  zsz,      // z-size of vol with z-coords
		    unsigned int  nmat,     // Number or Wir matrices
		    unsigned int  xzp,      // xz-plane being considered
		    // Output
		    float         *Wir);    // Row-first Design matrices for irregularly sampled splines

__global__ void Wirty(// Input
		      const float   *y,       // Volume of y-vectors
		      const float   *Wir,     // Set of row-first Wir matrices
		      unsigned int  xsz,      // x-size of vol with z-coords
		      unsigned int  ysz,      // y-size of vol with z-coords
		      unsigned int  zsz,      // z-size of vol with z-coords
		      unsigned int  nmat,     // Number or Wir matrices
		      unsigned int  xzp,      // xz-plane being considered
		      // Output
		      float         *Wirty);  // Wir.t()*y vectors

__global__ void Atb(// Input
		    const float   *A,      // Row-first A matrix
		    const float   *b,      // b-vectors
		    unsigned int  m,       // No of rows of A
		    unsigned int  n,       // No of columns of A and rows of b
		    unsigned int  nmat,    // Number of matrices. Should be 1 or nvec
		    unsigned int  nvec,    // Number of vectors
		    // Output
		    float         *Atb);    // A'*b, nx1

__global__ void Ab(// Input
		   const float   *A,      // Row-first A matrix
		   const float   *b,      // b-vectors
		   unsigned int  m,       // No of rows of A
		   unsigned int  n,       // No of columns of A and rows of b
		   unsigned int  nmat,    // Number of matrices. Should be 1 or nvec
		   unsigned int  nvec,    // Number of vectors
		   // Output
		   float         *Ab);    // nvec set of A*b, mx1

__global__ void DiagwA(const float *w,    // mx1 vectors
		       const float *A,    // mxn matrix
		       unsigned int m,    // No. of rows
		       unsigned int n,    // No. of colums
		       unsigned int nvec, // No. of vectors/matrices
		       float        *wA); // diag{w}*A

} // End namespace EddyMatrixKernels

#endif // End #ifndef EddyMatrixKernels_h
