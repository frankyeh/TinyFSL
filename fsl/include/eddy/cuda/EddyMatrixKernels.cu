/*! \file EddyMatrixKernels.cu
    \brief Contains definitions of kernels used for LS resampling in Eddy project

    \author Jesper Andersson
    \version 1.0b, July, 2013
*/

#include <stdio.h>
#include <cuda.h>
#include <math_constants.h>
#include <math_functions.h>
#include "EddyKernels.h"
#include "eddy_matrix_kernels_internal.h"
#include "EddyMatrixKernels.h"

/*****************************************************************//**
*
* \brief Contains __global__ CUDA kernels that can be called by the 
* application programmer
*
* Contains __global__ CUDA kernels that was implemented to allow for
* LS-resampling of pairs of images as part of the eddy project.
* There are routines for multiplying a matrix with its own transpose,
* for multiplying a matrix with a vector, for performing QR
* decomposition of a matrix and for performing backsubstitution based
* on the QR decomposition.
* 
*********************************************************************/

namespace EddyMatrixKernels {

using namespace EMKI;

/*****************************************************************//**
*
* Performs a QR decompostion on a set of mxn matrices. The QR
* decomposition of an mxn matrix M will produce mxm matrix Q and
* mxn matrix R so that M=Q*R where Q is orthogonal and R is 
* upper diagonal. 
* \param[in]  K A device-pointer to a set of matrices where each matrix
*             has m*n elements and is organised row-first
* \param[in]  m Number of rows of each matrix in K
* \param[in]  n Number of columns of each matrix in K
* \param[in]  nmat Number of matrices in K
* \param[out] Qt The mxm Q matrices (transposed) for each matrix in K.
*             Row first.
* \param[out] R The mxn R matrices for each matrix in K.
*             Row first. 
*
* \attention  This routine uses dynamic shared memory so the kernel
*             call pre-amble needs to include the size of the shared
*             memory, which is 2*m*sizeof(float).
*             I.e. <<<num_blocks,threads_per_block,2*m*sizeof(float)>>>
* 
*********************************************************************/					
__global__ void QR(// Input
		   const float  *K,     // Row-first matrices to decompose
		   unsigned int m,      // Number of rows of K
		   unsigned int n,      // Number of columns of K
		   unsigned int nmat,   // Number of matrices
		   // Output
		   float        *Qt,    // nmat mxm Q matrices
		   float        *R)     // nmat mxn R matrices
{
  extern __shared__ float scratch[];

  if (blockIdx.x < nmat && threadIdx.x < m) {
    unsigned int id = threadIdx.x;
    unsigned int ntpm = min(m,blockDim.x); // Number of threads per matrix
    float *v = scratch;
    float *w = &scratch[m];
    const float *lK = &K[blockIdx.x*m*n];
    float *lQt = &Qt[blockIdx.x*m*m];
    float *lR = &R[blockIdx.x*m*n];
    qr_single(lK,m,n,v,w,id,ntpm,lQt,lR);
  }
  return;
}

/*****************************************************************//**
*
* Solves for b in y=Q*R*b for a set of matrices and vectors.
* It solves it by Q'*y = y' = R*b since Q is orthognal and since
* R is upper diagonal it is easily solved through back substitution.
* \param[in]  Qt A set of row-first orthogonal mxm matrices. Typically these 
*             are the returned matrices from a call to QR.
* \param[in]  R A set of row-first upper diagonal mxn matrices. Typically 
*             these are the returned matrices from a call to QR.
* \param[in]  y A set of mx1 column vectors.
* \param[in]  m No. of rows of Qt, R and y.
* \param[in]  n No. of columns of R.
* \param[in]  nmat No. of matrices and vectors.
* \param[out] y_hat A set of nx1 solutions corresponding to b in the
*             equation above. 
* 
*********************************************************************/					
__global__ void Solve(// Input
		      const float *Qt,   // Orthogonal mxm matrices
		      const float *R,    // Upper diagonal mxn matrix
		      const float *y,    // mx1 column vectors
		      unsigned int m,    // No. of rows of Q and R
		      unsigned int n,    // No. of columns of R
		      unsigned int nmat, // No. of matrices and vectors
		      // Output
		      float       *y_hat)// Solution b to y = Q*R*b
{
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nmat) return;
  const float *lQt = &Qt[id*m*m];
  const float *lR = &R[id*m*n];
  const float *ly = &y[id*m];
  float *ly_hat = &y_hat[id*n];
  solve_single(lQt,lR,ly,m,n,ly_hat);
  return;
}

/*****************************************************************//**
*
* Pre-multiplies a set of input matrices, K, with its transpose to 
* generate a set of K'*K matrices. If lambda is non-zero it will 
* generate KtK+lambda*StS where the same (pre-defined) matrix StS 
* is used for all (the different) KtK matrices.
* \param[in]  K A set of mxn matrices.
* \param[in]  m No. of rows of K
* \param[in]  n No. of columns of K.
* \param[in]  nmat No. of matrices.
* \param[in]  StS A single nxn matrix that gets added to each KtK.
* \param[in]  lambda The weight of StS
* \param[in]  rf If true, K and StS are expected to be row-first
* \param[out] KtK A set of nmat nxn matrices, always row_first.
* 
*********************************************************************/					
__global__ void KtK(// Input
		    const float  *K,     // Row/Column first input matrix
		    unsigned int m,      // No of rows of K
		    unsigned int n,      // No of columns of K
		    unsigned int nmat,   // Number of matrices
		    const float  *StS,   // Matrix to add to K'*K, Row/Column first
		    float        lambda, // Weight of StS
		    bool         rf,     // If true, K and StS are row-first
		    // Output
		    float        *KtK)   // K'*K + lambda*StS, row first
{
  if (blockIdx.x < nmat) {
    const float    *lK = &K[blockIdx.x*m*n];
    float          *lKtK = &KtK[blockIdx.x*n*n];
    if (rf) rf_KtK_one_mat(lK,m,n,StS,lambda,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,lKtK);
    else cf_KtK_one_mat(lK,m,n,StS,lambda,threadIdx.x,threadIdx.y,blockDim.x,blockDim.y,lKtK);
  }
  return;
}

/*****************************************************************//**
*
* Multiplies a set of vectors, y, with the transposes of a set of
* matrices K.
* \param[in]  K A set of column-first mxn matrices.
* \param[in]  y A set of mx1 vectors.
* \param[in]  m No. of rows of K
* \param[in]  n No. of columns of K.
* \param[in]  nmat No. of matrices.
* \param[out] Kty A set of nmat nx1 vectors.
* 
*********************************************************************/					
__global__ void Kty(// Input
                    const float   *K,      // Column first K matrices
		    const float   *y,      // y-vectors
		    unsigned int  m,       // No of rows of K and y
		    unsigned int  n,       // No of columns of K
		    unsigned int  nmat,    // Number of matrices
		    // Output
		    float         *Kty)    // K'*y, nx1
{
  if (blockIdx.x < nmat) {
    const float *lK = &K[blockIdx.x*m*n];
    const float *ly = &y[blockIdx.x*m];
    float *lKty = &Kty[blockIdx.x*n];
    Kty_one_mat(lK,ly,m,n,threadIdx.x,blockDim.x,lKty);
  }
  return;
}

/*****************************************************************//**
*
* Makes a "design matrix" for cubic B-splines on an irregular grid.
*
* \param[in]  zccord Columns of irregular z-ccordinates in volume format
* \param[in]  xsz x-size of zcoord
* \param[in]  xsz x-size of zcoord
* \param[in]  xsz x-size of zcoord
* \param[in]  nmat No. of matrices.
* \param[in]  xzp index of xz-plane being considered
* \param[out] Wir A set of row-first nmat xsz*xsz matrices.
* 
*********************************************************************/					
__global__ void Wir(// Input
                    const float   *zcoord,  // Volume of z-coordinates
		    unsigned int  xsz,      // x-size of vol with z-coords
		    unsigned int  ysz,      // y-size of vol with z-coords
		    unsigned int  zsz,      // z-size of vol with z-coords
		    unsigned int  nmat,     // Number or Wir matrices
		    unsigned int  xzp,      // xz-plane being considered
		    // Output
		    float         *Wir)     // Row-first Design matrices for irregularly sampled splines
{
  if (blockIdx.x < nmat) {
    float *lWir = &Wir[blockIdx.x*zsz*zsz];
    const float *lzcoord = &zcoord[xzp*xsz+blockIdx.x];
    unsigned int zstep = xsz*ysz;
    Wir_one_mat(lzcoord,zstep,zsz,threadIdx.x,blockDim.x,lWir);
  }
  return;
}

/*****************************************************************//**
*
* Multiply transposed "design matrices" for cubic B-splines on an 
* irregular grid with y-vectors embedded as the columns in the 
* z-direction of an image volume
*
* \param[in]  y Volume where each column in z-dir is a y-vector
* \param[in]  Wir Set of nmat row-first zsz*zsz spline matrices
* \param[in]  xsz x-size of zcoord
* \param[in]  xsz x-size of zcoord
* \param[in]  xsz x-size of zcoord
* \param[in]  nmat No. of matrices.
* \param[in]  xzp index of xz-plane being considered
* \param[out] Wirty A set of Wir.t()*y vectors
* 
*********************************************************************/					
__global__ void Wirty(// Input
		      const float   *y,       // Volume of y-vectors
		      const float   *Wir,     // Set of row-first Wir matrices
		      unsigned int  xsz,      // x-size of vol with y-vectors
		      unsigned int  ysz,      // y-size of vol with y-vectors
		      unsigned int  zsz,      // z-size of vol with y-vectors
		      unsigned int  nmat,     // Number or Wir matrices
		      unsigned int  xzp,      // xz-plane being considered
		      // Output
		      float         *Wirty)   // Wir.t()*y vectors
{
  if (blockIdx.x < nmat) {
    const float *ly = &y[xzp*xsz+blockIdx.x];
    const float *lWir = &Wir[blockIdx.x*zsz*zsz];
    unsigned int zstep = xsz*ysz;
    float *lWirty = &Wirty[blockIdx.x*zsz];
    Wirty_one_mat(ly,lWir,zstep,zsz,threadIdx.x,blockDim.x,lWirty);
  }
  return;
}

/*****************************************************************//**
*
* Multiplies a set of vectors, b, with the transposes of a set 
* of matrices A or a single matrix A. Should be called with <<<nvec, >>>
* \param[in]  A A single or a set of row-first mxn matrices.
* \param[in]  b A set of mx1 vectors.
* \param[in]  m No. of rows of A
* \param[in]  n No. of columns of A.
* \param[in]  nmat No. of matrices. Should be 1 or nvec
* \param[in]  nvec No. of vectors.
* \param[out] Ab A set of nvec nx1 vectors.
* 
*********************************************************************/	
__global__ void Atb(// Input
		    const float   *A,      // Row-first A matrix
		    const float   *b,      // b-vectors
		    unsigned int  m,       // No of rows of A
		    unsigned int  n,       // No of columns of A and rows of b
		    unsigned int  nmat,    // Number of matrices. Should be 1 or nvec
		    unsigned int  nvec,    // Number of vectors
		    // Output
		    float         *Atb)     // A'*b, nx1
{
  if (blockIdx.x < nvec) {
    const float *lA;
    if (nmat==1) lA = A;
    else if (nmat==nvec) lA = &A[blockIdx.x*m*n];
    else *(int*)0 = 0; // Throw a fit
    const float *lb = &b[blockIdx.x*n];
    float *lAtb = &Atb[blockIdx.x*m];
    Atb_one_mat(lA,lb,m,n,threadIdx.x,blockDim.x,lAtb);
  }
  return;
}
/*****************************************************************//**
*
* Multiplies a set of vectors, b, with a single matrix A or a 
* set of matrices A. Should be called with <<<nvec, >>>
* \param[in]  A A single or nvec row-first mxn matrix/matrices.
* \param[in]  b A set of nx1 vectors.
* \param[in]  m No. of rows of A
* \param[in]  n No. of columns of A.
* \param[in]  nmat No. of matrices. Should be 1 or nvec
* \param[in]  nvec No. of vectors.
* \param[out] Ab A set of nvec mx1 vectors.
* 
*********************************************************************/					
__global__ void Ab(// Input
		   const float   *A,      // Row-first A matrix
		   const float   *b,      // b-vectors
		   unsigned int  m,       // No of rows of A
		   unsigned int  n,       // No of columns of A and rows of b
		   unsigned int  nmat,    // Number of matrices. Should be 1 or nvec
		   unsigned int  nvec,    // Number of vectors
		   // Output
		   float         *Ab)     // A*b, mx1
{
  if (blockIdx.x < nvec) {
    const float *lA;
    if (nmat==1) lA = A;
    else if (nmat==nvec) lA = &A[blockIdx.x*m*n];
    else *(int*)0 = 0; // Throw a fit
    const float *lb = &b[blockIdx.x*n];
    float *lAb = &Ab[blockIdx.x*m];
    Ab_one_mat(lA,lb,m,n,threadIdx.x,blockDim.x,lAb);
  }
  return;
}

/*****************************************************************//**
*
* Given a set of vectors, w, and a matrix A it calculates
* a set of diag{w}*A matrices. It must be called with <<<nvec,m>>>
* \param[in]  w A set of mx1 vectors
* \param[in]  A A set of mxn row-first matrices.
* \param[in]  m No. of rows of A
* \param[in]  n No. of columns of A.
* \param[in]  nvec No. of vectors.
* \param[out] wA A set of nvec mxn row-first matrices.
* 
*********************************************************************/					
__global__ void DiagwA(const float *w,    // mx1 vectors
		       const float *A,    // mxn matrix
		       unsigned int m,    // No. of rows
		       unsigned int n,    // No. of colums
		       unsigned int nvec, // No. of vectors/matrices
		       float        *wA)  // diag{w}*A
{
  unsigned int mat=blockIdx.x;
  unsigned int row=threadIdx.x;
  if (mat<nvec && row<m) {
    float wgt = w[mat*m+row];
    float *wAp = wA + mat*m*n;
    for (unsigned int c=0; c<n; c++) {
      wAp[rf_indx(row,c,m,n)] = A[rf_indx(row,c,m,n)] * wgt;
    }
  }
}

} // End namespace EddyMatrixKernels

/// Contains device kernels that are used internally by the functions in EddyMatrixKernels
namespace EMKI { // Eddy Matrix Kernels Internal

// First a set of __device__ kernels used by __global__ kernel QR

__device__ void qr_single(// Input
			  const float *K,       // mxn matrix to be QR-decomposed
			  unsigned int m,
			  unsigned int n,
			  // Scratch
			  float       *v,       // m elements of scratch space
			  float       *w,       // m elements of scratch space
			  // Thread info
			  unsigned int id,       // Thread id
			  unsigned int nt,       // # of threads for this matrix
			  // Output
			  float       *Qt,      // Q' in K = Q'*R
			  float       *R)       // R in K = Q'*R
{
  // R = K;
  M_eq_M(R,K,m,n,nt,id);
  // Q = eye;
  set_to_identity(Qt,m,nt,id);
  // Loop over columns of K
  for (int j=0; j<min(m-1,n); j++) {
    // alfa = sign(x(i))*norm(x) where x is the i'th column of R
    float alfa = get_alfa(R,m,n,j,nt,id,v);
    // v is used to contruct Householder matrix as H = I - 2*v*v'
    get_v(R,m,n,j,alfa,nt,id,v);
    // w is used as: H*R = (I-2*v*v')*R = R-v*w' where w'=2*v'*R
    two_x_vt_x_R(R,m,n,j,nt,id,v,w);
    // R = H*R = R - v*w';
    R_minus_v_x_wt(m,n,j,nt,id,v,w,R);
    // w is used as: H*Q' = (I-2*v*v')*Q' = Q'-v*w' where w'=2*v'*Q'
    two_x_vt_x_Qt(Qt,m,m,j,nt,id,v,w);
    // Q' = H*Q' = Q' - v*w';
    Qt_minus_v_x_wt(m,m,j,nt,id,v,w,Qt);
  }
  return;
}

__device__ void M_eq_M(float       *dM, // Destination matrix
                       const float *sM, // Source matrix
		       unsigned int m,   // No. of rows
		       unsigned int n,   // No. of columns
		       unsigned int nt,  // No. of threads for this matrix
		       unsigned int id)  // Number of this thread (among nt)
{
  unsigned int rpt = m/nt; rpt += (m%nt) ? 1 : 0;
  for (int r=0; r<rpt; r++) {
    unsigned int i = id + r*nt;
    if (i < m) {
      for (int j=0; j<n; j++) {
        unsigned int ii = rf_indx(i,j,m,n);
	dM[ii] = sM[ii];
      }
    }
  }
  return;
}

__device__ void set_to_identity(float        *M,   // Matrix to set
                                unsigned int  m,   // Size of matrix
				unsigned int  nt,  // Number of threads for this matrix
				unsigned int  id)  // Number of this thread (among nt)
{
  unsigned int rpt = m/nt; rpt += (m%nt) ? 1 : 0;
  for (int r=0; r<rpt; r++) {
    unsigned int i = id + r*nt;
    if (i < m) {
      for (int j=0; j<m; j++) {
	unsigned int ii = rf_indx(i,j,m,m);
	if (i==j) M[ii] = 1.0;
	else M[ii] = 0.0;
      }
    }
  }
  return;
}

__device__ float get_alfa(const float  *M,   // Matrix
			  unsigned int m,    // No of rows
			  unsigned int n,    // No of columns
			  unsigned int j,    // Column to get alfa for
			  unsigned int nt,   // No of threads for this matrix
			  unsigned int id,   // Number of thread (among nt)
			  float        *scr) // Scratch space for reduction
{
  unsigned int ept = (m-j)/nt; ept += ((m-j)%nt) ? 1 : 0; // Elements per thread
  scr[id] = 0.0;
  for (int r=0; r<ept; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) scr[id] += sqr(M[rf_indx(i,j,m,n)]);
  }
  __syncthreads();
  unsigned int npt = nt;
  if (!is_pow_of_two(npt)) npt = next_pow_of_two(nt);
  unsigned int s = npt>>1;
  if (id<s && id+s<nt) scr[id] += scr[id+s];
  __syncthreads();
  for (s>>=1; s>0; s>>=1) {
    if (id<s) scr[id] += scr[id+s];
    __syncthreads();
  }
  float alfa = sqrt((float) scr[0]);
  alfa = (M[rf_indx(j,j,m,n)] > 0) ? alfa : -alfa;

  return(alfa);
}

__device__ void get_v(const float  *M,  // The matrix (full size)
		      unsigned int m,   // No. of rows of M
		      unsigned int n,   // No. of columns of M
		      unsigned int j,   // Column to extract v for
		      float        alfa,// Alfa
		      unsigned int nt,  // No. of threads for this matrix
		      unsigned int id,  // Thread number among nt
		      float        *v)  // Space for v
{
  unsigned int ept = (m-j)/nt; ept += ((m-j)%nt) ? 1 : 0; // elements per thread
  // First put raw vector into v
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i<m) v[i] = M[rf_indx(i,j,m,n)];
  }
  __syncthreads();
  // Next calculate the norm of v
  float norm_v = sqrt(2.0*sqr(alfa) + 2.0*v[j]*alfa);
  // Normalise v
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i==j) v[i] += alfa;
    if (i<m) v[i] /= norm_v;
  }
  __syncthreads();
  return;
}

__device__ void two_x_vt_x_R(const float  *R,      // R (first j columns upper diag)
			     unsigned int m,       // No. of rows of R
			     unsigned int n,       // No. of rows of R
			     unsigned int j,       // The column we currently work on
                             unsigned int nt,      // No. of threads for this matrix
                             unsigned int id,      // Thread number (out of nt)
			     const float  *v,      // v (first j elements zero)
			     float        *twovtR) // 2.0*v'*R, first j elements zero
{
  // Zero the first j elements in twovtR
  unsigned int ept = j/nt; ept += (j%nt) ? 1 : 0; // elements per thread  
  for (int e=0; e<ept; e++) {
    unsigned int i = id + e*nt;
    if (i<j) twovtR[i] = 0.0;
  }
  __syncthreads();
  // Calculate the rest of the elements in twovtR
  ept = (n-j)/nt; ept += ((n-j)%nt) ? 1 : 0; // elements per thread
  // printf("ept = %d\n",ept);
  for (int e=0; e<ept; e++) {
    unsigned int i = j + id + e*nt;
    if (i<n) {
      twovtR[i] = 0.0;
      for (int ii=j; ii<m; ii++) twovtR[i] += v[ii]*R[rf_indx(ii,i,m,n)];
      twovtR[i] *= 2.0;
      // printf("id = %d, i = %d, twovtR[%d] = %f\n",id,i,i,twovtR[i]);
    }
  }  
  __syncthreads();
  return;
}

__device__ void R_minus_v_x_wt(unsigned int m,  // No. of rows of R
			       unsigned int n,  // No. of columns of R
			       unsigned int j,  // The column we are currently on
			       unsigned int nt, // No. of threads for this matrix
			       unsigned int id, // Number of this thread (within nt)
			       const float  *v, // v, mx1, j first elements zero
			       const float  *w, // w, nx1, j first elements zero
			       float        *R) // R in, R - vw' out 
{
  unsigned int rpt = (m-j)/nt; rpt += ((m-j)%nt) ? 1 : 0; // rows per thread
  for (unsigned int r=0; r<rpt; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) {
      for (int jj=j; jj<n; jj++) R[rf_indx(i,jj,m,n)] -= v[i]*w[jj];
    }
  }
  __syncthreads();
  return;
}

__device__ void two_x_vt_x_Qt(const float  *Qt,     // Q'
			      unsigned int m,       // No. of rows of Qt
			      unsigned int n,       // No. of columns of Qt
			      unsigned int j,       // The column we currently work on
			      unsigned int nt,      // No. of threads for this matrix
			      unsigned int id,      // Thread number (out of nt)
			      const float  *v,      // v (first j elements zero)
			      float        *twovtQt)// 2.0*v'*Q'
{
  // Calculate the elements in twovtQt
  unsigned int ept = m/nt; ept += (m%nt) ? 1 : 0; // elements per thread
  for (int e=0; e<ept; e++) {
    unsigned int i = id + e*nt;
    if (i<n) {
      twovtQt[i] = 0.0;
      for (int ii=j; ii<m; ii++) twovtQt[i] += v[ii]*Qt[rf_indx(ii,i,m,n)];
      twovtQt[i] *= 2.0;
    }
  }  
  __syncthreads();
  return;
}

__device__ void Qt_minus_v_x_wt(unsigned int m,  // No. of rows of Q'
			        unsigned int n,  // No. of columns of Q'
			        unsigned int j,  // The column we are currently on
			        unsigned int nt, // No. of threads for this matrix
			        unsigned int id, // Number of this thread (within nt)
			        const float  *v, // v, mx1, j first elements zero
			        const float  *w, // w, nx1
			        float        *Qt)// Q' in, Q' - vw' out 
{
  unsigned int rpt = (m-j)/nt; rpt += ((m-j)%nt) ? 1 : 0; // rows per thread
  for (int r=0; r<rpt; r++) {
    unsigned int i = j + id + nt*r;
    if (i<m) {
      for (int jj=0; jj<n; jj++) Qt[rf_indx(i,jj,m,n)] -= v[i]*w[jj];
    }
  }
  __syncthreads();
  return;
}

__device__ bool is_pow_of_two(unsigned int n)
{
  return(!(n & (n-1)));
}
__device__ unsigned int next_pow_of_two(unsigned int n)
{
  n--; n|=n>>1; n|=n>>2; n|=n>>4; n|=n>>8; n|=n>>16; n++;
  return(n);
}

// Row-first indexing
__device__ unsigned int rf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n)
{
  return(i+j*m);    // This is for row-first matrices
}

// Column-first indexing
__device__ unsigned int cf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n)
{
  return(i*n+j);
}


// Here comes a set of __device__ kernels used by __global__ kernel Solve

__device__ void solve_single(// Input
			     const float *Qt,     // Row-first orthogonal mxm matrix  
			     const float *R,      // Row-first upper-diagonal mxn matrix
			     const float *y,      // mx1 vector
			     unsigned int m,
			     unsigned int n,
			     // Output
			     float       *y_hat)  // nx1 solution vector
{
  // Calculates top half of Q'*y and puts it in y_hat
  M_times_v(Qt,y,n,m,y_hat);
  // Does a backsubstitution "in place" in y_hat
  back_substitute(R,n,n,y_hat);
}

__device__ void back_substitute(const float  *R,
				unsigned int  m,
				unsigned int  n,
				float        *v)
{
  for (int i=n-1; i>=0; i--) {
    float tmp = v[i];
    for (int j=n-1; j>i; j--) tmp -= R[rf_indx(i,j,m,n)] * v[j];
    v[i] = tmp / R[rf_indx(i,i,m,n)];
  }
  return;
}

__device__ void M_times_v(const float *M,
			  const float *v,
			  unsigned int m,
			  unsigned int n,
			  float       *Mv)
{
  for (int i=0; i<m; i++) {
    Mv[i] = 0.0;
    for (int j=0; j<n; j++) Mv[i] += M[rf_indx(i,j,m,n)] * v[j];
  }
  return;
}

// __device__ routines for matrix-matrix and matrix-vector multiplication

__device__ void cf_KtK_one_mat(const float  *K,   // Column-first mxn
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, // Column-first nxn
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK) // Row-first nxn
{
  unsigned int rpt = n/ntr; rpt += (n%ntr) ? 1 : 0; // No of row indicies per thread
  unsigned int cpt = n/ntc; cpt += (n%ntc) ? 1 : 0; // No of column indicies per thread
  for (int i=0; i<rpt; i++) {
    unsigned int r = idr + i*ntr;
    if (r < n) {
      for (int j=0; j<cpt; j++) {
	unsigned int c = idc + j*ntc;
	if (c < n) {
	  float val = 0.0;
	  for (int jj=0; jj<m; jj++) {
	    val += K[cf_indx(jj,r,m,n)]*K[cf_indx(jj,c,m,n)];
	  }
	  if (StS && lambda) KtK[rf_indx(r,c,n,n)] = val + StS[cf_indx(r,c,n,n)];
	  else KtK[rf_indx(r,c,n,n)] = val;
	}
      }
    }
  }
  return;
}

__device__ void rf_KtK_one_mat(const float  *K,   // Row-first mxn
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, // Row-first nxn
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK) // Row-first nxn
{
  unsigned int rpt = n/ntr; rpt += (n%ntr) ? 1 : 0; // No of row indicies per thread
  unsigned int cpt = n/ntc; cpt += (n%ntc) ? 1 : 0; // No of column indicies per thread
  for (int i=0; i<rpt; i++) {
    unsigned int r = idr + i*ntr;
    if (r < n) {
      for (int j=0; j<cpt; j++) {
	unsigned int c = idc + j*ntc;
	if (c < n) {
	  float val = 0.0;
	  for (int jj=0; jj<m; jj++) {
	    val += K[rf_indx(jj,r,m,n)]*K[rf_indx(jj,c,m,n)];
	  }
	  if (StS && lambda) KtK[rf_indx(r,c,n,n)] = val + StS[rf_indx(r,c,n,n)];
	  else KtK[rf_indx(r,c,n,n)] = val;
	}
      }
    }
  }
  return;
}

__device__ void Ab_one_mat(// Input
			   const float   *A,  // Row-first
			   const float   *b,
			   unsigned int  m,
			   unsigned int  n,
			   unsigned int  id,
			   unsigned int  ntr,
			   // Output
			   float         *Ab)
{
  unsigned int ept = m/ntr; ept += (m%ntr) ? 1 : 0; // No of elements (of Ab) per thread
  for (int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < m) {
      float val = 0.0;
      for (int c=0; c<n; c++) {
	val += A[rf_indx(r,c,m,n)]*b[c];            
      }
      Ab[r] = val;
    }
  }
  return;
}

__device__ void Atb_one_mat(// Input
			    const float   *A,  // Row-first
			    const float   *b,
			    unsigned int  m,
			    unsigned int  n,
			    unsigned int  id,
			    unsigned int  ntr,
			    // Output
			    float         *Atb)
{
  unsigned int ept = n/ntr; ept += (n%ntr) ? 1 : 0; // No of elements (of Atb) per thread
  for (int i=0; i<ept; i++) {
    unsigned int c = id + i*ntr;
    if (c < n) {
      float val = 0.0;
      for (int r=0; r<m; r++) {
	val += A[rf_indx(r,c,m,n)]*b[r];            
      }
      Atb[c] = val;
    }
  }
  return;
}

__device__ void Kty_one_mat(// Input
			    const float     *K, // Column-first
			    const float     *y,
			    unsigned int    m,
			    unsigned int    n,
			    unsigned int    id,
			    unsigned int    ntr,
			    // Output
			    float           *Kty)
{
  unsigned int ept = n/ntr; ept += (n%ntr) ? 1 : 0; // No of elements (of Kty) per thread
  for (int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < n) {
      float val = 0.0;
      for (int c=0; c<m; c++) {
	val += K[cf_indx(c,r,m,n)]*y[c]; // r and c switched -> transpose
      }
      Kty[r] = val;
    }
  }
  return;
}

__device__ void Wirty_one_mat(// Input
			      const float   *y,
			      const float   *Wir,  // Row-first
			      unsigned int  zstep,
			      unsigned int  mn,
			      unsigned int  id,
			      unsigned int  ntr,
			      // Output
			      float         *Wirty)
{
  unsigned int ept = mn/ntr; ept += (mn%ntr) ? 1 : 0; // Number of elements of Wir per thread
  for (unsigned int i=0; i<ept; i++) {
    unsigned int r = id + i*ntr;
    if (r < mn) {
      Wirty[r] = 0.0;
      for (unsigned int c=0; c<mn; c++) Wirty[r] += Wir[rf_indx(c,r,mn,mn)] * y[c*zstep]; 
    }
  }
  return;
}

__device__ void Wir_one_mat(// Input
			    const float   *zcoord,
			    unsigned int  zstep,
			    unsigned int  mn,
			    unsigned int  id,
			    unsigned int  ntr,
			    // Output
			    float         *Wir)  // Row-first
{
  unsigned int rpt = mn/ntr; rpt += (mn%ntr) ? 1 : 0;  // Number of rows of Wir per thread
  for (unsigned int i=0; i<rpt; i++) {
    unsigned int r = id + i*ntr;
    if (r < mn) {
      for (unsigned int c=0; c<mn; c++) Wir[rf_indx(r,c,mn,mn)] = 0.0;
      float z = zcoord[r*zstep];
      if (z>=0 && z<=(mn-1)) {
        int iz = static_cast<int>(z);
	for (int c=iz-2; c<iz+3; c++) {
	  Wir[rf_indx(r,min(max(0,c),static_cast<int>(mn)-1),mn,mn)] += wgt_at(z-c);
	}
      } 
    }
  }
  return;
}

__device__ float wgt_at(float x)
{
  float wgt = 0;
  x = (x<0.0) ? -x : x;
  if (x < 1) wgt = 2.0/3.0 + 0.5*x*x*(x-2.0);
  else if (x < 2) wgt = (1.0/6.0) * (2.0-x)*(2.0-x)*(2.0-x);

  return(wgt);
}

} // End namespace EMKI
