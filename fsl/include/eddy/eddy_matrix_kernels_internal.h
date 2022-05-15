/*! \file eddy_matrix_kernels_internal.h
    \brief Contains declarations of __device__ kernels used internally for LS resampling in Eddy project

    The kernels for CUDA based Matrix manipulation have been divided into __global__
    ones that have an API from any .c or .cpp file and __device__ ones that are only
    callable from within other kernels. The former category are exposed to the application
    programmer through EddyMatrixKernels.h and placed in the EddyMatrixKernels namebase.
    The latter are "hidden" by virtue of being declared in this .h file which is _only_
    intended to be included by EddyMatrixKernels.cu and by residing in the EMKI namespace.

    \author Jesper Andersson
    \version 1.0b, July, 2013
*/

#ifndef eddy_matrix_kernels_internal_h
#define eddy_matrix_kernels_internal_h

#include <cuda.h>

namespace EMKI {

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
			  float       *R);      // R in K = Q'*R

__device__ void M_eq_M(float       *dM, // Destination matrix
                       const float *sM, // Source matrix
		       unsigned int m,   // No. of rows
		       unsigned int n,   // No. of columns
		       unsigned int nt,  // No. of threads for this matrix
		       unsigned int id); // Number of this thread (among nt)

__device__ void set_to_identity(float        *M,   // Matrix to set
                                unsigned int  m,   // Size of matrix
				unsigned int  nt,  // Number of threads for this matrix
				unsigned int  id); // Number of this thread (among nt)

__device__ float get_alfa(const float  *M,   // Matrix
			  unsigned int m,    // No of rows
			  unsigned int n,    // No of columns
			  unsigned int j,    // Column to get alfa for
			  unsigned int nt,   // No of threads for this matrix
			  unsigned int id,   // Number of thread (among nt)
			  float        *scr);// Scratch space for reduction

__device__ void get_v(const float  *M,  // The matrix (full size)
		      unsigned int m,   // No. of rows of M
		      unsigned int n,   // No. of columns of M
		      unsigned int j,   // Column to extract v for
		      float        alfa,// Alfa
		      unsigned int nt,  // No. of threads for this matrix
		      unsigned int id,  // Thread number among nt
		      float        *v); // Space for v

__device__ void two_x_vt_x_R(const float  *R,      // R (first j columns upper diag)
			     unsigned int m,       // No. of rows of R
			     unsigned int n,       // No. of rows of R
			     unsigned int j,       // The column we currently work on
                             unsigned int nt,      // No. of threads for this matrix
                             unsigned int id,      // Thread number (out of nt)
			     const float  *v,      // v (first j elements zero)
			     float        *twovtR);// 2.0*v'*R, first j elements zero

__device__ void R_minus_v_x_wt(unsigned int m,  // No. of rows of R
			       unsigned int n,  // No. of columns of R
			       unsigned int j,  // The column we are currently on
			       unsigned int nt, // No. of threads for this matrix
			       unsigned int id, // Number of this thread (within nt)
			       const float  *v, // v, mx1, j first elements zero
			       const float  *w, // w, nx1, j first elements zero
			       float        *R);// R in, R - vw' out 

__device__ void two_x_vt_x_Qt(const float  *Qt,      // Q'
			      unsigned int m,        // No. of rows of Qt
			      unsigned int n,        // No. of columns of Qt
			      unsigned int j,        // The column we currently work on
			      unsigned int nt,       // No. of threads for this matrix
			      unsigned int id,       // Thread number (out of nt)
			      const float  *v,       // v (first j elements zero)
			      float        *twovtQt);// 2.0*v'*Q'

__device__ void Qt_minus_v_x_wt(unsigned int m,   // No. of rows of Q'
			        unsigned int n,   // No. of columns of Q'
			        unsigned int j,   // The column we are currently on
			        unsigned int nt,  // No. of threads for this matrix
			        unsigned int id,  // Number of this thread (within nt)
			        const float  *v,  // v, mx1, j first elements zero
			        const float  *w,  // w, nx1
			        float        *Qt);// Q' in, Q' - vw' out 

__device__ bool is_pow_of_two(unsigned int n);
__device__ unsigned int next_pow_of_two(unsigned int n);
__device__ unsigned int rf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n);
__device__ unsigned int cf_indx(unsigned int i, unsigned int j, unsigned int m, unsigned int n);
template <typename T>
__device__ T sqr(const T& v) { return(v*v); }
template <typename T>
__device__ T min(const T& p1, const T& p2) { return((p1<p2) ? p1 : p2); }
template <typename T>
__device__ T max(const T& p1, const T& p2) { return((p1<p2) ? p2 : p1); }
__device__ float wgt_at(float x);



__device__ void solve_single(// Input
			     const float *Qt,     // Row-first orthogonal mxm matrix  
			     const float *R,      // Row-first upper-diagonal mxn matrix
			     const float *y,      // mx1 vector
			     unsigned int m,
			     unsigned int n,
			     // Output
			     float       *y_hat); // nx1 solution vector

__device__ void back_substitute(const float  *R,
				unsigned int  m,
				unsigned int  n,
				float        *v);

__device__ void M_times_v(const float *M,
			  const float *v,
			  unsigned int m,
			  unsigned int n,
			  float       *Mv);

__device__ void cf_KtK_one_mat(const float  *K,   // Column-first mxn
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, // Column-first nxn
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK);// Row-first nxn

__device__ void rf_KtK_one_mat(const float  *K,   // Row-first mxn
			       unsigned int m,
			       unsigned int n,
			       const float  *StS, // Row-first nxn
			       float        lambda,
			       unsigned int idr,
			       unsigned int idc,
			       unsigned int ntr,
			       unsigned int ntc,
			       float        *KtK);// Row-first nxn

__device__ void Ab_one_mat(// Input
			   const float   *A,  // Row-first
			   const float   *b,
			   unsigned int  m,
			   unsigned int  n,
			   unsigned int  id,
			   unsigned int  ntr,
			   // Output
			   float         *Ab);

__device__ void Atb_one_mat(// Input
			    const float   *A,  // Row-first
			    const float   *b,
			    unsigned int  m,
			    unsigned int  n,
			    unsigned int  id,
			    unsigned int  ntr,
			    // Output
			    float         *Atb);

__device__ void Kty_one_mat(// Input
			    const float     *K,
			    const float     *y,
			    unsigned int    m,
			    unsigned int    n,
			    unsigned int    id,
			    unsigned int    ntr,
			    // Output
			    float           *Kty);

__device__ void Wir_one_mat(// Input
			    const float   *zcoord,
			    unsigned int  zstep,
			    unsigned int  mn,
			    unsigned int  id,
			    unsigned int  ntr,
			    // Output
			    float         *Wir);

__device__ void Wirty_one_mat(// Input
			      const float   *y,
			      const float   *Wir,  // Row-first
			      unsigned int  zstep,
			      unsigned int  mn,
			      unsigned int  id,
			      unsigned int  ntr,
			      // Output
			      float         *Wirty);

} // End namespace EMKI

#endif // End #ifndef eddy_matrix_kernels_internal_h
