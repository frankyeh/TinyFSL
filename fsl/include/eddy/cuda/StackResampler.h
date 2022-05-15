/*! \file StackResampler.h
    \brief Contains declaration of CUDA implementation of a class for spline resampling of irregularly sampled columns in the z-direction

    \author Jesper Andersson
    \version 1.0b, March, 2016.
*/
//
// StackResampler.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "armawrap/newmat.h"
#include "newimage/newimage.h"
#include "EddyHelperClasses.h"
#include "CudaVolume.h"
#pragma pop

namespace EDDY {

class StackResampler
{
public:
  /// Constructor. Performs the actual work so that when the object is created there is already a resampled image ready.

  /// This version of the constructor uses a predicted volume and the laplacian for regularisation
  StackResampler(const EDDY::CudaVolume&  stack,
		 const EDDY::CudaVolume&  zcoord,
		 const EDDY::CudaVolume&  pred,
		 const EDDY::CudaVolume&  mask,
		 double                   lambda=0.005);

  /// This version of the constructor uses either splines and Laplacian regularisation or linear interpolation.
  StackResampler(const EDDY::CudaVolume&  stack,
		 const EDDY::CudaVolume&  zcoord,
		 const EDDY::CudaVolume&  mask,
		 NEWIMAGE::interpolation  interp=NEWIMAGE::spline,
		 double                   lambda=0.005);
  ~StackResampler()  {}
  /// Returns the resampled volue
  const EDDY::CudaVolume& GetResampledIma() const EddyTry { return(_resvol); } EddyCatch
  NEWIMAGE::volume<float> GetResampledImaAsNEWIMAGE() const EddyTry { return(_resvol.GetVolume()); } EddyCatch
  /// Returns binary mask to indicate valid voxels
  const EDDY::CudaVolume& GetMask() const EddyTry { return(_mask); } EddyCatch
private:
  static const int         _threads_per_block_QR = 128;
  static const int         _threads_per_block_Solve = 128;
  static const int         _threads_per_block_Wirty = 128;
  static const int         _threads_per_block_Wir = 128;
  static const int         _threads_per_block_yhat = 128;
  static const int         _threads_per_block_transfer = 128;
  static const dim3        _threads_per_block_WtW_StS;
  EDDY::CudaVolume         _resvol;
  EDDY::CudaVolume         _mask;

  /// Get Laplacian for regularisation. Runs on CPU.
  void get_StS(unsigned int sz, double lambda, thrust::device_vector<float>& StS) const;
  /// Get "design matrix" for splines on regular grid. Runs on CPU.
  void get_regular_W(unsigned int sz, thrust::device_vector<float>& W) const;
  /// Make binary mask with one for valid voxels
  void make_mask(const EDDY::CudaVolume&   inmask,
		 const EDDY::CudaVolume&   zcoord,
		 bool                      zync,
		 EDDY::CudaVolume&         omask);
  /// Take the z-columns in zccord, sort them and put them in z-x-y order in szcoord
  void sort_zcoords(const EDDY::CudaVolume&        zcoord,
		    bool                           sync,
		    thrust::device_vector<float>&  szcoord) const;
  /// Make z-first vectors of weights for the predictions
  void make_weight_vectors(const thrust::device_vector<float>&  zcoord,
			   unsigned int                         xsz,
			   unsigned int                         zsz,
			   unsigned int                         xzp,
			   bool                                 sync,
			   thrust::device_vector<float>&        weights) const;
  /// Insert one xz-plane of weights into volume
  void insert_weights(const thrust::device_vector<float>&  wvec,
		      unsigned int                         j,
		      bool                                 sync,
		      EDDY::CudaVolume&                    wvol) const;
  /// Elementwise multiplication of weights with predictions.
  void make_diagw_p_vectors(const EDDY::CudaVolume&               pred,
			    const thrust::device_vector<float>&   wgts,
			    unsigned int                          xzp,
			    bool                                  sync,
			    thrust::device_vector<float>&         wp) const;
  /// Premultiply W matrix by diag{w}
  void make_diagw_W_matrices(const thrust::device_vector<float>&   w,
			     const thrust::device_vector<float>&   W,
			     unsigned int                          matsz,
			     unsigned int                          nmat,
			     bool                                  sync,
			     thrust::device_vector<float>&         diagwW) const;
  /// Multiply a set of (diag{w}W)' matrices with a bunch of diag{w}p vectors
  void make_dwWt_dwp_vectors(const thrust::device_vector<float>& dW,
			     const thrust::device_vector<float>& dp,
			     unsigned int                        matsz,
			     unsigned int                        nmat,
			     bool                                sync,
			     thrust::device_vector<float>&       dWtdp) const;
  /// Make a set of "design matrices" for irregularly sampled splines. GPU.
  void make_Wir_matrices(const EDDY::CudaVolume&       zcoord,
			 unsigned int                  xzp,
			 bool                          sync,
			 thrust::device_vector<float>& Wir) const;
  /// Multiply a set of Wir matrices with a set of intensity vectors. GPU.
  void make_Wir_t_y_vectors(const EDDY::CudaVolume&                 y,
			    const thrust::device_vector<float>&     Wir,
			    unsigned int                            xzp,
			    bool                                    sync,
			    thrust::device_vector<float>&           Wirty) const;
  /// Multiply a set of Wir matrices by transpose of self and add regularisation matrix. GPU.
  void make_WtW_StS_matrices(const thrust::device_vector<float>&  Wir,
			     unsigned int                         mn,
			     unsigned int                         nmat,
			     const thrust::device_vector<float>&  StS,
			     bool                                 sync,
			     thrust::device_vector<float>&        WtW) const;
  /// Solve for spline coefficients
  void solve_for_c_hat(// Input
		       const thrust::device_vector<float>& WtW,         // WtW+StS matrices for one xz-plane.
		       const thrust::device_vector<float>& Wty,         // Wty vectors for one xz-plane
		       unsigned int                        n,           // Size of KtK (nxn)
		       unsigned int                        nmat,        // Number of matrices for one xz-plane
		       bool                                sync,        // If true syncs after submitting kernel
		       // Output
		       thrust::device_vector<float>&       chat) const; // Returns inv(WtW)*Wty for all matrices in xz-plane
  /// Multiply spline coefficients with regularly sampled spline matrix
  void make_y_hat_vectors(// Input
			  const thrust::device_vector<float>& W,
			  const thrust::device_vector<float>& chat,
			  unsigned int                        mn,
			  unsigned int                        nvec,
			  bool                                sync,
			  // Output
			  thrust::device_vector<float>&       yhat) const;
  /// Transfer y_hat (interpolated data) to volume
  void transfer_y_hat_to_volume(// Input
				const thrust::device_vector<float>& yhat,
				unsigned int                        xzp,
				bool                                sync,
				// Output
				EDDY::CudaVolume&                   ovol) const;
  /// Sorts with zccord as key and reorders data in the same way
  void sort_zcoords_and_intensities(const EDDY::CudaVolume&        zcoord,
				    const EDDY::CudaVolume&        data,
				    bool                           sync,
				    thrust::device_vector<float>&  szcoord,
				    thrust::device_vector<float>&  sdata) const;
  /// Takes sorted vectors of z-coords and values and linearly interpolates onto 0--(zsz-1)
  void linear_interpolate_columns(const thrust::device_vector<float>&  zcoord,
				  const thrust::device_vector<float>&  val,
				  unsigned int                         xsz,
				  unsigned int                         ysz,
				  unsigned int                         zsz,
				  bool                                 sync,
				  thrust::device_vector<float>&        ival) const;
  void transfer_interpolated_columns_to_volume(const thrust::device_vector<float>&  zcols,
					       bool                                 sync,
					       EDDY::CudaVolume&                    vol);
  /// i,j->linear index for row-first square matrix. Matrix addressed M(0:mn-1,0:mn-1)
  unsigned int rfindx(unsigned int i, unsigned int j, unsigned int mn) const { return(i+j*mn); }
  /// i,j->linear index for column-first square matrix. Matrix addressed M(0:mn-1,0:mn-1)
  unsigned int cfindx(unsigned int i, unsigned int j, unsigned int mn) const { return(i*mn+j); }
  /// Local sqr, just to be safe
  template <typename T> T sqr(T a) const { return(a*a); }
  /// Writes a single matrix residing on the GPU as a Newmat text file
  void write_matrix(const thrust::device_vector<float>& mats,
		    unsigned int                        offs,
		    unsigned int                        m,
		    unsigned int                        n,
		    const std::string&                  fname) const;
  /// Writes a block of matrices residing on the GPU as a set of Newmat text files
  void write_matrices(const thrust::device_vector<float>& mats,
		      unsigned int                        nmat,
		      unsigned int                        m,
		      unsigned int                        n,
		      const std::string&                  basefname) const;
  /// Writes all neccessary info for one column for debugging
  void write_debug_info_for_pred_resampling(unsigned int                         x,
					    unsigned int                         y,
					    const std::string&                   bfname,
					    const EDDY::CudaVolume&              z,
					    const EDDY::CudaVolume&              g,
					    const EDDY::CudaVolume&              p,
					    const thrust::device_vector<float>&  sz,
					    const thrust::device_vector<float>&  W,
					    const thrust::device_vector<float>&  Wir,
					    const thrust::device_vector<float>&  w,
					    const thrust::device_vector<float>&  wp,
					    const thrust::device_vector<float>&  wW,
					    const thrust::device_vector<float>&  Wirtg,
					    const thrust::device_vector<float>&  wWtwp,
					    const thrust::device_vector<float>&  WirtWir,
					    const thrust::device_vector<float>&  wWtwW,
					    const thrust::device_vector<float>&  sum_vec,
					    const thrust::device_vector<float>&  sum_mat,
					    const thrust::device_vector<float>&  c_hat,
					    const thrust::device_vector<float>&  y_hat) const;

};

} // End namespace EDDY
