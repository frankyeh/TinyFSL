/*! \file StackResampler.cu
    \brief Contains definition of CUDA implementation of a class for spline resampling of irregularly sampled columns in the z-direction

    \author Jesper Andersson
    \version 1.0b, March, 2016.
*/
//
// StackResampler.cu
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford
//

// Because of a bug in cuda_fp16.hpp, that gets included by cublas_v2.h, it has to
// be included before any include files that set up anything related to the std-lib.
// If not, there will be an ambiguity in cuda_fp16.hpp about wether to use the
// old-style C isinf or the new (since C++11) std::isinf.
#include "cublas_v2.h"

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
#include "EddyCudaHelperFunctions.h"
#include "StackResampler.h"
#include "EddyKernels.h"
#include "EddyMatrixKernels.h"
#include "EddyFunctors.h"

namespace EDDY {

const dim3 StackResampler::_threads_per_block_WtW_StS = dim3(16,16);

StackResampler::StackResampler(const EDDY::CudaVolume&  stack,
			       const EDDY::CudaVolume&  zcoord,
			       const EDDY::CudaVolume&  pred,
			       const EDDY::CudaVolume&  mask,
			       double                   lambda) EddyTry : _resvol(stack,false), _mask(stack,false)
{
  // static unsigned int cnt = 0;
  unsigned int matsz = stack.Size(2);
  unsigned int nmat = stack.Size(0);    // Number of matrices per xz-plane
  // Allocate memory for storing matrices/vectors on the GPU
  thrust::device_vector<float> StS_matrix(sqr(matsz));
  thrust::device_vector<float> empty_StS_matrix(sqr(matsz));
  thrust::device_vector<float> gpu_W_matrix(sqr(matsz));
  thrust::device_vector<float> gpu_sorted_zcoord(zcoord.Size());
  thrust::device_vector<float> gpu_Wir_matrices(nmat*sqr(matsz));
  thrust::device_vector<float> gpu_weights(nmat*matsz);
  thrust::device_vector<float> gpu_diagw_p_vectors(nmat*matsz);
  thrust::device_vector<float> gpu_diagw_W_matrices(nmat*sqr(matsz));
  thrust::device_vector<float> gpu_Wirty_vectors(nmat*matsz);
  thrust::device_vector<float> gpu_dWtdp_vectors(nmat*matsz);
  thrust::device_vector<float> gpu_WtW_matrices(nmat*sqr(matsz));
  thrust::device_vector<float> gpu_dWtdW_matrices(nmat*sqr(matsz));
  thrust::device_vector<float> gpu_sum_vectors(nmat*matsz);
  thrust::device_vector<float> gpu_sum_matrices(nmat*sqr(matsz));
  thrust::device_vector<float> gpu_c_hat_vectors(nmat*matsz);
  thrust::device_vector<float> gpu_y_hat_vectors(nmat*matsz);

  // EDDY::CudaVolume weights = pred;
  // weights = 0.0;

  // Get regularisation matrix once and for all
  //cout << "Making reg matrix" << endl; cout.flush();
  get_StS(matsz,lambda,StS_matrix);
  get_StS(matsz,0.0,empty_StS_matrix);
  // Get weight matrix for regular grid once and for all
  //cout << "Making W" << endl; cout.flush();
  get_regular_W(matsz,gpu_W_matrix);
  // Populate mask
  //cout << "Making mask" << endl; cout.flush();
  make_mask(mask,zcoord,true,_mask);
  // Get a volume of sorted z-ccords
  //cout << "sorting zcoords" << endl; cout.flush();
  sort_zcoords(zcoord,true,gpu_sorted_zcoord);
  // Interpolate along z-columns
  for (unsigned int j=0; j<stack.Size(1); j++) { // Loop over all xz-planes
    // Make all Wir matrices for this xz-plane
    //cout << "Making Wir" << endl; cout.flush();
    make_Wir_matrices(zcoord,j,true,gpu_Wir_matrices);
    // Make vectors of weights for predictions
    //cout << "Making weights" << endl; cout.flush();
    make_weight_vectors(gpu_sorted_zcoord,stack.Size(0),stack.Size(2),j,true,gpu_weights);
    // insert_weights(gpu_weights,j,true,weights);
    // Make diag{w}p vectors, where p are the predictions
    //cout << "Making diag{w}p" << endl; cout.flush();
    make_diagw_p_vectors(pred,gpu_weights,j,true,gpu_diagw_p_vectors);
    // Make diag{w}W matrices, where W is the spline design matrix for regular sampling
    //cout << "Making diag{w}W" << endl; cout.flush();
    make_diagw_W_matrices(gpu_weights,gpu_W_matrix,matsz,nmat,true,gpu_diagw_W_matrices);
    // Calculate all Wir'*y vectors for this plane
    //cout << "Making Wirty" << endl; cout.flush();
    make_Wir_t_y_vectors(stack,gpu_Wir_matrices,j,true,gpu_Wirty_vectors);
    // Calculate all (diag{w}W)'*diag{w}p vectors for this plane
    //cout << "Making diag{w}W'diag{w}p" << endl; cout.flush();
    make_dwWt_dwp_vectors(gpu_diagw_W_matrices,gpu_diagw_p_vectors,matsz,nmat,true,gpu_dWtdp_vectors);
    // Calculate all Wir'*Wir matrices
    //cout << "Making Wir'Wir" << endl; cout.flush();
    make_WtW_StS_matrices(gpu_Wir_matrices,matsz,nmat,StS_matrix,true,gpu_WtW_matrices);
    // Calculate all (diag{w}W)'*(diag{w}W) matrices
    //cout << "Making diag{w}W'diag{w}W" << endl; cout.flush();
    make_WtW_StS_matrices(gpu_diagw_W_matrices,matsz,nmat,empty_StS_matrix,true,gpu_dWtdW_matrices);
    // Add (pairwise) all Wir'*y and (diag{w}W)'*diag{w}p vectors
    //cout << "Making sum vectors" << endl; cout.flush();
    thrust::transform(gpu_Wirty_vectors.begin(),gpu_Wirty_vectors.end(),gpu_dWtdp_vectors.begin(),gpu_sum_vectors.begin(),thrust::plus<float>());
    // Add (pairwise) all Wir'*Wir and (diag{w}W)'*(diag{w}W)
    //cout << "Making sum matrices" << endl; cout.flush();
    thrust::transform(gpu_WtW_matrices.begin(),gpu_WtW_matrices.end(),gpu_dWtdW_matrices.begin(),gpu_sum_matrices.begin(),thrust::plus<float>());
    // Solve for spline coefficients
    //cout << "Making c_hat" << endl; cout.flush();
    solve_for_c_hat(gpu_sum_matrices,gpu_sum_vectors,matsz,nmat,true,gpu_c_hat_vectors);
    // Multiply by W to solve for y_hat (interpolate in z-direction)
    //cout << "Making y_hat" << endl; cout.flush();
    make_y_hat_vectors(gpu_W_matrix,gpu_c_hat_vectors,matsz,nmat,true,gpu_y_hat_vectors);
    // Transfer solution vectors to result volume
    //cout << "Transferring y_hat" << endl; cout.flush();
    transfer_y_hat_to_volume(gpu_y_hat_vectors,j,true,_resvol);
    /*
    if (cnt==19 && j==30) {
      write_debug_info_for_pred_resampling(54,j,"debug_info.txt",zcoord,stack,pred,gpu_sorted_zcoord,
					   gpu_W_matrix,gpu_Wir_matrices,gpu_weights,gpu_diagw_p_vectors,
					   gpu_diagw_W_matrices,gpu_Wirty_vectors,gpu_dWtdp_vectors,
					   gpu_WtW_matrices,gpu_dWtdW_matrices,gpu_sum_vectors,
					   gpu_sum_matrices,gpu_c_hat_vectors,gpu_y_hat_vectors);
    }
    */
  }
  // char fname[256]; sprintf(fname,"weights_%02d",cnt);
  // weights.Write(std::string(fname));
  // cnt++;
} EddyCatch

StackResampler::StackResampler(const EDDY::CudaVolume&  stack,
			       const EDDY::CudaVolume&  zcoord,
			       const EDDY::CudaVolume&  mask,
			       NEWIMAGE::interpolation  interp,
			       double                   lambda) EddyTry : _resvol(stack,false), _mask(stack,false)
{
  // Populate mask
  make_mask(mask,zcoord,true,_mask);
  if (interp == NEWIMAGE::spline) {
    unsigned int matsz = stack.Size(2);
    unsigned int nmat = stack.Size(0);    // Number of matrices per xz-plane
    // Allocate memory for storing matrices/vectors on the GPU
    thrust::device_vector<float> gpu_StS_matrix(sqr(matsz));
    thrust::device_vector<float> gpu_W_matrix(sqr(matsz));
    thrust::device_vector<float> gpu_Wir_matrices(nmat*sqr(matsz));
    thrust::device_vector<float> gpu_WtW_StS_matrices(nmat*sqr(matsz));
    thrust::device_vector<float> gpu_Wirty_vectors(nmat*matsz);
    thrust::device_vector<float> gpu_c_hat_vectors(nmat*matsz);
    thrust::device_vector<float> gpu_y_hat_vectors(nmat*matsz);
    // Get regularisation matrix once and for all
    get_StS(matsz,lambda,gpu_StS_matrix);
    // Get spline matrix for regular grid once and for all
    get_regular_W(matsz,gpu_W_matrix);
    // Interpolate along z-columns
    for (unsigned int j=0; j<stack.Size(1); j++) { // Loop over all xz-planes
      // Make all Wir matrices for this xz-plane
      make_Wir_matrices(zcoord,j,true,gpu_Wir_matrices);
      // Pre-multiply Wir by transpose of self and add lambda*StS.
      make_WtW_StS_matrices(gpu_Wir_matrices,matsz,nmat,gpu_StS_matrix,true,gpu_WtW_StS_matrices);
      // Multiply Wir transposed with y (observed intensities).
      make_Wir_t_y_vectors(stack,gpu_Wir_matrices,j,true,gpu_Wirty_vectors);
      // Solve for spline coefficients
      solve_for_c_hat(gpu_WtW_StS_matrices,gpu_Wirty_vectors,matsz,nmat,true,gpu_c_hat_vectors);
      // Multiply by W to solve for y_hat (interpolate in z-direction)
      make_y_hat_vectors(gpu_W_matrix,gpu_c_hat_vectors,matsz,nmat,true,gpu_y_hat_vectors);
      // Transfer solution vectors to result volume
      transfer_y_hat_to_volume(gpu_y_hat_vectors,j,true,_resvol);
    }
  }
  else if (interp == NEWIMAGE::trilinear) {
    thrust::device_vector<float> gpu_sorted_zcoord(zcoord.Size());
    thrust::device_vector<float> gpu_sorted_intensities(zcoord.Size());
    thrust::device_vector<float> gpu_interpolated_columns(zcoord.Size(),0.0);
    // Sort the z-ccordinates and the intensities with z-coord as key
    sort_zcoords_and_intensities(zcoord,stack,true,gpu_sorted_zcoord,gpu_sorted_intensities);
    // Do linear interpolation along z-columns
    linear_interpolate_columns(gpu_sorted_zcoord,gpu_sorted_intensities,zcoord.Size(0),zcoord.Size(1),zcoord.Size(2),true,gpu_interpolated_columns);
    // Transfer interpolated vectors to results volume
    transfer_interpolated_columns_to_volume(gpu_interpolated_columns,true,_resvol);
  }
  else throw EddyException("StackResampler::StackResampler: Invalid interpolation method");
} EddyCatch

void StackResampler::get_StS(unsigned int sz, double lambda, thrust::device_vector<float>& StS) const EddyTry
{
  float six = lambda * 6.0; float minusfour = - lambda * 4.0; float one = lambda;
  thrust::host_vector<float> hStS(StS.size(),0.0);
  hStS[rfindx(0,0,sz)] = six; hStS[rfindx(0,1,sz)] = minusfour; hStS[rfindx(0,2,sz)] = one; hStS[rfindx(0,(sz-2),sz)] = one; hStS[rfindx(0,(sz-1),sz)] = minusfour;
  hStS[rfindx(1,0,sz)] = minusfour; hStS[rfindx(1,1,sz)] = six; hStS[rfindx(1,2,sz)] = minusfour; hStS[rfindx(1,3,sz)] = one; hStS[rfindx(1,(sz-1),sz)] = one;
  for ( unsigned int i=2; i<(sz-2); i++) {
    hStS[rfindx(i,i-2,sz)] = one;
    hStS[rfindx(i,i-1,sz)] = minusfour;
    hStS[rfindx(i,i,sz)] = six;
    hStS[rfindx(i,i+1,sz)] = minusfour;
    hStS[rfindx(i,i+2,sz)] = one;
  }
  hStS[rfindx((sz-2),0,sz)] = one; hStS[rfindx((sz-2),(sz-4),sz)] = one; hStS[rfindx((sz-2),(sz-3),sz)] = minusfour; hStS[rfindx((sz-2),(sz-2),sz)] = six; hStS[rfindx((sz-2),(sz-1),sz)] = minusfour;
  hStS[rfindx((sz-1),0,sz)] = minusfour; hStS[rfindx((sz-1),1,sz)] = one; hStS[rfindx((sz-1),(sz-3),sz)] = one; hStS[rfindx((sz-1),(sz-2),sz)] = minusfour; hStS[rfindx((sz-1),(sz-1),sz)] = six;
  StS = hStS;
  return;
} EddyCatch

void StackResampler::get_regular_W(unsigned int sz, thrust::device_vector<float>& W) const EddyTry
{
  thrust::host_vector<float> hW(W.size(),0.0);
  hW[rfindx(0,0,sz)] = 5.0/6.0; hW[rfindx(0,1,sz)] = 1.0/6.0;
  for ( unsigned int i=1; i<(sz-1); i++) {
    hW[rfindx(i,i-1,sz)] = 1.0/6.0;
    hW[rfindx(i,i,sz)] = 4.0/6.0;
    hW[rfindx(i,i+1,sz)] = 1.0/6.0;
  }
  hW[rfindx((sz-1),(sz-2),sz)] = 1.0/6.0; hW[rfindx((sz-1),(sz-1),sz)] = 5.0/6.0;
  W = hW;
  return;
} EddyCatch

void StackResampler::make_mask(const EDDY::CudaVolume&   inmask,
			       const EDDY::CudaVolume&   zcoord,
			       bool                      sync,
			       EDDY::CudaVolume&         omask) EddyTry
{
  omask = 0.0;
  int nblocks = inmask.Size(1);
  int tpb = inmask.Size(0);
  EddyKernels::make_mask_from_stack<<<nblocks,tpb>>>(inmask.GetPtr(),zcoord.GetPtr(),inmask.Size(0),
						     inmask.Size(1),inmask.Size(2),omask.GetPtr());
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::make_mask_from_stack");
} EddyCatch

void StackResampler::sort_zcoords(const EDDY::CudaVolume&        zcoord,
				  bool                           sync,
				  thrust::device_vector<float>&  szcoord) const EddyTry
{
  // First transfer all columns and check if they are sorted
  thrust::device_vector<unsigned int> ns_flags(zcoord.Size(0)*zcoord.Size(1),0); // Non-zero if a column is "not sorted"
  EddyKernels::TransferAndCheckSorting<<<zcoord.Size(1),zcoord.Size(0)>>>(zcoord.GetPtr(),zcoord.Size(0),
									  zcoord.Size(1),zcoord.Size(2),
									  thrust::raw_pointer_cast(szcoord.data()),
									  thrust::raw_pointer_cast(ns_flags.data()));
  EddyCudaHelperFunctions::CudaSync("EddyKernels::TransferAndCheckSorting");
  unsigned int nnsort = thrust::reduce(ns_flags.begin(),ns_flags.end(),0,EDDY::Sum<unsigned int,unsigned int>());
  if (nnsort) { // If there are columns that are not sorted
    thrust::host_vector<unsigned int> host_ns_flags = ns_flags;
    thrust::host_vector<unsigned int> host_nsort_indx(nnsort,0);
    for (unsigned int i=0, n=0; i<zcoord.Size(0)*zcoord.Size(1); i++) {
      if (host_ns_flags[i]) { host_nsort_indx[n] = i; n++; }
    }
    thrust::device_vector<unsigned int> device_nsort_indx = host_nsort_indx;
    int nb = (nnsort / 32) + 1;
    EddyKernels::SortVectors<<<nb,32>>>(thrust::raw_pointer_cast(device_nsort_indx.data()),nnsort,
					zcoord.Size(2),thrust::raw_pointer_cast(szcoord.data()),NULL);
    if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::SortVectors");
    /*
    // Check that vectors really are sorted. Only for use during testing.
    thrust::host_vector<float> hvec = szcoord;
    for (unsigned int vi=0; vi<zcoord.Size(1)*zcoord.Size(0); vi++) {
      unsigned int offs = vi*zcoord.Size(2);
      for (unsigned int i=1; i<zcoord.Size(2); i++) {
	if (hvec[offs+i] < hvec[offs+i-1]) {
	  cout << "Vector " << vi << " was not sucessfully sorted" << endl;
	  cout.flush();
	  for (unsigned int j=0; j<zcoord.Size(2); j++) {
	    cout << "hvec[" << j << "] = " << hvec[offs+j] << endl;
	  }
	  exit(0);
	}
      }
    }
    */
  }
} EddyCatch

void StackResampler::sort_zcoords_and_intensities(const EDDY::CudaVolume&        zcoord,
						  const EDDY::CudaVolume&        data,
						  bool                           sync,
						  thrust::device_vector<float>&  szcoord,
						  thrust::device_vector<float>&  sdata) const EddyTry
{
  // First transfer all zcoord columns and check if they are sorted
  thrust::device_vector<unsigned int> ns_flags(zcoord.Size(0)*zcoord.Size(1),0); // Non-zero if a column is "not sorted"
  EddyKernels::TransferAndCheckSorting<<<zcoord.Size(1),zcoord.Size(0)>>>(zcoord.GetPtr(),zcoord.Size(0),
									  zcoord.Size(1),zcoord.Size(2),
									  thrust::raw_pointer_cast(szcoord.data()),
									  thrust::raw_pointer_cast(ns_flags.data()));
  EddyCudaHelperFunctions::CudaSync("EddyKernels::TransferAndCheckSorting");
  unsigned int nnsort = thrust::reduce(ns_flags.begin(),ns_flags.end(),0,EDDY::Sum<unsigned int,unsigned int>());
  // Transfer all intensity values as well
  EddyKernels::TransferVolumeToVectors<<<data.Size(1),data.Size(0)>>>(data.GetPtr(),data.Size(0),
								      data.Size(1),data.Size(2),
								      thrust::raw_pointer_cast(sdata.data()));
  if (nnsort) { // If there are columns that are not sorted
    thrust::host_vector<unsigned int> host_ns_flags = ns_flags;
    thrust::host_vector<unsigned int> host_nsort_indx(nnsort,0);
    for (unsigned int i=0, n=0; i<zcoord.Size(0)*zcoord.Size(1); i++) {
      if (host_ns_flags[i]) { host_nsort_indx[n] = i; n++; }
    }
    thrust::device_vector<unsigned int> device_nsort_indx = host_nsort_indx;
    int nb = (nnsort / 32) + 1;
    EddyKernels::SortVectors<<<nb,32>>>(thrust::raw_pointer_cast(device_nsort_indx.data()),nnsort,
					zcoord.Size(2),thrust::raw_pointer_cast(szcoord.data()),
					thrust::raw_pointer_cast(sdata.data()));
    if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::SortVectors");
  }
} EddyCatch

void StackResampler::linear_interpolate_columns(const thrust::device_vector<float>&  zcoord,
						const thrust::device_vector<float>&  val,
						unsigned int                         xsz,
						unsigned int                         ysz,
						unsigned int                         zsz,
						bool                                 sync,
						thrust::device_vector<float>&        ival) const EddyTry
{
  EddyKernels::LinearInterpolate<<<ysz,xsz>>>(thrust::raw_pointer_cast(zcoord.data()),
					      thrust::raw_pointer_cast(val.data()),zsz,
					      thrust::raw_pointer_cast(ival.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::LinearInterpolate");
} EddyCatch

void StackResampler::transfer_interpolated_columns_to_volume(const thrust::device_vector<float>&  zcols,
							     bool                                 sync,
							     EDDY::CudaVolume&                    vol) EddyTry
{
  EddyKernels::TransferColumnsToVolume<<<vol.Size(1),vol.Size(0)>>>(thrust::raw_pointer_cast(zcols.data()),
									 vol.Size(0),vol.Size(1),
									 vol.Size(2),vol.GetPtr());
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::TransferColumnsToVolume");
} EddyCatch

void StackResampler::make_weight_vectors(const thrust::device_vector<float>&  zcoord,
					 unsigned int                         xsz,
					 unsigned int                         zsz,
					 unsigned int                         xzp,
					 bool                                 sync,
					 thrust::device_vector<float>&        weights) const EddyTry
{
  EddyKernels::MakeWeights<<<xsz,zsz>>>(thrust::raw_pointer_cast(zcoord.data()),xsz,
					zsz,xzp,thrust::raw_pointer_cast(weights.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::MakeWeights");
} EddyCatch

void StackResampler::insert_weights(const thrust::device_vector<float>&  wvec,
				    unsigned int                         j,
				    bool                                 sync,
				    EDDY::CudaVolume&                    wvol) const EddyTry
{
  EddyKernels::InsertWeights<<<wvol.Size(0),wvol.Size(2)>>>(thrust::raw_pointer_cast(wvec.data()),j,wvol.Size(0),
							    wvol.Size(1),wvol.Size(2),wvol.GetPtr());
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::InsertWeights");
} EddyCatch

void StackResampler::make_diagw_p_vectors(const EDDY::CudaVolume&               pred,
					  const thrust::device_vector<float>&   wgts,
					  unsigned int                          xzp,
					  bool                                  sync,
					  thrust::device_vector<float>&         wp) const EddyTry
{
  EddyKernels::MakeDiagwpVecs<<<pred.Size(0),pred.Size(2)>>>(pred.GetPtr(),thrust::raw_pointer_cast(wgts.data()),
							     pred.Size(0),pred.Size(1),pred.Size(2),xzp,
							     thrust::raw_pointer_cast(wp.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyKernels::MakeDiagwpVecs");
} EddyCatch

void StackResampler::make_diagw_W_matrices(const thrust::device_vector<float>&   wgts,
					   const thrust::device_vector<float>&   W,
					   unsigned int                          matsz,
					   unsigned int                          nmat,
					   bool                                  sync,
					   thrust::device_vector<float>&         diagwW) const EddyTry
{
  EddyMatrixKernels::DiagwA<<<nmat,matsz>>>(thrust::raw_pointer_cast(wgts.data()),
					    thrust::raw_pointer_cast(W.data()),
					    matsz,matsz,nmat,
					    thrust::raw_pointer_cast(diagwW.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::DiagwA");
} EddyCatch

void StackResampler::make_dwWt_dwp_vectors(const thrust::device_vector<float>& dW,
					   const thrust::device_vector<float>& dp,
					   unsigned int                        matsz,
					   unsigned int                        nmat,
					   bool                                sync,
					   thrust::device_vector<float>&       dWtdp) const EddyTry
{
  EddyMatrixKernels::Atb<<<nmat,matsz>>>(thrust::raw_pointer_cast(dW.data()),
					 thrust::raw_pointer_cast(dp.data()),
					 matsz,matsz,nmat,nmat,
					 thrust::raw_pointer_cast(dWtdp.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::Atb");
  return;
} EddyCatch

void StackResampler::make_Wir_matrices(const EDDY::CudaVolume&       zcoord,
				       unsigned int                  xzp,
				       bool                          sync,
				       thrust::device_vector<float>& Wir) const EddyTry
{
  int tpb = _threads_per_block_Wir;
  EddyMatrixKernels::Wir<<<zcoord.Size(0),tpb>>>(zcoord.GetPtr(),zcoord.Size(0),zcoord.Size(1),
						 zcoord.Size(2),zcoord.Size(0),xzp,
						 thrust::raw_pointer_cast(Wir.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::Wir");
} EddyCatch

void StackResampler::make_Wir_t_y_vectors(const EDDY::CudaVolume&                 y,
					  const thrust::device_vector<float>&     Wir,
					  unsigned int                            xzp,
					  bool                                    sync,
					  thrust::device_vector<float>&           Wirty) const EddyTry
{
  int tpb = _threads_per_block_Wirty;
  EddyMatrixKernels::Wirty<<<y.Size(0),tpb>>>(y.GetPtr(),thrust::raw_pointer_cast(Wir.data()),
					      y.Size(0),y.Size(1),y.Size(2),y.Size(0),xzp,
					      thrust::raw_pointer_cast(Wirty.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::Wirty");
} EddyCatch

void StackResampler::make_WtW_StS_matrices(const thrust::device_vector<float>&  Wir,
					   unsigned int                         mn,
					   unsigned int                         nmat,
					   const thrust::device_vector<float>&  StS,
					   bool                                 sync,
					   thrust::device_vector<float>&        WtW) const EddyTry
{
  dim3 block = _threads_per_block_WtW_StS;
  EddyMatrixKernels::KtK<<<nmat,block>>>(thrust::raw_pointer_cast(Wir.data()),mn,mn,nmat,
					 thrust::raw_pointer_cast(StS.data()),1.0,true,
					 thrust::raw_pointer_cast(WtW.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("KtK_Kernels::KtK");
  return;
} EddyCatch

void StackResampler::solve_for_c_hat(// Input
				     const thrust::device_vector<float>& WtW,        // WtW+StS matrices for one xz-plane.
				     const thrust::device_vector<float>& Wty,        // Wty vectors for one xz-plane
				     unsigned int                        n,          // Size of KtK (nxn)
				     unsigned int                        nmat,       // Number of matrices for one xz-plane
				     bool                                sync,       // If true syncs after submitting kernel
				     // Output
				     thrust::device_vector<float>&       chat) const EddyTry // Returns inv(WtW)*Wty for all matrices in xz-plane
{
  // Allocate memory for Q and R matrices for QR decomposition
  thrust::device_vector<float> Qt(nmat*n*n);
  thrust::device_vector<float> R(nmat*n*n);
  // Dynamically allocated shared memory, per block (matrix)
  size_t sh_mem_sz = 2*n*sizeof(float);
  int tpb = _threads_per_block_QR;
  EddyMatrixKernels::QR<<<nmat,tpb,sh_mem_sz>>>(thrust::raw_pointer_cast(WtW.data()),n,n,nmat,
						thrust::raw_pointer_cast(Qt.data()),
						thrust::raw_pointer_cast(R.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("QR_Kernels::QR");
  tpb = _threads_per_block_Solve;
  EddyMatrixKernels::Solve<<<nmat,tpb>>>(thrust::raw_pointer_cast(Qt.data()),
					 thrust::raw_pointer_cast(R.data()),
					 thrust::raw_pointer_cast(Wty.data()),n,n,nmat,
					 thrust::raw_pointer_cast(chat.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("QR_Kernels::Solve");
  return;
} EddyCatch

void StackResampler::make_y_hat_vectors(// Input
					const thrust::device_vector<float>& W,
					const thrust::device_vector<float>& chat,
					unsigned int                        mn,
					unsigned int                        nvec,
					bool                                sync,
					// Output
					thrust::device_vector<float>&       yhat) const EddyTry
{
  int tpb = _threads_per_block_yhat;
  EddyMatrixKernels::Ab<<<nvec,tpb>>>(thrust::raw_pointer_cast(W.data()),
				      thrust::raw_pointer_cast(chat.data()),
				      mn,mn,1,nvec,thrust::raw_pointer_cast(yhat.data()));
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::Ab");
  return;
} EddyCatch

void StackResampler::transfer_y_hat_to_volume(// Input
					      const thrust::device_vector<float>& yhat,
					      unsigned int                        xzp,
					      bool                                sync,
					      // Output
					      EDDY::CudaVolume&                   ovol) const EddyTry
{
  int tpb = _threads_per_block_transfer;
  int nblocks = (ovol.Size(0)%tpb) ? ovol.Size(0) / tpb + 1 : ovol.Size(0) / tpb;
  EddyKernels::transfer_y_hat_to_volume<<<nblocks,tpb>>>(thrust::raw_pointer_cast(yhat.data()),ovol.Size(0),
							 ovol.Size(1),ovol.Size(2),xzp,ovol.GetPtr());
  if (sync) EddyCudaHelperFunctions::CudaSync("EddyMatrixKernels::transfer_y_hat_to_volume");
  return;
} EddyCatch

void StackResampler::write_matrix(const thrust::device_vector<float>& mats,
				  unsigned int                        offs,
				  unsigned int                        m,
				  unsigned int                        n,
				  const std::string&                  fname) const EddyTry
{
  thrust::device_vector<float>::const_iterator first = mats.begin() + offs;
  thrust::device_vector<float>::const_iterator last = mats.begin() + offs + m*n;
  thrust::host_vector<float> mat(first,last);
  NEWMAT::Matrix newmat(m,n);
  for (unsigned int i=0; i<m; i++) {
    for (unsigned int j=0; j<n; j++) {
      newmat(i+1,j+1) = mat[rfindx(i,j,m)];
    }
  }
  MISCMATHS::write_ascii_matrix(fname+std::string(".txt"),newmat);
} EddyCatch

void StackResampler::write_matrices(const thrust::device_vector<float>& mats,
				    unsigned int                        nmat,
				    unsigned int                        m,
				    unsigned int                        n,
				    const std::string&                  basefname) const EddyTry
{
  char fname[256];
  for (unsigned int f=0; f<nmat; f++) {
    sprintf(fname,"%s_%03d",basefname.c_str(),f);
    write_matrix(mats,f*m*n,m,n,std::string(fname));
  }
} EddyCatch

void StackResampler::write_debug_info_for_pred_resampling(unsigned int                         x,
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
							  const thrust::device_vector<float>&  y_hat) const EddyTry
{
  unsigned int xs = z.Size(0);
  unsigned int ys = z.Size(1);
  unsigned int zs = z.Size(2);

  NEWIMAGE::volume<float> tmpvol = z.GetVolume();
  NEWMAT::ColumnVector tmpvec(zs);
  for (unsigned int k=0; k<zs; k++) tmpvec(k+1) = tmpvol(x,y,k);
  std::string tmpfname = bfname + "_z";
  MISCMATHS::write_ascii_matrix(tmpfname,tmpvec);

  tmpvol = g.GetVolume();
  for (unsigned int k=0; k<zs; k++) tmpvec(k+1) = tmpvol(x,y,k);
  tmpfname = bfname + "_g";
  MISCMATHS::write_ascii_matrix(tmpfname,tmpvec);

  tmpvol = p.GetVolume();
  for (unsigned int k=0; k<zs; k++) tmpvec(k+1) = tmpvol(x,y,k);
  tmpfname = bfname + "_p";
  MISCMATHS::write_ascii_matrix(tmpfname,tmpvec);

  tmpfname = bfname + "_sz";
  write_matrix(sz,(y*xs+x)*zs,zs,1,tmpfname);

  tmpfname = bfname + "_W";
  write_matrix(W,0,zs,zs,tmpfname);

  tmpfname = bfname + "_Wir";
  write_matrix(Wir,x*zs*zs,zs,zs,tmpfname);

  tmpfname = bfname + "_wgt";
  write_matrix(w,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_wp";
  write_matrix(wp,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_wW";
  write_matrix(wW,x*zs*zs,zs,zs,tmpfname);

  tmpfname = bfname + "_Wirtg";
  write_matrix(Wirtg,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_wWtwp";
  write_matrix(wWtwp,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_WirtWir";
  write_matrix(WirtWir,x*zs*zs,zs,zs,tmpfname);

  tmpfname = bfname + "_wWtwW";
  write_matrix(wWtwW,x*zs*zs,zs,zs,tmpfname);

  tmpfname = bfname + "_sum_vec";
  write_matrix(sum_vec,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_sum_mat";
  write_matrix(sum_mat,x*zs*zs,zs,zs,tmpfname);

  tmpfname = bfname + "_c_hat";
  write_matrix(c_hat,x*zs,zs,1,tmpfname);

  tmpfname = bfname + "_y_hat";
  write_matrix(y_hat,x*zs,zs,1,tmpfname);
} EddyCatch

} // End namespace EDDY
