/*! \file CPUStackResampler.cpp
    \brief Contains definitions of a class for spline/tri-linear resampling of irregularly sampled columns in the z-direction.

    \author Jesper Andersson
    \version 1.0b, May, 2021.
*/
//
// CPUStackResampler.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2021 University of Oxford 
//

#include <algorithm>
#include "CPUStackResampler.h"

namespace EDDY {

CPUStackResampler::CPUStackResampler(const NEWIMAGE::volume<float>&  stack,
				     const NEWIMAGE::volume<float>&  zcoord,
				     const NEWIMAGE::volume<float>&  pred,
				     const NEWIMAGE::volume<float>&  mask,
				     double                          lambda)
{
  _ovol = stack;
  _ovol = 0.0;
  _omask = _ovol;
  spline_interpolate_slice_stack(stack,zcoord,mask,lambda,&pred,_ovol,_omask);
}

CPUStackResampler::CPUStackResampler(const NEWIMAGE::volume<float>&  stack,
				     const NEWIMAGE::volume<float>&  zcoord,
				     const NEWIMAGE::volume<float>&  mask,
				     NEWIMAGE::interpolation         interp,
				     double                          lambda)
{
  _ovol = stack;
  _ovol = 0.0;
  _omask = _ovol;
  if (interp==NEWIMAGE::spline) {
    spline_interpolate_slice_stack(stack,zcoord,mask,lambda,nullptr,_ovol,_omask);
  }
  else if (interp==NEWIMAGE::trilinear) {
    linear_interpolate_slice_stack(stack,zcoord,mask,_ovol,_omask);
  }
}

void CPUStackResampler::spline_interpolate_slice_stack(// Input
						       const NEWIMAGE::volume<float>&   slice_stack,
						       const NEWIMAGE::volume<float>&   z_coord,
						       const NEWIMAGE::volume<float>&   stack_mask,
						       double                           lambda,
						       // Optional input
						       const NEWIMAGE::volume<float>    *pred_ptr,
						       // Output
						       NEWIMAGE::volume<float>&         ovol,
						       NEWIMAGE::volume<float>&         omask) EddyTry
{
  // Get regularisation and regular sampline spline matrices once and for all
  arma::Mat<float> StS = get_StS(ovol.zsize(),lambda);
  arma::Mat<float> W = get_regular_W(ovol.zsize());
  for (int i=0; i<ovol.xsize(); i++) {
    for (int j=0; j<ovol.ysize(); j++) {
      arma::Mat<float> Wir = get_Wir(z_coord,i,j);
      arma::Col<float> y = get_y(slice_stack,i,j);
      arma::Col<float> interpolated_column;
      if (pred_ptr == nullptr) { // If we don't use predictions for support
	interpolated_column = W * solve(Wir.t()*Wir + StS,Wir.t()*y);
      }
      else { // If we want to use predictions
	std::vector<float> sorted_zcoords = sort_zcoord(z_coord,i,j);
	arma::Mat<float> PW = get_prediction_weights(sorted_zcoords);
	arma::Mat<float> WirW = arma::join_cols(Wir,PW*W);
	arma::Col<float> y_pred = arma::join_cols(y,PW*get_y(*pred_ptr,i,j));
	interpolated_column = W * solve(WirW.t()*WirW + StS,WirW.t()*y_pred);
      }
      for (int k=0; k<ovol.zsize(); k++) ovol(i,j,k) = interpolated_column[k]; // Insert column
    }
  }
  omask = stack_mask; // Revisit
} EddyCatch

void CPUStackResampler::linear_interpolate_slice_stack(// Input
						       const NEWIMAGE::volume<float>&   slice_stack,
						       const NEWIMAGE::volume<float>&   z_coord,
						       const NEWIMAGE::volume<float>&   stack_mask,
						       // Output
						       NEWIMAGE::volume<float>&         ovol,
						       NEWIMAGE::volume<float>&         omask) EddyTry
{
  struct triplet {
    triplet(float zz, float ii, float mm) : z(zz), i(ii), m(mm) {}
    triplet() : z(0.0), i(0.0), m(0.0) {}
    float z, i, m; // z-coord, ima-value, vaild_mask
  };
  // Allocate vector for sorting
  std::vector<triplet> z_col(ovol.zsize()); 
  // Do the interpolation
  for (int j=0; j<ovol.ysize(); j++) {
    for (int i=0; i<ovol.xsize(); i++) {
      // Repack z-column into vector and sort if needed
      z_col[0] = triplet(z_coord(i,j,0),slice_stack(i,j,0),stack_mask(i,j,0));
      bool needs_sorting = false;
      for (int k=1; k<ovol.zsize(); k++) {
	z_col[k] = triplet(z_coord(i,j,k),slice_stack(i,j,k),stack_mask(i,j,k));
	if (z_col[k].z < z_col[k-1].z) needs_sorting = true;
      }
      if (needs_sorting) std::sort(z_col.begin(),z_col.end(),[](const triplet& a, const triplet& b) { return(a.z < b.z); });
      // Here starts the actual interpolation
      for (int k=0; k<ovol.zsize(); k++) {
	int kk=0;
	for (kk=0; kk<ovol.zsize(); kk++) if (z_col[kk].z > k) break;
	if (kk==0) { 
	  if (z_col[kk].z < 0.5 && z_col[kk].m) { ovol(i,j,k) = z_col[kk].i; omask(i,j,k) = 1; }
	  else { ovol(i,j,k) = 0; omask(i,j,k) = 0; }
	}
	else if (kk==ovol.zsize()) { 
	  if (z_col[kk-1].z > ovol.zsize() - 0.5 && z_col[kk-1].m) { ovol(i,j,k) = z_col[kk-1].i; omask(i,j,k) = 1; }
	  else { ovol(i,j,k) = 0; omask(i,j,k) = 0; }
	}
	else {
	  if (z_col[kk-1].m && z_col[kk].m) {
	    ovol(i,j,k) = z_col[kk-1].i + (k-z_col[kk-1].z) * (z_col[kk].i-z_col[kk-1].i) / (z_col[kk].z-z_col[kk-1].z);
	    omask(i,j,k) = 1;
	  }
	  else { ovol(i,j,k) = 0; omask(i,j,k) = 0; }
	}
      }
    }
  }  
  return;
} EddyCatch

arma::Mat<float> CPUStackResampler::get_StS(int sz, float lambda) const EddyTry
{
  arma::Mat<float> StS(sz,sz,arma::fill::zeros);
  StS(0,0) = 6.0*lambda; StS(0,1) = -4.0*lambda; StS(0,2) = lambda; StS(0,sz-2) = lambda; StS(0,sz-1) = -4.0*lambda; 
  StS(1,0) = -4.0*lambda; StS(1,1) = 6.0*lambda; StS(1,2) = -4.0*lambda; StS(1,3) = lambda; StS(1,sz-1) = lambda;
  for (int i=2; i<(sz-2); i++) {
    StS(i,i-2) = lambda; StS(i,i-1) = -4.0*lambda; StS(i,i) = 6.0*lambda; StS(i,i+1) = -4.0*lambda; StS(i,i+2) = lambda;
  }
  StS(sz-2,sz-4) = lambda; StS(sz-2,sz-3) = -4.0*lambda; StS(sz-2,sz-2) = 6.0*lambda; StS(sz-2,sz-1) = -4.0*lambda; StS(sz-2,0) = lambda;
  StS(sz-1,sz-3) = lambda; StS(sz-1,sz-2) = -4.0*lambda; StS(sz-1,sz-1) = 6.0*lambda; StS(sz-1,0) = -4.0*lambda; StS(sz-1,1) = lambda;
  return(StS);
} EddyCatch

arma::Mat<float> CPUStackResampler::get_regular_W(int sz) const EddyTry
{
  arma::Mat<float> W(sz,sz,arma::fill::zeros);
  W(0,0) = 5.0/6.0; W(0,1) = 1.0/6.0;
  for (int i=1; i<(sz-1); i++) {
    W(i,i-1) = 1.0/6.0; W(i,i) = 4.0/6.0; W(i,i+1) = 1.0/6.0;
  }
  W(sz-1,sz-2) = 1.0/6.0; W(sz-1,sz-1) = 5.0/6.0;
  return(W);
} EddyCatch

arma::Mat<float> CPUStackResampler::get_Wir(const NEWIMAGE::volume<float>& zcoord,
					    int i, int j) const EddyTry
{
  arma::Mat<float> Wir(zcoord.zsize(),zcoord.zsize(),arma::fill::zeros);
  for (int k=0; k<zcoord.zsize(); k++) {
    if (zcoord(i,j,k)>=0 && zcoord(i,j,k)<=(zcoord.zsize()-1)) { // If in valid range
      int iz = static_cast<int>(zcoord(i,j,k));
      for (int c=iz-2; c<iz+3; c++) {
	Wir(k,std::min(std::max(0,c),static_cast<int>(zcoord.zsize()-1))) += wgt_at(zcoord(i,j,k)-static_cast<float>(c));
      }
    }
  }
  return(Wir);
} EddyCatch

float CPUStackResampler::wgt_at(float x) const EddyTry
{
  float wgt = 0.0;
  x = (x<0.0) ? -x : x;
  if (x < 1) wgt = 2.0/3.0 + 0.5*x*x*(x-2.0);
  else if (x < 2) wgt = (1.0/6.0) * (2.0-x)*(2.0-x)*(2.0-x);

  return(wgt);
} EddyCatch

std::vector<float> CPUStackResampler::sort_zcoord(const NEWIMAGE::volume<float>& zcoord,
						  int i, int j) const EddyTry
{
  std::vector<float> ovec(zcoord.zsize());
  bool needs_sorting = false;
  ovec[0] = zcoord(i,j,0);
  for (int k=1; k<zcoord.zsize(); k++) {
    ovec[k] = zcoord(i,j,k);
    if (ovec[k] < ovec[k-1]) needs_sorting = true;
  }
  if (needs_sorting) std::sort(ovec.begin(),ovec.end());

  return(ovec);
} EddyCatch

arma::Mat<float> CPUStackResampler::get_prediction_weights(const std::vector<float> zcoord) const EddyTry
{
  arma::Mat<float> wgts(zcoord.size(),zcoord.size(),arma::fill::zeros);
  for (unsigned int k=0; k<zcoord.size(); k++) {
    unsigned int i=0;
    for (i=0; i<zcoord.size(); i++) if (zcoord[i] > k) break;
    if (i==0) {
      wgts(k,k) = std::min(zcoord[i]-k,1.0f);
    }
    else if (i==zcoord.size()) {
      wgts(k,k) = std::min(k-zcoord[i-1],1.0f);
    }
    else {
      float gap = zcoord[i]-zcoord[i-1];
      if (gap < 1.0) { // If gap < one voxel
	wgts(k,k) = 0.0;
      }
      else if (gap < 2.0 && std::max(k-zcoord[i-1],zcoord[i]-k) < 1.0) { // If gap < 2 voxels and only one prediction in gap
	wgts(k,k) = gap - 1.0f;
      }
      else { // If there is more than one prediction in gap
	wgts(k,k) = std::min(1.0f,std::min(k-zcoord[i-1],zcoord[i]-k));
      }
    }
    if (wgts(k,k) > 1e-12) wgts(k,k) = std::sqrt(wgts(k,k)); // Avoid taking sqrt of very small value
  }
  return(wgts);
} EddyCatch

} // End namespace EDDY
