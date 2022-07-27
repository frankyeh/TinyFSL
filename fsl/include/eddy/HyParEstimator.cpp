/*! \file HyParEstimator.cpp
    \brief Contains definitions of classes implementing hyper parameter estimation for DWI GPs

    \author Jesper Andersson
    \version 1.0b, Nov., 2013.
*/
// Definitions of classes implementing hyper parameter estimation for DWI GPs
//
// HyParEstimator.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford
//

#include <cstdlib>
#include <string>
#include <exception>
#include <vector>
#include <cmath>
#include <limits>
#include <ctime>
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "HyParEstimator.h"
#include "TIPL/tipl.hpp"

using namespace std;
using namespace EDDY;

NEWMAT::Matrix DataSelector::AsNEWMAT() const EddyTry
{
  NEWMAT::Matrix m(_data[0].Nrows(),_data.size());
  for (unsigned int i=0; i<_data.size(); i++) {
    m.Column(i+1) = _data[i];
  }
  return(m);
} EddyCatch

NEWMAT::Matrix DataSelector::CoordsAsNEWMAT() const EddyTry
{
  NEWMAT::Matrix m(3,_coords.size());
  for (unsigned int i=0; i<_coords.size(); i++) {
    m(1,i+1) = _coords[i][0]; m(2,i+1) = _coords[i][1]; m(3,i+1) = _coords[i][2];
  }
  return(m);
} EddyCatch

void DataSelector::common_constructor(const std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >& idata,
				      const NEWIMAGE::volume<float>&                                 mask,
				      unsigned int                                                   rnvox,
				      float                                                          fwhm,
				      int                                                            rndinit) EddyTry
{
  // Initialise rand if no explicit seed given
  if (rndinit) srand(rndinit);
  else srand(time(NULL));
  // Check that there are at least rnvox voxels available
  unsigned int nnz = 0; // Number of non-zero voxels in mask
  for (auto it = mask.fbegin(); it!=mask.fend(); ++it) nnz += (*it) ? 1 : 0;
  if (nnz < rnvox) throw EddyException("DataSelector::common_constructor: rnvox greater than number of non-zero voxels in mask.");
  // Select rnvox voxel coordinates within mask
  unsigned int nx = static_cast<unsigned int>(mask.xsize());
  unsigned int ny = static_cast<unsigned int>(mask.ysize());
  unsigned int nz = static_cast<unsigned int>(mask.zsize());
  std::vector<std::vector<unsigned int> > coords(rnvox);
  unsigned int nvox = 0;
  if (rnvox) {
    unsigned int l=0; unsigned int maxtry=1e8;
    for (l=0; l<maxtry; l++) {
      std::vector<unsigned int> nc(3);
      nc[0] = rand() % nx; nc[1] = rand() % ny; nc[2] = rand() % nz;
      if (mask(nc[0],nc[1],nc[2])) {
	unsigned int vox;
	for (vox=0; vox<nvox; vox++) if (coords[vox]==nc) break;
	if (vox == nvox) coords[nvox++] = nc; // If no duplicate found
      }
      if (nvox == rnvox) break;
    }
    if (l==maxtry) throw EddyException("DataSelector::common_constructor: unable to find requested number of unique voxels");
  }
  std::vector<NEWMAT::ColumnVector> kernels(3);
  // Make convolution kernels if fwhm>0
  if (fwhm > 0.0) {
    float sx = (fwhm/std::sqrt(8.0*std::log(2.0)))/idata[0]->xdim();
    float sy = (fwhm/std::sqrt(8.0*std::log(2.0)))/idata[0]->ydim();
    float sz = (fwhm/std::sqrt(8.0*std::log(2.0)))/idata[0]->zdim();
    int nx=((int) (sx-0.001))*2 + 3;
    int ny=((int) (sy-0.001))*2 + 3;
    int nz=((int) (sz-0.001))*2 + 3;
    kernels[0] = NEWIMAGE::gaussian_kernel1D(sx,nx);
    kernels[1] = NEWIMAGE::gaussian_kernel1D(sy,ny);
    kernels[2] = NEWIMAGE::gaussian_kernel1D(sz,nz);
  }

  // Get time-series from the selected coordinates
  NEWMAT::ColumnVector vec(idata.size());
  _data.resize(nvox,vec);
  for (unsigned int i=0; i<idata.size(); i++) {
    const NEWIMAGE::volume<float>& vol = *idata[i];
    for (unsigned int j=0; j<nvox; j++) {
      if (fwhm > 0.0) _data[j](i+1) = get_smooth(vol,mask,coords[j],kernels);
      else _data[j](i+1) = vol(coords[j][0],coords[j][1],coords[j][2]);
    }
  }
  // Save coordinates
  _coords = coords;
} EddyCatch

float DataSelector::get_smooth(const NEWIMAGE::volume<float>&           ima,
			       const NEWIMAGE::volume<float>&           mask,
			       const std::vector<unsigned int>&         coords,
			       const std::vector<NEWMAT::ColumnVector>& kernels) EddyTry
{
  float oval = 0.0;
  float totwgt = 0.0;
  for (int k=0; k<kernels[2].Nrows(); k++) {
    float kwgt = kernels[2](k+1);
    for (int j=0; j<kernels[1].Nrows(); j++) {
      float jkwgt = kwgt*kernels[1](j+1);
      for (int i=0; i<kernels[0].Nrows(); i++) {
	int kk = coords[2] - kernels[2].Nrows()/2 + k;
	if (kk>=0 && kk<ima.zsize()) {
	  int jj = coords[1] - kernels[1].Nrows()/2 + j;
	  if (jj>=0 && jj<ima.ysize()) {
	    int ii = coords[0] - kernels[0].Nrows()/2 + i;
	    if (ii>=0 && ii<ima.xsize() && mask(ii,jj,kk)) {
	      float tmp = jkwgt*kernels[0](i+1);
	      totwgt += tmp;
	      oval += tmp*ima(ii,jj,kk);
	    }
	  }
	}
      }
    }
  }
  oval /= totwgt;
  return(oval);
} EddyCatch

double MMLHyParCF::cf(const NEWMAT::ColumnVector& p) const EddyTry
{
  std::vector<double> hp(p.Nrows());
  for (int i=0; i<p.Nrows(); i++) hp[i] = p(i+1);
  _K->SetHyperPar(hp);
  if (!_K->IsValid()) return(std::numeric_limits<double>::max());
  double ldKy = _K->LogDet();
  double neg_log_marg_ll = 0.0;
  for (unsigned int i=0; i<_data.size(); i++)  neg_log_marg_ll += NEWMAT::DotProduct(_data[i],_K->iKy(_data[i]));
  neg_log_marg_ll += _data.size() * ldKy;
  neg_log_marg_ll *= 0.5;
  return(neg_log_marg_ll);
} EddyCatch

double CVHyParCF::cf(const NEWMAT::ColumnVector& p) const EddyTry
{
  std::vector<double> hp(p.Nrows());
  for (int i=0; i<p.Nrows(); i++) hp[i] = p(i+1);
  _K->SetHyperPar(hp);
  if (!_K->IsValid()) return(std::numeric_limits<double>::max());
  NEWMAT::ColumnVector ssd_vec(_K->NoOfScans());
  ssd_vec = 0.0;
  /*
  for (unsigned int i=0; i<_data.size(); i++) {
    NEWMAT::ColumnVector qn = _K->iKy(_data[i]);
    ssd_vec += NEWMAT::SP(qn,qn);
  }
  */
  // tinyFSL replaces the above with the following to boost the speed
  {
      std::vector<NEWMAT::ColumnVector> ssd_vecs;
      for(size_t i = 0;i < std::thread::hardware_concurrency();++i)
      {
          ssd_vecs.push_back(NEWMAT::ColumnVector(_K->NoOfScans()));
          ssd_vecs.back() = 0.0;
      }
      tipl::par_for(_data.size(),[&](size_t i,unsigned int thread){
          NEWMAT::ColumnVector qn = _K->iKy(_data[i]);
          ssd_vecs[thread] += NEWMAT::SP(qn,qn);
      });
      for(size_t i = 0;i < std::thread::hardware_concurrency();++i)
          ssd_vec += ssd_vecs[i];
  }

  NEWMAT::SymmetricMatrix iK = _K->iK();
  double ssd = 0.0;
  for (unsigned int i=0; i<_K->NoOfScans(); i++) ssd += ssd_vec(i+1) / sqr(iK(i+1,i+1));
  ssd /= _K->NoOfScans();
  return(ssd);
} EddyCatch

double GPPHyParCF::cf(const NEWMAT::ColumnVector& p) const EddyTry
{
  std::vector<double> hp(p.Nrows());
  for (int i=0; i<p.Nrows(); i++) hp[i] = p(i+1);
  _K->SetHyperPar(hp);
  if (!_K->IsValid()) return(std::numeric_limits<double>::max());
  NEWMAT::ColumnVector gpp_vec(_K->NoOfScans());
  gpp_vec = 0.0;
  for (unsigned int i=0; i<_data.size(); i++) {
    NEWMAT::ColumnVector qn = _K->iKy(_data[i]);
    gpp_vec += NEWMAT::SP(qn,qn);
  }
  NEWMAT::SymmetricMatrix iK = _K->iK();
  for (unsigned int i=0; i<_K->NoOfScans(); i++) {
    gpp_vec(i+1) /= iK(i+1,i+1);
    gpp_vec(i+1) -= _data.size() * std::log(iK(i+1,i+1));
  }
  double gpp = 0.5 * gpp_vec.Sum() / _K->NoOfScans();
  return(gpp);
} EddyCatch

void CheapAndCheerfulHyParEstimator::Estimate(std::shared_ptr<const KMatrix>  K,
					      bool                            verbose) EddyTry
{
  std::shared_ptr<KMatrix> lK = K->Clone(); // Local copy of K
  lK->SetHyperPar(lK->GetHyperParGuess(_data));
  lK->MulErrVarBy(5.0); // Corresponds to ~ ff=5
  set_hpar(stl_2_newmat(lK->GetHyperPar()));
  if (verbose) cout << "Hyperparameters guesstimated to be: " << get_hpar() << endl;
} EddyCatch

void FullMontyHyParEstimator::Estimate(std::shared_ptr<const KMatrix>  K,
				       bool                            vbs) EddyTry
{
  std::shared_ptr<KMatrix> lK = K->Clone(); // Local copy of K
  set_hpar(stl_2_newmat(lK->GetHyperParGuess(_data)));
  if (_v) cout << "Initial guess for hyperparameters: " << get_hpar() << endl;
  _cf->SetData(_data);
  _cf->SetKMatrix(lK);
  MISCMATHS::NonlinParam nlpar(get_hpar().Nrows(),MISCMATHS::NL_NM,get_hpar());
  nlpar.SetVerbose(_v);
  nlpar.SetPrintWidthPrecision(10,5,15,10);
  nlpar.SetMaxIter(MITER);
  MISCMATHS::nonlin(nlpar,*_cf);
  set_hpar(nlpar.Par());
  if (_evff > 1.0) {
    if (_v) {
      cout << "Fudging parameters" << endl;
      cout << "Parameters start out as " << get_hpar() << endl;
    }
    std::shared_ptr<KMatrix> KK = lK->Clone();
    std::vector<double> hpar(get_hpar().Nrows());
    for (unsigned int i=0; i<hpar.size(); i++) hpar[i] = get_hpar()(i+1);
    KK->SetHyperPar(hpar);
    KK->MulErrVarBy(_evff);
    hpar = KK->GetHyperPar();
    NEWMAT::ColumnVector tmp_par(get_hpar().Nrows());
    for (unsigned int i=0; i<hpar.size(); i++) tmp_par(i+1) = hpar[i];
    set_hpar(tmp_par);
    if (_v) cout << "Parameters end up as " << get_hpar() << endl;
  }
  if (vbs) cout << "Estimated hyperparameters: " << get_hpar() << endl;
} EddyCatch
