/*! \file KMatrix.cpp
    \brief Contains definitions for classes implementing covariance matrices for DWI GPs.

    \author Jesper Andersson
    \version 1.0b, Oct., 2013.
*/
// Definitions of class implementing covariance matrices for DWI GPs
//
// KMatrix.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "EddyUtils.h"
#include "KMatrix.h"

using namespace std;
using namespace EDDY;

/////////////////////////////////////////////////////////////////////
//
// Class MultiShellKMatrix
//
/////////////////////////////////////////////////////////////////////

/****************************************************************//**
*
* Constructs a MultiShellKMatrix object given b-vecs, shell information
* and values of hyperparameters.
* Note that MultiShellKMatrix is still a virtual class that cannot
* be instantiated. Despite that its constructor does a significant
* amount of work.
* \param dpars Vector of DiffPara objects giving b-value and b-vector
*        for each scan.
* \param npps Number of hyper-parameters per shell.
*
********************************************************************/
MultiShellKMatrix::MultiShellKMatrix(const std::vector<DiffPara>&          dpars,
				     bool                                  dcsh) EddyTry
: _dpars(dpars), _K_ok(false), _iK_ok(false), _pv(dpars.size()), _pv_ok(dpars.size(),false), _dcsh(dcsh)
{
  // Get information on grouping
  if (!EddyUtils::GetGroups(dpars,_grps,_grpi,_grpb) && !_dcsh) throw EddyException("MultiShellKMatrix::MultiShellKMatrix: Data not shelled");
  _ngrp = _grpb.size();

  // Populate lower diagonal of matrix of angles between b-vecs
  make_angle_mat();
} EddyCatch

NEWMAT::ColumnVector MultiShellKMatrix::iKy(const NEWMAT::ColumnVector& y) const EddyTry
{
  if (y.Nrows() != _K.Nrows()) throw EddyException("MultiShellKMatrix::iKy:const: Invalid size of vector y");
  if (!_K_ok) throw EddyException("MultiShellKMatrix::iKy:const: Attempting to use invalid (NPD) K-matrix");
  if (!_iK_ok) throw EddyException("MultiShellKMatrix::iKy:const: Cannot use lazy evaluation for const object");
  return(_iK*y);
} EddyCatch

NEWMAT::ColumnVector MultiShellKMatrix::iKy(const NEWMAT::ColumnVector& y) EddyTry
{
  if (y.Nrows() != _K.Nrows()) throw EddyException("MultiShellKMatrix::iKy: Invalid size of vector y");
  if (!_K_ok) throw EddyException("MultiShellKMatrix::iKy: Attempting to use invalid (NPD) K-matrix");
  if (!_iK_ok) calculate_iK();
  return(_iK*y);
} EddyCatch

NEWMAT::RowVector MultiShellKMatrix::PredVec(unsigned int i,
					     bool         excl) const EddyTry
{
  if (int(i)>(_K.Nrows()-1)) throw EddyException("MultiShellKMatrix::PredVec:const: Invalid index");
  if (!_K_ok) throw EddyException("MultiShellKMatrix::PredVec:const: Attempting to use invalid (NPD) K-matrix");
  if (!excl) {
    if (!_iK_ok) throw EddyException("MultiShellKMatrix::PredVec:const: Cannot use lazy evaluation for const object");
    NEWMAT::RowVector v = k_row(i,excl,_grpi,_ngrp,_thpar,_angle_mat);
    return(v*_iK);
  }
  else {
    if (!_pv_ok[i]) throw EddyException("MultiShellKMatrix::PredVec:const: Cannot use lazy evaluation for const object");
    return(_pv[i]);
  }
} EddyCatch

NEWMAT::RowVector MultiShellKMatrix::PredVec(unsigned int i,
					     bool         excl) EddyTry
{
  if (int(i)>(_K.Nrows()-1)) throw EddyException("MultiShellKMatrix::PredVec: Invalid index");
  if (!_K_ok) throw EddyException("MultiShellKMatrix::PredVec: Attempting to use invalid (NPD) K-matrix");
  if (!excl) {
    if (!_iK_ok) calculate_iK();
    NEWMAT::RowVector v = k_row(i,excl,_grpi,_ngrp,_thpar,_angle_mat);
    return(v*_iK);
  }
  else {
    if (!_pv_ok[i]) {
      NEWMAT::RowVector v = k_row(i,excl,_grpi,_ngrp,_thpar,_angle_mat);
      _pv[i] = v*calculate_iK_index(i);
      _pv_ok[i] = true;
    }
    return(_pv[i]);
  }
} EddyCatch

double MultiShellKMatrix::PredVar(unsigned int i,
				  bool         excl) EddyTry
{
  if (int(i)>(_K.Nrows()-1)) throw EddyException("MultiShellKMatrix::PredVar: Invalid index");
  if (!_K_ok) throw EddyException("MultiShellKMatrix::PredVar: Attempting to use invalid (NPD) K-matrix");
  if (!excl) {
    if (!_iK_ok) calculate_iK();
    NEWMAT::RowVector v = k_row(i,excl,_grpi,_ngrp,_thpar,_angle_mat);
    double sv = sig_var(i,_grpi,_ngrp,_thpar);
    return(sv - (v*_iK*v.t()).AsScalar());
  }
  else {
    NEWMAT::RowVector v = k_row(i,excl,_grpi,_ngrp,_thpar,_angle_mat);
    double sv = sig_var(i,_grpi,_ngrp,_thpar);
    NEWMAT::SymmetricMatrix iKi = calculate_iK_index(i);
    return(sv - (v*calculate_iK_index(i)*v.t()).AsScalar());
  }
} EddyCatch

const NEWMAT::SymmetricMatrix& MultiShellKMatrix::iK() const EddyTry
{
  if (!_K_ok) throw EddyException("MultiShellKMatrix::iK:const: Attempting to use invalid (NPD) K-matrix");
  if (!_iK_ok) throw EddyException("MultiShellKKMatrix::iK:const: Cannot use lazy evaluation for const object");
  return(_iK);
} EddyCatch

const NEWMAT::SymmetricMatrix& MultiShellKMatrix::iK() EddyTry
{
  if (!_K_ok) throw EddyException("MultiShellKMatrix::iK: Attempting to use invalid (NPD) K-matrix");
  if (!_iK_ok) calculate_iK();
  return(_iK);
} EddyCatch

void MultiShellKMatrix::CalculateInvK() EddyTry
{
  if (!_K_ok) throw EddyException("MultiShellKMatrix::CalculateInvK: Attempting to use invalid (NPD) K-matrix");
  if (!_iK_ok) calculate_iK();
} EddyCatch

void MultiShellKMatrix::Reset() EddyTry
{
  _dpars.clear(); _grpi.clear(); _grps.clear(); _hpar.clear(); _thpar.clear(); _pv.clear(); _pv_ok.clear();
  _ngrp = 0;
  _K_ok = _iK_ok = false;
  _angle_mat.CleanUp(); _K.CleanUp(); _cK.CleanUp(); _iK.CleanUp();
} EddyCatch

void MultiShellKMatrix::SetDiffusionPar(const std::vector<DiffPara>& dpars) EddyTry
{
  _dpars = dpars;
  if (!EddyUtils::GetGroups(_dpars,_grps,_grpi,_grpb) && !_dcsh) throw EddyException("MultiShellKMatrix::SetDiffusionPar: Data not shelled");
  _ngrp = _grpb.size();
  // Populate (lower diagonal of) matrix of angles between b-vecs
  make_angle_mat();
  // Set arbitrary hyper-parameters (using functions supplied by derived class)
  _hpar = get_arbitrary_hpar(_ngrp);
  _thpar = exp_hpar(_ngrp,_hpar);
  // Calculate K-matrix (using function supplied by derived class)
  calculate_K_matrix(_grpi,_ngrp,_thpar,_angle_mat,_K);
  validate_K_matrix();
  _iK_ok = false;
  _pv_ok.assign(_dpars.size(),false);
  _pv.resize(_dpars.size());
} EddyCatch

void MultiShellKMatrix::SetHyperPar(const std::vector<double>& hpar) EddyTry
{
  if (hpar.size() != n_par(_ngrp)) throw EddyException("MultiShellKMatrix::SetHyperPar: Invalid # of hyperparameters");
  _hpar = hpar;
  _thpar = exp_hpar(_ngrp,_hpar);
  calculate_K_matrix(_grpi,_ngrp,_thpar,_angle_mat,_K);
  validate_K_matrix();
  _iK_ok = false;
  _pv_ok.assign(_dpars.size(),false);
} EddyCatch

void MultiShellKMatrix::MulErrVarBy(double ff) EddyTry
{
  if (ff < 1.0) throw EddyException("MultiShellKMatrix::MulErrVarBy: Fudge factor must be > 1.0");
  else{
    std::vector<double> hpar = GetHyperPar();
    for (unsigned int j=0; j<NoOfGroups(); j++) {
      for (unsigned int i=j; i<NoOfGroups(); i++) {
	unsigned int pi = ij_to_parameter_index(i,j,NoOfGroups());
	if (i==j) hpar[pi+2] += std::log(ff);
      }
    }
    set_hpar(hpar);
    set_thpar(exp_hpar(NoOfGroups(),GetHyperPar()));
  }
} EddyCatch

double MultiShellKMatrix::LogDet() const EddyTry
{
  if (!_K_ok) throw EddyException("MultiShellKMatrix::LogDet: Attempting to use invalid (NPD) K-matrix");
  double ld = 0.0;
  for (int i=0; i<_cK.Nrows(); i++) ld += log(_cK(i+1,i+1));
  return(2.0*ld);
} EddyCatch

NEWMAT::SymmetricMatrix MultiShellKMatrix::GetDeriv(unsigned int di) const EddyTry
{
  std::pair<unsigned int, unsigned int> ij = parameter_index_to_ij(di,_ngrp);
  unsigned int bdi = ij_to_parameter_index(ij.first,ij.second,_ngrp);
  NEWMAT::SymmetricMatrix dK(NoOfScans());
  calculate_dK_matrix(_grpi,_ngrp,_thpar,_angle_mat,ij.first,ij.second,di-bdi,dK);
  return(dK);
} EddyCatch

void MultiShellKMatrix::GetAllDerivs(std::vector<NEWMAT::SymmetricMatrix>& derivs) const EddyTry
{
  for (unsigned int di=0; di<NoOfHyperPar(); di++) {
    if (derivs[di].Nrows() != int(NoOfScans())) derivs[di].ReSize(NoOfScans());
    std::pair<unsigned int, unsigned int> ij = parameter_index_to_ij(di,_ngrp);
    unsigned int bdi = ij_to_parameter_index(ij.first,ij.second,_ngrp);
    calculate_dK_matrix(_grpi,_ngrp,_thpar,_angle_mat,ij.first,ij.second,di-bdi,derivs[di]);
  }
  return;
} EddyCatch

void MultiShellKMatrix::Print() const EddyTry
{
  cout << "_ngrp = " << _ngrp << endl;
  cout << "_K_ok = " << _K_ok << endl;
  cout << "_iK_ok = " << _iK_ok << endl;
} EddyCatch

void MultiShellKMatrix::Write(const std::string& basefname) const EddyTry
{
  std::string fname = basefname + ".angle_mat.txt";
  MISCMATHS::write_ascii_matrix(_angle_mat,fname);
  fname = basefname + ".K.txt";
  MISCMATHS::write_ascii_matrix(_K,fname);
  fname = basefname + ".cK.txt";
  MISCMATHS::write_ascii_matrix(_cK,fname);
  fname = basefname + ".iK.txt";
  MISCMATHS::write_ascii_matrix(_iK,fname);
  fname = basefname + ".pv.txt";
  NEWMAT::Matrix pvs = make_pred_vec_matrix();
  MISCMATHS::write_ascii_matrix(pvs,fname);
} EddyCatch

void MultiShellKMatrix::validate_K_matrix() EddyTry
{
  try {
    _cK = Cholesky(_K);
    _K_ok = true;
  }
  catch (NEWMAT::NPDException& e) {
    _K_ok = false;
  }
  return;
} EddyCatch

void MultiShellKMatrix::calculate_iK() EddyTry
{
  NEWMAT::LowerTriangularMatrix iL = _cK.i();
  _iK << iL.t()*iL; // N.B. _ik = iL.t()*iL; does NOT work
  _iK_ok = true;
  return;
} EddyCatch

NEWMAT::SymmetricMatrix MultiShellKMatrix::calculate_iK_index(unsigned int i) const EddyTry
{
  int n = _K.Nrows(); // # of rows AND # of columns
  NEWMAT::Matrix top = (_K.SubMatrix(1,i,1,i) | _K.SubMatrix(1,i,i+2,n));
  NEWMAT::Matrix bottom = (_K.SubMatrix(i+2,n,1,i) | _K.SubMatrix(i+2,n,i+2,n));
  top &= bottom;
  NEWMAT::SymmetricMatrix K_indx;
  K_indx << top;
  NEWMAT::SymmetricMatrix iK_indx = K_indx.i();
  return(iK_indx);
} EddyCatch

NEWMAT::Matrix MultiShellKMatrix::make_pred_vec_matrix(bool excl) const EddyTry
{
  NEWMAT::Matrix M(_K.Nrows(),_K.Ncols());
  for (int i=0; i<M.Nrows(); i++) M.Row(i+1) = PredVec(i,excl);
  return(M);
} EddyCatch

/*
unsigned int MultiShellKMatrix::ij_to_parameter_index(unsigned int i,
						      unsigned int j,
						      unsigned int n) const
{
  if (n==1) { // The one--three shell cases hardcoded for speed
    return(0);
  }
  else if (n==2) {
    if (!j) return((_npps)*i);
    else return(_npps+_npps-1);
  }
  else if (n==3) {
    if (!j) { if (i) return(i*_npps-i+1); else return(0); }
    else if (j==1) return(2*_npps+i*_npps-2);
    else return(5*_npps-3); // for the i==j==2 case
  }
  else { // Not hardcoded. Would be nice if I could find a faster algorithm.
    unsigned int rval = 0;
    for (unsigned int ii=0; ii<j; ii++) { // Sum up index of preceeding columns
      rval += _npps + (_npps-1)*(n-1-ii);
    }
    if (i>j) rval += _npps + (_npps-1)*(i-j-1);
    return(rval);
  }
  return(0); // To stop compiler complaining.
}
*/

std::pair<unsigned int, unsigned int> MultiShellKMatrix::parameter_index_to_ij(unsigned int pi,
									       unsigned int ngrp) const EddyTry
{
  std::pair<unsigned int, unsigned int> ij;
  for (unsigned int j=0; j<ngrp; j++) {
    for (unsigned int i=j; i<ngrp; i++) {
      // ij_to_parameter_index will be defined in a derived class
      if (ij_to_parameter_index(i,j,ngrp) > pi) { // Will alawys fail first time
	break;
      }
      ij.first = i; ij.second = j;
    }
  }
  return(ij);
} EddyCatch


void MultiShellKMatrix::make_angle_mat() EddyTry
{
  _angle_mat.ReSize(_dpars.size());
  for (unsigned int j=0; j<_dpars.size(); j++) {
    for (unsigned int i=j; i<_dpars.size(); i++) {
      _angle_mat(i+1,j+1) = std::acos(std::min(1.0,std::abs(NEWMAT::DotProduct(_dpars[i].bVec(),_dpars[j].bVec()))));
    }
  }
} EddyCatch

double MultiShellKMatrix::mean(const NEWMAT::ColumnVector&      data,
			       const std::vector<unsigned int>& indx) const EddyTry
{
  double m = 0.0;
  for (unsigned int i=0; i<indx.size(); i++) m += data(indx[i]+1);
  return(m / indx.size());
} EddyCatch

double MultiShellKMatrix::variance(const NEWMAT::ColumnVector&      data,
				   const std::vector<unsigned int>& indx) const EddyTry
{
  double m = mean(data,indx);
  double v = 0.0;
  for (unsigned int i=0; i<indx.size(); i++) v += sqr(data(indx[i]+1) - m);
  return(v / (indx.size() - 1));
} EddyCatch

bool MultiShellKMatrix::valid_hpars(const std::vector<double>& hpar) const EddyTry
{
  std::vector<double> thpar = exp_hpar(NoOfGroups(),hpar);
  NEWMAT::SymmetricMatrix K(NoOfScans());
  calculate_K_matrix(grpi(),NoOfGroups(),thpar,angle_mat(),K);
  try { NEWMAT::LowerTriangularMatrix cK = Cholesky(K); }
  catch (NEWMAT::NPDException& e) { return(false); }
  return(true);
} EddyCatch

NewSphericalKMatrix::NewSphericalKMatrix(const std::vector<DiffPara>&          dpars,
					 bool                                  dcsh) EddyTry
: MultiShellKMatrix(dpars,dcsh)
{
  // Set (arbitrary) hyper-parameters that will at least yield a valid K (Gram) matrix
  set_hpar(get_arbitrary_hpar(NoOfGroups()));
  // Transform variance related hyperparameters
  set_thpar(exp_hpar(NoOfGroups(),GetHyperPar()));
  // Calculate K matrix
  NEWMAT::SymmetricMatrix& K = give_me_a_K(); // Read/Write reference to _K in parent class.
  if (K.Nrows() != int(NoOfScans())) K.ReSize(NoOfScans());
  calculate_K_matrix(grpi(),NoOfGroups(),thpar(),angle_mat(),K);
  // Make sure it is positive definite
  validate_K_matrix();
} EddyCatch

std::vector<double> NewSphericalKMatrix::GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const EddyTry
{
  // Calculate group-wise variances (across directions) averaged over voxels
  std::vector<double> vars(NoOfGroups(),0.0);
  double var = 0.0;
  for (unsigned int i=0; i<NoOfGroups(); i++) {
    for (unsigned int j=0; j<data.size(); j++) {
      vars[i] += variance(data[j],grps()[i]);
    }
    vars[i] /= data.size();
    var += vars[i];
  }
  var /= NoOfGroups();

  // Make (semi-educated) guesses for hyper parameters based on findings in GP paper
  std::vector<double> hpar(n_par(NoOfGroups()));
  hpar[0] = 0.9*std::log(var/3.0);
  hpar[1] = 0.45;
  if (NoOfGroups() == 1) hpar[2] = std::log(var/3.0);
  else {
    hpar[2] = 0.0;
    double delta = 0.2 / (NoOfGroups()-1);
    for (unsigned int i=0; i<NoOfGroups(); i++) hpar[3+i] = (1.1 - i*delta) * std::log(var/3.0);
  }
  // Make sure they result in a valid K matrix
  if (!valid_hpars(hpar)) throw EddyException("NewSphericalKMatrix::GetHyperParGuess: Unable to find valid hyperparameters");

  return(hpar);
} EddyCatch

void NewSphericalKMatrix::SetHyperPar(const std::vector<double>& hpar) EddyTry
{
  MultiShellKMatrix::SetHyperPar(hpar);
  // if (std::exp(hpar[1]) > 3.141592653589793 || (NoOfGroups() > 1 && std::abs(hpar[2]) > 1.0)) set_K_matrix_invalid();

  return;
} EddyCatch

void NewSphericalKMatrix::MulErrVarBy(double ff) EddyTry
{
  std::vector<double> hpar = GetHyperPar();
  double *ev = NULL;
  ev = (NoOfGroups() > 1) ? &(hpar[3]) : &(hpar[2]);
  for (unsigned int i=0; i<NoOfGroups(); i++) ev[i] += std::log(ff);
  set_hpar(hpar);
  set_thpar(exp_hpar(NoOfGroups(),GetHyperPar()));
} EddyCatch

void NewSphericalKMatrix::calculate_K_matrix(const std::vector<unsigned int>& grpi,
					     unsigned int                     ngrp,
					     const std::vector<double>&       thpar,
					     const NEWMAT::SymmetricMatrix&   angle_mat,
					     NEWMAT::SymmetricMatrix&         K) const EddyTry
{
  if (K.Nrows() != angle_mat.Nrows()) K.ReSize(angle_mat.Nrows());
  // First pass for angular covariance
  double sm = thpar[0]; double a = thpar[1];
  for (int j=0; j<K.Ncols(); j++) {
    for (int i=j; i<K.Nrows(); i++) {
      double th = angle_mat(i+1,j+1);
      if (a>th) K(i+1,j+1) = sm * (1.0 - 1.5*th/a + 0.5*(th*th*th)/(a*a*a));
      else K(i+1,j+1) = 0.0;
    }
  }
  // Second pass for b-value covariance
  if (ngrp > 1) {
    std::vector<double> log_grpb = grpb(); for (unsigned int i=0; i<grpb().size(); i++) log_grpb[i] = std::log(grpb()[i]);
    double l = thpar[2];
    for (int j=0; j<K.Ncols(); j++) {
      for (int i=j+1; i<K.Nrows(); i++) {
	double bvdiff = log_grpb[grpi[i]] - log_grpb[grpi[j]];
	if (bvdiff) K(i+1,j+1) *= std::exp(-(bvdiff*bvdiff) / (2*l*l));
      }
    }
  }
  // Third pass for error variances
  const double *ev = NULL;
  ev = (ngrp > 1) ? &(thpar[3]) : &(thpar[2]);
  for (int i=0; i<K.Ncols(); i++) {
    K(i+1,i+1) += ev[grpi[i]];
  }

  return;
} EddyCatch

void NewSphericalKMatrix::calculate_dK_matrix(const std::vector<unsigned int>& grpi,
					      unsigned int                     ngrp,
					      const std::vector<double>&       thpar,
					      const NEWMAT::SymmetricMatrix&   angle_mat,
					      unsigned int                     gi,
					      unsigned int                     gj,
					      unsigned int                     off,
					      NEWMAT::SymmetricMatrix&         dK) const EddyTry
{
  throw EddyException("NewSphericalKMatrix::calculate_dK_matrix: NYI");
  return;
} EddyCatch

NEWMAT::RowVector NewSphericalKMatrix::k_row(unsigned int                     indx,
					     bool                             excl,
					     const std::vector<unsigned int>& grpi,
					     unsigned int                     ngrp,
					     const std::vector<double>&       thpar,
					     const NEWMAT::SymmetricMatrix&   angle_mat) const EddyTry
{
  NEWMAT::RowVector kr(angle_mat.Ncols());
  // First pass for angular covariance
  double sm = thpar[0]; double a = thpar[1];
  for (int j=0; j<angle_mat.Ncols(); j++) {
    double th = angle_mat(indx+1,j+1);
    if (a>th) kr(j+1) = sm * (1.0 - 1.5*th/a + 0.5*(th*th*th)/(a*a*a));
    else kr(j+1) = 0.0;
  }
  // Second pass for b-value covariance
  if (ngrp > 1) {
    std::vector<double> log_grpb = grpb(); for (unsigned int i=0; i<grpb().size(); i++) log_grpb[i] = std::log(grpb()[i]);
    double l = thpar[2]; double log_b = log_grpb[grpi[indx]];
    for (int j=0; j<angle_mat.Ncols(); j++) {
      double bvdiff = log_grpb[grpi[j]] - log_b;
      if (bvdiff) kr(j+1) *= std::exp(-(bvdiff*bvdiff) / (2*l*l));
    }
  }
  if (excl) kr = kr.Columns(1,indx) | kr.Columns(indx+2,angle_mat.Ncols());
  return(kr);
} EddyCatch

std::vector<double> NewSphericalKMatrix::get_arbitrary_hpar(unsigned int ngrp) const EddyTry
{
  std::vector<double> hpar(n_par(ngrp));
  if (ngrp == 1) { hpar[0] = 1.0; hpar[1] = 0.0; hpar[2] = 1.0; }
  else {
    hpar[0] = 1.0; hpar[1] = 0.0; hpar[2] = 0.0;
    for (unsigned int i=0; i<ngrp; i++) hpar[3+i] = 1.0;
  }
  return(hpar);
} EddyCatch

SphericalKMatrix::SphericalKMatrix(const std::vector<DiffPara>&          dpars,
				   bool                                  dcsh) EddyTry
: MultiShellKMatrix(dpars,dcsh)
{
  // Set (arbitrary) hyper-parameters that will at least yield a valid K (Gram) matrix
  set_hpar(get_arbitrary_hpar(NoOfGroups()));
  // Transform variance related hyperparameters
  set_thpar(exp_hpar(NoOfGroups(),GetHyperPar()));
  // Calculate K matrix
  NEWMAT::SymmetricMatrix& K = give_me_a_K(); // Read/Write reference to _K in parent class.
  if (K.Nrows() != int(NoOfScans())) K.ReSize(NoOfScans());
  calculate_K_matrix(grpi(),NoOfGroups(),thpar(),angle_mat(),K);
  // Make sure it is positive definite
  validate_K_matrix();
} EddyCatch

std::vector<double> SphericalKMatrix::GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const EddyTry
{
  // Calculate group-wise variances (across directions) averaged over voxels
  std::vector<double> vars(NoOfGroups(),0.0);
  for (unsigned int i=0; i<NoOfGroups(); i++) {
    for (unsigned int j=0; j<data.size(); j++) {
      vars[i] += variance(data[j],grps()[i]);
    }
    vars[i] /= data.size();
  }
  // Make (semi-educated) guesses for hyper parameters based on findings in GP paper
  std::vector<double> hpar(n_par(NoOfGroups()));
  double sn_wgt = 0.25;      // Determines the noise-variance
  unsigned int cntr = 0;     // Break-out counter to see how many adjustments has been made
  for (cntr=0; cntr<100; cntr++) { // Gradually adjust hyp-pars until they yield a valid K-matrix
    for (unsigned int j=0; j<NoOfGroups(); j++) {
      for (unsigned int i=j; i<NoOfGroups(); i++) {
	unsigned int pi = ij_to_parameter_index(i,j,NoOfGroups());
	if (i==j) {
	  hpar[pi] = std::log(vars[i] / 2.7);
	  hpar[pi+1] = 1.5;
	  hpar[pi+2] = std::log(sn_wgt * vars[i]);
	}
	else {
	  hpar[pi] = std::log(0.8 * std::min(vars[i],vars[j]) / 2.7);
	  hpar[pi+1] = 1.5;
	}
      }
    }
    if (!valid_hpars(hpar)) sn_wgt *= 1.1;
    else break;
  }
  if (cntr==100) { // If we just can't find valid hyperparamaters
    throw EddyException("SphericalKMatrix::GetHyperParGuess: Unable to find valid hyperparameters");
  }
  return(hpar);
} EddyCatch

unsigned int SphericalKMatrix::ij_to_parameter_index(unsigned int i,
						      unsigned int j,
						      unsigned int n) const EddyTry
{
  if (n==1) { // The one--three shell cases hardcoded for speed
    return(0);
  }
  else if (n==2) {
    if (!j) return(3*i);
    else return(5);
  }
  else if (n==3) {
    if (!j) { if (i) return(2*i+1); else return(0); }
    else if (j==1) return(4+3*i);
    else return(12); // for the i==j==2 case
  }
  else { // Not hardcoded. Would be nice if I could find a faster algorithm.
    unsigned int rval = 0;
    for (unsigned int ii=0; ii<j; ii++) { // Sum up index of preceeding columns
      rval += 1 + 2*(n-ii);
    }
    if (i>j) rval += 1 + 2*(i-j);
    return(rval);
  }
  return(0); // To stop compiler complaining.
} EddyCatch

void SphericalKMatrix::calculate_K_matrix(const std::vector<unsigned int>& grpi,
					  unsigned int                     ngrp,
					  const std::vector<double>&       thpar,
					  const NEWMAT::SymmetricMatrix&   angle_mat,
					  NEWMAT::SymmetricMatrix&         K) const EddyTry
{
  if (K.Nrows() != angle_mat.Nrows()) K.ReSize(angle_mat.Nrows());
  for (int j=0; j<K.Ncols(); j++) {
    for (int i=j; i<K.Nrows(); i++) {
      unsigned int pindx = ij_to_parameter_index(grpi[i],grpi[j],ngrp);
      double sm = thpar[pindx]; double a = thpar[pindx+1]; double th = angle_mat(i+1,j+1);
      if (a>th) K(i+1,j+1) = sm * (1.0 - 1.5*th/a + 0.5*(th*th*th)/(a*a*a));
      else K(i+1,j+1) = 0.0;
      if (i==j) K(i+1,j+1) += thpar[pindx+2];
    }
  }
  return;
} EddyCatch

void SphericalKMatrix::calculate_dK_matrix(const std::vector<unsigned int>& grpi,
					   unsigned int                     ngrp,
					   const std::vector<double>&       thpar,
					   const NEWMAT::SymmetricMatrix&   angle_mat,
					   unsigned int                     gi,
					   unsigned int                     gj,
					   unsigned int                     off,
					   NEWMAT::SymmetricMatrix&         dK) const EddyTry
{
  unsigned int pindx = ij_to_parameter_index(gi,gj,ngrp);
  double a = thpar[pindx+1]; double a2 = a*a; double a3 = a*a2; double a4 = a*a3;
  dK = 0;
  for (int j=0; j<dK.Ncols(); j++) {
    for (int i=j; i<dK.Nrows(); i++) {
      if (grpi[i] == gi && grpi[j] == gj) { // If this entry pertains to the parameter we want derivative wrt
	double sm = thpar[pindx]; double th = angle_mat(i+1,j+1);
	if (off == 0) { // If variable denoted \sigma_m in paper
	  if (a>th) dK(i+1,j+1) = sm * (1.0 - 1.5*th/a + 0.5*(th*th*th)/a3);
	}
	else if (off == 1) { // If variable denoted a in paper
	  if (a>th) dK(i+1,j+1) = sm * (1.5*th/a2 - 1.5*(th*th*th)/a4);
	}
	else if (off == 2 && i==j) { // If variable denoted \sigma_n in paper
	  dK(i+1,j+1) = thpar[pindx+2];
	}
      }
    }
  }
  return;
} EddyCatch

NEWMAT::RowVector SphericalKMatrix::k_row(unsigned int                     indx,
					  bool                             excl,
					  const std::vector<unsigned int>& grpi,
					  unsigned int                     ngrp,
					  const std::vector<double>&       thpar,
					  const NEWMAT::SymmetricMatrix&   angle_mat) const EddyTry
{
  NEWMAT::RowVector kr(angle_mat.Ncols());
  for (int j=0; j<angle_mat.Ncols(); j++) {
    unsigned int pindx = ij_to_parameter_index(grpi[indx],grpi[j],ngrp);
    double sm = thpar[pindx]; double a = thpar[pindx+1]; double th = angle_mat(indx+1,j+1);
    if (a>th) kr(j+1) = sm * (1.0 - 1.5*th/a + 0.5*(th*th*th)/(a*a*a));
    else kr(j+1) = 0.0;
  }
  if (excl) kr = kr.Columns(1,indx) | kr.Columns(indx+2,angle_mat.Ncols());
  return(kr);
} EddyCatch

std::vector<double> SphericalKMatrix::exp_hpar(unsigned int               ngrp,
					       const std::vector<double>& hpar) const EddyTry
{
  std::vector<double> thpar = hpar;
  for (unsigned int j=0; j<ngrp; j++) {
    for (unsigned int i=j; i<ngrp; i++) {
      unsigned int pi = ij_to_parameter_index(i,j,ngrp);
      thpar[pi] = exp(thpar[pi]);
      if (i==j) thpar[pi+2] = exp(thpar[pi+2]);
    }
  }
  return(thpar);
} EddyCatch

std::vector<double> SphericalKMatrix::get_arbitrary_hpar(unsigned int ngrp) const EddyTry
{
  std::vector<double> hpar(n_par(ngrp));
  for (unsigned int j=0; j<ngrp; j++) {
    for (unsigned int i=j; i<ngrp; i++) {
      unsigned int pi = ij_to_parameter_index(i,j,ngrp);
      if (i==j) {
	hpar[pi] = 1.0;
	hpar[pi+1] = 1.5;
	hpar[pi+2] = 1.0;
      }
      else {
	hpar[pi] = 0.5;
	hpar[pi+1] = 1.5;
      }
    }
  }
  return(hpar);
} EddyCatch

ExponentialKMatrix::ExponentialKMatrix(const std::vector<DiffPara>&          dpars,
				       bool                                  dcsh) EddyTry
: MultiShellKMatrix(dpars,dcsh)
{
  // Set (arbitrary) hyper-parameters that will at least yield a valid K (Gram) matrix
  set_hpar(get_arbitrary_hpar(NoOfGroups()));
  // Transform variance related hyperparameters
  set_thpar(exp_hpar(NoOfGroups(),GetHyperPar()));
  // Calculate K matrix
  NEWMAT::SymmetricMatrix& K = give_me_a_K(); // Read/Write reference to _K in parent class.
  if (K.Nrows() != int(NoOfScans())) K.ReSize(NoOfScans());
  calculate_K_matrix(grpi(),NoOfGroups(),thpar(),angle_mat(),K);
  // Make sure it is positive definite
  validate_K_matrix();
} EddyCatch

std::vector<double> ExponentialKMatrix::GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const EddyTry
{
  // Calculate group-wise variances (across directions) averaged over voxels
  std::vector<double> vars(NoOfGroups(),0.0);
  for (unsigned int i=0; i<NoOfGroups(); i++) {
    for (unsigned int j=0; j<data.size(); j++) {
      vars[i] += variance(data[j],grps()[i]);
    }
    vars[i] /= grps()[i].size();
  }
  // Make (semi-educated) guesses for hyper parameters based on findings in GP paper
  std::vector<double> hpar(n_par(NoOfGroups()));
  for (unsigned int j=0; j<NoOfGroups(); j++) {
    for (unsigned int i=j; i<NoOfGroups(); i++) {
      unsigned int pi = ij_to_parameter_index(i,j,NoOfGroups());
      if (i==j) {
	hpar[pi] = std::log(vars[i] / 2.0);
	hpar[pi+1] = 1.2;
	hpar[pi+2] = std::log(vars[i] / 4.0);
      }
      else {
	hpar[pi] = std::log(0.8 * std::min(vars[i],vars[j]) / 2.0);
	hpar[pi+1] = 1.2;
      }
    }
  }
  return(hpar);
} EddyCatch

unsigned int ExponentialKMatrix::ij_to_parameter_index(unsigned int i,
						       unsigned int j,
						       unsigned int n) const EddyTry
{
  if (n==1) { // The one--three shell cases hardcoded for speed
    return(0);
  }
  else if (n==2) {
    if (!j) return(3*i);
    else return(5);
  }
  else if (n==3) {
    if (!j) { if (i) return(2*i+1); else return(0); }
    else if (j==1) return(4+3*i);
    else return(12); // for the i==j==2 case
  }
  else { // Not hardcoded. Would be nice if I could find a faster algorithm.
    unsigned int rval = 0;
    for (unsigned int ii=0; ii<j; ii++) { // Sum up index of preceeding columns
      rval += 1 + 2*(n-ii);
    }
    if (i>j) rval += 1 + 2*(i-j);
    return(rval);
  }
  return(0); // To stop compiler complaining.
} EddyCatch

void ExponentialKMatrix::calculate_K_matrix(const std::vector<unsigned int>& grpi,
					  unsigned int                     ngrp,
					  const std::vector<double>&       thpar,
					  const NEWMAT::SymmetricMatrix&   angle_mat,
					  NEWMAT::SymmetricMatrix&         K) const EddyTry
{
  if (K.Nrows() != angle_mat.Nrows()) K.ReSize(angle_mat.Nrows());
  for (int j=0; j<K.Ncols(); j++) {
    for (int i=j; i<K.Nrows(); i++) {
      unsigned int pindx = ij_to_parameter_index(grpi[i],grpi[j],ngrp);
      K(i+1,j+1) = thpar[pindx] * exp(-angle_mat(i+1,j+1)/thpar[pindx+1]);
      if (i==j) K(i+1,j+1) += thpar[pindx+2];
    }
  }
  return;
} EddyCatch

void ExponentialKMatrix::calculate_dK_matrix(const std::vector<unsigned int>& grpi,
					     unsigned int                     ngrp,
					     const std::vector<double>&       thpar,
					     const NEWMAT::SymmetricMatrix&   angle_mat,
					     unsigned int                     gi,
					     unsigned int                     gj,
					     unsigned int                     off,
					     NEWMAT::SymmetricMatrix&         dK) const EddyTry
{
  unsigned int pindx = ij_to_parameter_index(gi,gj,ngrp);
  for (int j=0; j<dK.Ncols(); j++) {
    for (int i=j; i<dK.Nrows(); i++) {
      if (grpi[i] == gi && grpi[j] == gj) { // If this entry pertains to the parameter we want derivative wrt
	if (off == 0) { // If variable denoted \sigma_m in paper
	  dK(i+1,j+1) = thpar[pindx] * exp(-angle_mat(i+1,j+1)/thpar[pindx+1]);
	}
	else if (off == 1) { // If variable denoted a in paper
	  double a = thpar[pindx+1]; double th = angle_mat(i+1,j+1);
	  dK(i+1,j+1) = th * thpar[pindx] * exp(-th/a) / (a*a);
	}
	else if (off == 2) { // If variable denoted \sigma_n in paper
	  if (i==j) dK(i+1,j+1) = thpar[pindx+2];
	  else dK(i+1,j+1) = 0.0;
	}
      }
    }
  }
  return;
} EddyCatch

NEWMAT::RowVector ExponentialKMatrix::k_row(unsigned int                     indx,
					    bool                             excl,
					    const std::vector<unsigned int>& grpi,
					    unsigned int                     ngrp,
					    const std::vector<double>&       thpar,
					    const NEWMAT::SymmetricMatrix&   angle_mat) const EddyTry
{
  NEWMAT::RowVector kr(angle_mat.Ncols());
  for (int j=0; j<angle_mat.Ncols(); j++) {
    unsigned int pindx = ij_to_parameter_index(grpi[indx],grpi[j],ngrp);
    kr(j+1) = thpar[pindx] * exp(-angle_mat(indx+1,j+1)/thpar[pindx+1]);
  }
  if (excl) kr = kr.Columns(1,indx) | kr.Columns(indx+2,angle_mat.Ncols());
  return(kr);
} EddyCatch

std::vector<double> ExponentialKMatrix::exp_hpar(unsigned int               ngrp,
						 const std::vector<double>& hpar) const EddyTry
{
  std::vector<double> thpar = hpar;
  for (unsigned int j=0; j<ngrp; j++) {
    for (unsigned int i=j; i<ngrp; i++) {
      unsigned int pi = ij_to_parameter_index(i,j,ngrp);
      thpar[pi] = exp(thpar[pi]);
      if (i==j) thpar[pi+2] = exp(thpar[pi+2]);
    }
  }
  return(thpar);
} EddyCatch

std::vector<double> ExponentialKMatrix::get_arbitrary_hpar(unsigned int ngrp) const EddyTry
{
  std::vector<double> hpar(n_par(ngrp));
  for (unsigned int j=0; j<ngrp; j++) {
    for (unsigned int i=j; i<ngrp; i++) {
      unsigned int pi = ij_to_parameter_index(i,j,ngrp);
      if (i==j) {
	hpar[pi] = 1.0;
	hpar[pi+1] = 1.2;
	hpar[pi+2] = 1.0;
      }
      else {
	hpar[pi] = 0.5;
	hpar[pi+1] = 1.2;
      }
    }
  }
  return(hpar);
} EddyCatch
