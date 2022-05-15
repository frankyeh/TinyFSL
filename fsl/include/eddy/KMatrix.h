/*! \file KMatrix.h
    \brief Contains declaration of virtual base class and a derived class for Covariance matrices for GP

    \author Jesper Andersson
    \version 1.0b, Oct., 2013.
*/
// Declarations of virtual base class for
// Covariance matrices for DWI data.
//
// KMatrix.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford
//

#ifndef KMatrix_h
#define KMatrix_h

#include <cstdlib>
#include <string>
#include <exception>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"

namespace EDDY {

/****************************************************************//**
*
* \brief Virtual base class for classes implementing Covariance
* matrices (Gram matrices) for Gaussian processes for diffusion data.
*
* This virtual base class has many non-virtual methods that perform
* a very significant of the work of any derived class. It is only
* a few private methods that are virtual. These are the ones that
* implement the details of how the hyperparameters are translated
* into a K-matrix (Gram matrix).
* All indicies in the interface are zero-offset.
********************************************************************/
class KMatrix
{
public:
  KMatrix() {}
  virtual ~KMatrix() {}
  /// Returns a pointer to a clone of self.
  virtual std::shared_ptr<KMatrix> Clone() const = 0;
  /// Multiplication of K^{-1} with column vector
  virtual NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) const = 0;
  /// Multiplication of K^{-1} with column vector
  virtual NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) = 0;
  /// Get prediction vector for the ith scan.
  virtual NEWMAT::RowVector PredVec(unsigned int i, bool excl) const = 0;
  /// Get prediction vector for the ith scan.
  virtual NEWMAT::RowVector PredVec(unsigned int i, bool excl) = 0;
  /// Get variance of prediction for the ith scan
  virtual double PredVar(unsigned int i, bool excl) = 0;
  /// Get error variance of the ith scan
  virtual double ErrVar(unsigned int i) const = 0;
  /// Returns K^{-1}
  virtual const NEWMAT::SymmetricMatrix& iK() const = 0;
  /// Reset (to state after construction by default constructor)
  virtual void Reset() = 0;
  /// Set diffusion parameters
  virtual void SetDiffusionPar(const std::vector<DiffPara>& dpars) = 0;
  /// Return indicies for calculating means
  virtual const std::vector<std::vector<unsigned int> >& GetMeanIndicies() const = 0;
  /// Set hyperparameters
  virtual void SetHyperPar(const std::vector<double>& hpar) = 0;
  /// Multiply error-variance hyperpar by factor > 1.0
  virtual void MulErrVarBy(double ff) = 0;
  /// Pre-calculate inverse to allow for (thread safe) const version of PredVec
  virtual void CalculateInvK() = 0;
  /// Get starting guess for hyperparameters based on data
  virtual std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const = 0;
  /// Get hyperparameters
  virtual const std::vector<double>& GetHyperPar() const = 0;
  /// Get No of Scans
  virtual unsigned int NoOfScans() const = 0;
  /// Get No of hyperparameters
  virtual unsigned int NoOfHyperPar() const = 0;
  /// Check for validity (i.e. positive defitness)
  virtual bool IsValid() const = 0;
  /// Returns log of determinant of K
  virtual double LogDet() const = 0;
  /// Returns derivative w.r.t. ith hyperparameter.
  virtual NEWMAT::SymmetricMatrix GetDeriv(unsigned int i) const = 0;
  /// Returns derivatives w.r.t. all hyperparameters
  virtual void GetAllDerivs(std::vector<NEWMAT::SymmetricMatrix>& derivs) const = 0;
  /// Return K-matrix
  virtual const NEWMAT::SymmetricMatrix& AsNewmat() const = 0;
  /// Writes useful debug info to the screen
  virtual void Print() const = 0;
  /// Writes useful (and more bountiful) debug info to a file
  virtual void Write(const std::string& basefname) const = 0;
};

/****************************************************************//**
*
* \brief Virtual class for classes implementing Covariance
* matrices (Gram matrices) for Gaussian processes for "shelled"
* diffusion data, where shelled can be single or multi-shell data.
*
* This virtual class has many non-virtual methods that perform
* a very significant par of the work of any derived class. It is only
* a few private methods that are virtual. These are the ones that
* implement the details of how the hyperparameters are translated
* into a K-matrix (Gram matrix).
* Several methods have both const and non-const implementations.
* This is so that we shall have a set of const versions to be
* used with const objects in a thread safe way.
* All indicies in the interface are zero-offset.
********************************************************************/
class MultiShellKMatrix : public KMatrix
{
public:
  MultiShellKMatrix(bool dcsh) EddyTry : _ngrp(0), _K_ok(false), _iK_ok(false), _dcsh(dcsh) {} EddyCatch
  MultiShellKMatrix(const std::vector<DiffPara>&          dpars,
		    bool                                  dcsh);
  ~MultiShellKMatrix() {}
  /// Multiplication of K^{-1} with column vector
  NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y) const;
  /// Multiplication of K^{-1} with column vector
  NEWMAT::ColumnVector iKy(const NEWMAT::ColumnVector& y);
  /// Get prediction vector for the ith scan.
  NEWMAT::RowVector PredVec(unsigned int i, bool excl) const;
  /// Get prediction vector for the ith scan.
  NEWMAT::RowVector PredVec(unsigned int i, bool excl);
  /// Get variance of prediction for the ith scan
  double PredVar(unsigned int i, bool excl);
  /// Get error variance of the ith scan
  double ErrVar(unsigned int i) const EddyTry { return(err_var(i,_grpi,_ngrp,_thpar)); } EddyCatch
  /// Returns K^{-1}
  const NEWMAT::SymmetricMatrix& iK() const;
  /// Returns K^{-1}
  const NEWMAT::SymmetricMatrix& iK();
  /// Reset (to state after construction by default constructor)
  void Reset();
  /// Set diffusion parameters
  void SetDiffusionPar(const std::vector<DiffPara>& dpars);
  /// Return indicies for calculating means
  const std::vector<std::vector<unsigned int> >& GetMeanIndicies() const EddyTry { return(_grps); } EddyCatch
  /// Set hyperparameters
  virtual void SetHyperPar(const std::vector<double>& hpar);
  /// Multiply error-variance hyperpar by factor > 1.0
  virtual void MulErrVarBy(double ff);
  /// Pre-calculate inverse to allow for (thread safe) const version of PredVec
  void CalculateInvK();
  /// Get starting guess for hyperparameters based on data
  virtual std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const = 0;
  /// Get hyperparameters
  const std::vector<double>& GetHyperPar() const EddyTry { return(_hpar); } EddyCatch
  /// Check for validity (i.e. positive definitness)
  bool IsValid() const { return(_K_ok); }
  /// Returns the # of scans
  unsigned int NoOfScans() const { return(_dpars.size()); }
  /// Returns the # of groups (shells)
  const unsigned int& NoOfGroups() const { return(_ngrp); }
  /// Returns the # of hyperparameters
  unsigned int NoOfHyperPar() const { return(_hpar.size()); }
  /// Returns log of determinant of K
  double LogDet() const;
  /// Returns derivative w.r.t. ith hyperparameter.
  NEWMAT::SymmetricMatrix GetDeriv(unsigned int di) const;
  /// Returns derivatives w.r.t. all hyperparameters
  void GetAllDerivs(std::vector<NEWMAT::SymmetricMatrix>& derivs) const;
  /// Return K-matrix
  const NEWMAT::SymmetricMatrix& AsNewmat() const EddyTry { return(_K); } EddyCatch
  /// Print out some information that can be useful for debugging
  void Print() const;
  /// Writes useful (and more bountiful) debug info to a file
  void Write(const std::string& basefname) const;
protected:
  std::pair<unsigned int, unsigned int> parameter_index_to_ij(unsigned int pi,
							      unsigned int ngrp) const;
  /*
  unsigned int ij_to_parameter_index(unsigned int i,
				     unsigned int j,
				     unsigned int n) const;
  unsigned int n_par(unsigned int ngrp) const { return(_npps*ngrp + (_npps-1)*(ngrp*ngrp - ngrp)/2); }
  */
  double variance(const NEWMAT::ColumnVector&      data,
		  const std::vector<unsigned int>& indx) const;
  /// Allow derived class to set _hpar
  void set_hpar(const std::vector<double>& hpar) EddyTry { _hpar = hpar; } EddyCatch
  /// Allow derived class to set _thpar
  void set_thpar(const std::vector<double>& thpar) EddyTry { _thpar = thpar; } EddyCatch
  /// Allow derived class to set _npps
  // void set_npps(unsigned int npps) { _npps = npps; }
  /// Give full access to _K to derived class.
  NEWMAT::SymmetricMatrix& give_me_a_K() EddyTry { if (_K_ok) return(_K); else throw EddyException("MultiShellKMatrix::give_me_a_K: invalid K"); } EddyCatch
  /// Allow derived class to validate _K
  void validate_K_matrix();
  /// Allow derived classes to explicitly invalidate K matrix
  void set_K_matrix_invalid() EddyTry { _K_ok = false; _iK_ok = false; _pv_ok.assign(_dpars.size(),false); } EddyCatch
  /// Enable derived class to check matrix for validity (positive defitness).
  bool valid_hpars(const std::vector<double>& hpar) const;
  /// Give derived class read access to selected members
  const std::vector<std::vector<unsigned int> >& grps() const EddyTry { return(_grps); } EddyCatch
  const std::vector<unsigned int>& grpi() const EddyTry { return(_grpi); } EddyCatch
  const std::vector<double>& grpb() const EddyTry { return(_grpb); } EddyCatch
  const std::vector<double>& thpar() const EddyTry { return(_thpar); } EddyCatch
  const NEWMAT::SymmetricMatrix& angle_mat() const EddyTry { return(_angle_mat); } EddyCatch
private:
  std::vector<DiffPara>                     _dpars;      // Diffusion parameters
  std::vector<unsigned int>                 _grpi;       // Array of group (shell) indicies
  std::vector<std::vector<unsigned int> >   _grps;       // Arrays of indicies into _dpars indicating groups
  std::vector<double>                       _grpb;       // Array of group-mean b-values
  unsigned int                              _ngrp;       // Number of groups (shells)
  std::vector<double>                       _hpar;       // Hyperparameters
  std::vector<double>                       _thpar;      // "Transformed" (for example exponentiated) hyperparameters
  NEWMAT::SymmetricMatrix                   _angle_mat;  // Matrix of angles between scans.
  bool                                      _K_ok;       // True if K has been confirmed to be positive definite
  NEWMAT::SymmetricMatrix                   _K;          // K
  NEWMAT::LowerTriangularMatrix             _cK;         // L from a Cholesky decomp of K
  bool                                      _iK_ok;      // True if K^{-1} has been calculated
  NEWMAT::SymmetricMatrix                   _iK;         // K^{-1}
  std::vector<NEWMAT::RowVector>            _pv;         // Array of prediction vectors for the case of exclusion.
  std::vector<bool>                         _pv_ok;      // Indicates which prediction vectors are valid at any time
  bool                                      _dcsh;       // Don't check that data is shelled if set.

  virtual unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j,
					     unsigned int n) const = 0;
  virtual unsigned int n_par(unsigned int ngrp) const = 0;
  virtual void calculate_K_matrix(const std::vector<unsigned int>& grpi,
				  unsigned int                     ngrp,
				  const std::vector<double>&       thpar,
				  const NEWMAT::SymmetricMatrix&   angle_mat,
				  NEWMAT::SymmetricMatrix&         K) const = 0;
  virtual void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
				   unsigned int                     ngrp,
				   const std::vector<double>&       thpar,
				   const NEWMAT::SymmetricMatrix&   angle_mat,
				   unsigned int                     i,
				   unsigned int                     j,
				   unsigned int                     off,
				   NEWMAT::SymmetricMatrix&         dK) const = 0;
  virtual NEWMAT::RowVector k_row(unsigned int                     indx,
				  bool                             excl,
				  const std::vector<unsigned int>& grpi,
				  unsigned int                     ngrp,
				  const std::vector<double>&       thpar,
				  const NEWMAT::SymmetricMatrix&   angle_mat) const = 0;
  virtual std::vector<double> exp_hpar(unsigned int               ngrp,
				       const std::vector<double>& hpar) const = 0;
  virtual std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const = 0;
  virtual double sig_var(unsigned int                     i,
			 const std::vector<unsigned int>& grpi,
			 unsigned int                     ngrp,
			 const std::vector<double>&       thpar) const = 0;
  virtual double err_var(unsigned int                     i,
			 const std::vector<unsigned int>& grpi,
			 unsigned int                     ngrp,
			 const std::vector<double>&       thpar) const = 0;
  void calculate_iK();
  NEWMAT::SymmetricMatrix calculate_iK_index(unsigned int i) const;
  NEWMAT::Matrix make_pred_vec_matrix(bool excl=false) const;
  void make_angle_mat();
  double mean(const NEWMAT::ColumnVector&      data,
	      const std::vector<unsigned int>& indx) const;
  double sqr(double a) const { return(a*a); }
};

/****************************************************************//**
*
* \brief Concrete descendant of virtual MultiShellKMatrix class
* implementing the spherical covariance function.
*
* This class implements exactly the same model as SphericalKMatrix
* (below) in the single shell-case. However, for the multi-shell case
* it is different and unlike that it is "guaranteed" to always yield
* a valid Gram-matrix. Will probably replace SphericalKMatrix.
* The number of hyperparameters is 3 for the single shell case
* and 3+no_of_shells for the multi-shell case.
* The hyperparameters are ordered as:
* Single shell case:
* [log(signal_variance) log(angular_length_scale) log(error_variance)]
* Multi shell case:
* [log(signal_variance) log(angular_length_scale) log(b_length_scale)
*  log(error_variance_shell_1) log(error_variance_shell_2) ...]
*
* All indicies in the interface are zero-offset.
********************************************************************/
class NewSphericalKMatrix : public MultiShellKMatrix
{
public:
  NewSphericalKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  NewSphericalKMatrix(const std::vector<DiffPara>&          dpars,
		      bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<NewSphericalKMatrix>(new NewSphericalKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
  void SetHyperPar(const std::vector<double>& hpar);
  void MulErrVarBy(double ff);
private:
  unsigned int n_par(unsigned int ngrp) const { return(ngrp==1 ? 3 : 3 + ngrp); }
  unsigned int ij_to_parameter_index(unsigned int i,
				     unsigned int j,
				     unsigned int n) const EddyTry { throw EddyException("NewSphericalKMatrix::ij_to_parameter_index: Invalid call"); } EddyCatch
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const EddyTry
  { std::vector<double> epar = hpar; for (unsigned int i=0; i<hpar.size(); i++) epar[i] = std::exp(hpar[i]); return(epar); } EddyCatch
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[0]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { const double *ev = (ngrp > 1) ? &(thpar[3]) : &(thpar[2]); return(ev[grpi[i]]); } EddyCatch
};

/****************************************************************//**
*
* \brief Concrete descendant of virtual MultiShellKMatrix class
* implementing the spherical covariance function.
*
* All indicies in the interface are zero-offset.
********************************************************************/
class SphericalKMatrix : public MultiShellKMatrix
{
public:
  SphericalKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  SphericalKMatrix(const std::vector<DiffPara>&          dpars,
		   bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<SphericalKMatrix>(new SphericalKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
private:
  unsigned int n_par(unsigned int ngrp) const { return(3*ngrp + (ngrp*ngrp - ngrp)); }
  unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j,
					     unsigned int n) const;
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const;
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)+2]); } EddyCatch
};

/****************************************************************//**
*
* \brief Concrete descendant of virtual MultiShellKMatrix class
* implementing the exponential covariance function.
*
* All indicies in the interface are zero-offset.
********************************************************************/
class ExponentialKMatrix : public MultiShellKMatrix
{
public:
  ExponentialKMatrix(bool dcsh=false) EddyTry : MultiShellKMatrix(dcsh) {} EddyCatch
  ExponentialKMatrix(const std::vector<DiffPara>&          dpars,
		     bool                                  dcsh=false);
  std::shared_ptr<KMatrix> Clone() const EddyTry { return(std::shared_ptr<ExponentialKMatrix>(new ExponentialKMatrix(*this))); } EddyCatch
  std::vector<double> GetHyperParGuess(const std::vector<NEWMAT::ColumnVector>& data) const;
private:
  unsigned int n_par(unsigned int ngrp) const { return(3*ngrp + (ngrp*ngrp - ngrp)); }
  unsigned int ij_to_parameter_index(unsigned int i,
					     unsigned int j,
					     unsigned int n) const;
  void calculate_K_matrix(const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat,
			  NEWMAT::SymmetricMatrix&         K) const;
  void calculate_dK_matrix(const std::vector<unsigned int>& grpi,
			   unsigned int                     ngrp,
			   const std::vector<double>&       thpar,
			   const NEWMAT::SymmetricMatrix&   angle_mat,
			   unsigned int                     gi,
			   unsigned int                     gj,
			   unsigned int                     off,
			   NEWMAT::SymmetricMatrix&         dK) const;
  NEWMAT::RowVector k_row(unsigned int                     indx,
			  bool                             excl,
			  const std::vector<unsigned int>& grpi,
			  unsigned int                     ngrp,
			  const std::vector<double>&       thpar,
			  const NEWMAT::SymmetricMatrix&   angle_mat) const;
  std::vector<double> exp_hpar(unsigned int               ngrp,
			       const std::vector<double>& hpar) const;
  std::vector<double> get_arbitrary_hpar(unsigned int ngrp) const;
  double sig_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)]); } EddyCatch
  double err_var(unsigned int                     i,
		 const std::vector<unsigned int>& grpi,
		 unsigned int                     ngrp,
		 const std::vector<double>&       thpar) const EddyTry { return(thpar[ij_to_parameter_index(grpi[i],grpi[i],ngrp)+2]); } EddyCatch
};

} // End namespace EDDY

#endif // end #ifndef KMatrix_h
