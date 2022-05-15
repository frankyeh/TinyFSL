/*! \file ECModels.h
    \brief Contains declaration of classes that implements models for fields from eddy currents.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of classes that implements a hirearchy
// of models for fields from eddy currents induced by
// diffusion gradients.
//
// ECModels.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef ECModels_h
#define ECModels_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"

namespace EDDY {

class SliceDerivModulator
{
public:
  SliceDerivModulator() {}
  SliceDerivModulator(unsigned int n) : _mod(n), _set(n,false) {}
  SliceDerivModulator(const std::vector<float>& mod) : _mod(mod), _set(mod.size(),true) {}
  void SetMod(float val, unsigned int i) EddyTry { check(i,"SetMod"); _mod[i]=val; _set[i]=true; } EddyCatch
  const std::vector<float>& GetMod() const EddyTry
  {
    if (!this->AllSet()) throw EddyException("SliceDerivModulator::GetMod: Invalid object"); else return(_mod);
  } EddyCatch
  float GetMod(unsigned int i) const EddyTry { check(i,"SetMod"); return(_mod[i]); } EddyCatch
  bool AllSet() const EddyTry { return(all_of(_set.begin(),_set.end(),[](bool x)->bool{return(x);})); } EddyCatch
private:
  std::vector<float> _mod;
  std::vector<bool>  _set;
  void check(unsigned int i, const std::string& caller) const EddyTry {
    if (i >= _mod.size()) throw EddyException("SliceDerivModulator::"+caller+": Index out of range");
  } EddyCatch
};

class SpatialDerivModulator
{
public:
  SpatialDerivModulator() : _mod{1,1,1} {} // Default constructor yields invalid modulation
  SpatialDerivModulator(const std::vector<unsigned int>& mod) : _mod(mod) {}
  const std::vector<unsigned int>& GetModulation() const {
    if (std::all_of(_mod.begin(),_mod.end(),[](int i){return(i==1);})) throw EddyException("SpatialDerivModulator::GetModulation: Invalid modulation object");
    return(_mod);
  }
private:
  std::vector<unsigned int> _mod;
};

enum class ModulationType { SliceWise, Spatial };
class DerivativeInstructions
{
public:
  DerivativeInstructions(unsigned int primi, float prims, ModulationType mt, unsigned int nscnd)
  : _prim(primi,prims), _mt(mt), _scnd(nscnd), _set(nscnd,false) {}
  unsigned int GetPrimaryIndex() const { return(_prim._index); }
  float GetPrimaryScale() const { return(_prim._scale); }
  unsigned int NSecondary() const { return(_scnd.size()); }
  bool AllSet() const EddyTry { return(all_of(_set.begin(),_set.end(),[](bool x)->bool{return(x);})); } EddyCatch
  bool IsSliceMod() const { return(_mt == ModulationType::SliceWise); }
  bool IsSpatiallyMod() const { return(!this->IsSliceMod()); }
  unsigned int GetSecondaryIndex(unsigned int i) const EddyTry {
    this->check_index(i,"GetSecondaryIndex");
    if (this->AllSet()) return(_scnd[i]._index); else throw EddyException("DerivativeInstructions::GetSecondaryIndex: Invalid object");
  } EddyCatch
  SpatialDerivModulator GetSpatialModulator(unsigned int i) const EddyTry {
    this->check_index(i,"GetSpatialModulator");
    if (this->AllSet()) return(_scnd[i]._spmod); else throw EddyException("DerivativeInstructions::GetSpatialModulator: Invalid object");
  } EddyCatch
  SliceDerivModulator GetSliceModulator(unsigned int i) const EddyTry {
    this->check_index(i,"GetSliceModulator");
    if (this->AllSet()) return(_scnd[i]._slmod); else throw EddyException("DerivativeInstructions::GetSliceModulator: Invalid object");
  } EddyCatch
  void SetSecondary(unsigned int i, unsigned int index, const SliceDerivModulator& sdm);
  void SetSecondary(unsigned int i, unsigned int index, const SpatialDerivModulator& sdm);
  void AddIndexOffset(unsigned int offs) EddyTry {
    if (this->AllSet()) {
      _prim._index += offs;
      for (unsigned int i=0; i<this->NSecondary(); i++) _scnd[i]._index += offs;
    }
    else throw EddyException("DerivativeInstructions::AddIndexOffset: Invalid object");
  } EddyCatch
private:
  struct Primary
  {
    Primary(unsigned int index, float scale) : _index(index), _scale(scale) {}
    unsigned int _index;
    float        _scale;
  };
  struct Secondary
  {
    unsigned int          _index;
    SliceDerivModulator   _slmod;
    SpatialDerivModulator _spmod;
  };
private:
  Primary                 _prim;
  ModulationType          _mt;
  std::vector<Secondary>  _scnd;
  std::vector<bool>       _set;
  void check_index(unsigned int i, const std::string& caller) const EddyTry {
    if (i>=_scnd.size()) throw EddyException("DerivativeInstructions::"+caller+": Index out of range");
  } EddyCatch
};

/****************************************************************//**
*
* \brief Class that is used to manage the movement model/parameters
* for the eddy project.
*
* The movement model is associated with an order, such that an order
* of zero indicates a single movement parameter per degree of
* freedom, i.e. the traditional rigid-body model. For higher orders
* the movement is modeled as a basis-function expansion over time,
* using an unnormalised discrete Cosine transform as a basis set.
*
********************************************************************/
class ScanMovementModel
{
public:
  ScanMovementModel(unsigned int order) EddyTry : _order(order), _mp(static_cast<int>(6*(order+1))) { _mp=0.0; } EddyCatch
  ScanMovementModel(unsigned int                 order,
		    const NEWMAT::ColumnVector&  mp) EddyTry : _order(order), _mp(mp) {
    if (mp.Nrows() != static_cast<int>(6*(order+1))) throw EddyException("ScanMovementModel::ScanMovementModel: Mismatch between order and mp");
  } EddyCatch
  ~ScanMovementModel() {}
  unsigned int Order() const { return(_order); }
  bool IsSliceToVol() const { return(_order!=0); }
  unsigned int NParam() const { return(6*(_order+1)); }
  NEWMAT::ColumnVector GetZeroOrderParams() const EddyTry { return(get_zero_order_mp()); } EddyCatch
  NEWMAT::ColumnVector GetParams() const EddyTry { return(_mp); } EddyCatch
  double GetParam(unsigned int indx) const EddyTry {
    if (int(indx) > _mp.Nrows()) throw EddyException("ScanMovementModel::GetParam: indx out of range");
    return(_mp(indx+1));
  } EddyCatch
  /// Get the six RB movement parameters for one group
  NEWMAT::ColumnVector GetGroupWiseParams(unsigned int grp, unsigned int ngrp) const EddyTry { return(get_gmp(grp,ngrp)); } EddyCatch
  /// Sets the movement parameter. If order>0 and p.Nrows()==6 the zero-order (const) params will be set.
  void SetParams(const NEWMAT::ColumnVector& p) EddyTry {
    if (p.Nrows() == 6) set_zero_order_mp(p);
    else if (p.Nrows() == _mp.Nrows()) _mp = p;
    else throw EddyException("ScanMovementModel::SetParams: mismatched p");
  } EddyCatch
  /// Set the parameter indicated by indx
  void SetParam(unsigned int indx, double val) EddyTry {
    if (int(indx) > _mp.Nrows()) throw EddyException("ScanMovementModel::SetParam: indx out of range");
    _mp(indx+1) = val;
  } EddyCatch
  /// Set parameters on a "per group" basis, which means the internal parameters (DCT coefs) have to be calculated/fitted.
  void SetGroupWiseParameters(const NEWMAT::Matrix& gwmp) EddyTry {
    if (gwmp.Nrows() != 6) throw EddyException("ScanMovementModel::SetGroupWiseParameters: gwmp must have 6 rows");
    NEWMAT::Matrix X = get_design(static_cast<unsigned int>(gwmp.Ncols()));
    NEWMAT::Matrix Hat = (X.t()*X).i()*X.t();
    NEWMAT::ColumnVector dctc;
    for (int i=0; i<6; i++) {
      dctc &= Hat*gwmp.Row(i+1).t();
    }
    _mp = dctc;
  } EddyCatch
  /// Sets the order of movemnt model. If order less than previously the higher order components will be lost.
  void SetOrder(unsigned int order) EddyTry {
    NEWMAT::ColumnVector tmp(6*(order+1)); tmp=0.0;
    unsigned int cpsz = (order < _order) ? order : _order;
    for (int i=0; i<6; i++) {
      tmp.Rows(i*(order+1)+1,i*(order+1)+cpsz+1) = _mp.Rows(i*(_order+1)+1,i*(_order+1)+cpsz+1);
    }
    _mp=tmp; _order=order;
  } EddyCatch
  /// Returns the number of parameters that are being estimated
  unsigned int NDerivs() const { return(NParam()); }
  /// Returns sutiable scales for evaluating numerical derivatives
  double GetDerivScale(unsigned int dindx) const EddyTry {
    if (dindx>=6*(_order+1)) throw EddyException("ScanMovementModel::GetDerivScale: dindx out of range");
    return( (dindx<3*(_order+1)) ? 1e-2 : 1e-5 );
  } EddyCatch
  /// Returns the number of "compound" derivatives
  unsigned int NCompoundDerivs() const { return(6); }
  /// Returns the instructions for the ith compound derivative
  DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const EDDY::MultiBandGroups& mbg) const;
  /// Returns the Hessian for a Laplacian regularisation of the movement
  NEWMAT::Matrix GetHessian(unsigned int ngrp) const EddyTry {
    NEWMAT::Matrix hess(NDerivs(),NDerivs()); hess = 0.0;
    if (_order) {
      NEWMAT::DiagonalMatrix D(6);
      for (int i=0; i<3; i++) D(i+1) = 1.0;
      for (int i=3; i<6; i++) D(i+1) = 100.0;
      hess = NEWMAT::KP(D,get_design_derivative(ngrp,2).t() * get_design_derivative(ngrp,2));
    }
    return(hess);
  } EddyCatch
  /// Returns matrix denoted \mathbf{R} in paper.
  NEWMAT::Matrix ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan) const;
  /// Returns matrix denoted \mathbf{R} in paper for mb-group grp.
  NEWMAT::Matrix ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan, unsigned int grp, unsigned int ngrp) const;
  /// Returns matrix denoted \nathbf{R}^{-1} in paper.
  NEWMAT::Matrix InverseMovementMatrix(const NEWIMAGE::volume<float>& scan) const EddyTry { return(ForwardMovementMatrix(scan).i()); } EddyCatch
  /// Returns matrix denoted \nathbf{R}^{-1} in paper for mb-group grp.
  NEWMAT::Matrix InverseMovementMatrix(const NEWIMAGE::volume<float>& scan, unsigned int grp, unsigned int ngrp) const EddyTry {
    if (grp>=ngrp) throw EddyException("ScanMovementModel::InverseMovementMatrix: grp has to be smaller than ngrp");
    return(ForwardMovementMatrix(scan,grp,ngrp).i());
  } EddyCatch
  /// The same as ForwardMovementMatrix, but excluding some movement parameters as specified by rindx
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 const std::vector<unsigned int>&     rindx) const;
  /// The same as ForwardMovementMatrix, but excluding some movement parameters as specified by rindx
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 unsigned int                         grp,
						 unsigned int                         ngrp,
						 const std::vector<unsigned int>&     rindx) const;
  /// The same as InverseMovementMatrix, but excluding some movement parameters as specified by rindx
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 const std::vector<unsigned int>&     rindx) const EddyTry { return(RestrictedForwardMovementMatrix(scan,rindx).i()); } EddyCatch
  /// The same as InverseMovementMatrix, but excluding some movement parameters as specified by rindx
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const NEWIMAGE::volume<float>&       scan,
						 unsigned int                         grp,
						 unsigned int                         ngrp,
						 const std::vector<unsigned int>&     rindx) const EddyTry {
    if (grp>=ngrp) throw EddyException("ScanMovementModel::RestrictedInverseMovementMatrix: grp has to be smaller than ngrp");
    return(RestrictedForwardMovementMatrix(scan,grp,ngrp,rindx).i());
  } EddyCatch

private:
  unsigned int           _order; // Order of DCT-set
  /// Movement parameters organised as [xt_const xt_dct_1 ... xt_dct_order yt_const ... zr_dct_order]
  NEWMAT::ColumnVector   _mp;

  NEWMAT::ColumnVector get_zero_order_mp() const EddyTry {
    NEWMAT::ColumnVector zmp(6); zmp=0.0;
    for (int i=0, j=0; i<6; i++, j+=(int(_order)+1)) zmp(i+1) = _mp(j+1);
    return(zmp);
  } EddyCatch

  void set_zero_order_mp(const NEWMAT::ColumnVector& mp) EddyTry { for (int i=0, j=0; i<6; i++, j+=(int(_order)+1)) _mp(j+1) = mp(i+1); } EddyCatch

  NEWMAT::ColumnVector get_gmp(unsigned int grp, unsigned int ngrp) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::ColumnVector gmp(6); gmp=0.0;
    for (unsigned int i=0; i<6; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) gmp(i+1) += _mp(i*(_order+1)+j+1);
	else gmp(i+1) += _mp(i*(_order+1)+j+1) * cos((pi*double(j)*double(2*grp+1))/double(2*ngrp));
      }
    }
    return(gmp);
  } EddyCatch

  NEWMAT::Matrix get_design(unsigned int ngrp) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::Matrix X(ngrp,_order+1);
    for (unsigned int i=0; i<ngrp; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) X(i+1,j+1) = 1.0;
	else X(i+1,j+1) = cos((pi*double(j)*double(2*i+1))/double(2*ngrp));
      }
    }
    return(X);
  } EddyCatch

  NEWMAT::Matrix get_design_derivative(unsigned int ngrp, unsigned int dorder) const EddyTry {
    double pi = 3.141592653589793;
    NEWMAT::Matrix dX(ngrp,_order+1);
    for (unsigned int i=0; i<ngrp; i++) {
      for (unsigned int j=0; j<(_order+1); j++) {
	if (j==0) dX(i+1,j+1) = 0.0;
	else {
	  if (dorder==1) dX(i+1,j+1) = - (pi*double(j)/double(ngrp)) * sin((pi*double(j)*double(2*i+1))/double(2*ngrp));
	  else if (dorder==2) dX(i+1,j+1) = - this->sqr((pi*double(j)/double(ngrp))) * cos((pi*double(j)*double(2*i+1))/double(2*ngrp));
	  else throw EddyException("ScanMovementModel::get_design_derivative: Invalid derivative");
	}
      }
    }
    return(dX);
  } EddyCatch

  double sqr(double x) const { return(x*x); }
};

/****************************************************************//**
*
* \brief Virtual base class for classes used to model the fields
* that may result from eddy currents.
*
* The classes in this hierarchy manages eddy current (EC) parameters
* for one scan. We can set the parameters with one call and obtain
* the resulting field with another. By deriving a set of classes from
* a virtual base class we are able to use the same code to estimate
* the parameters for different EC-models.
*
********************************************************************/
class ScanECModel
{
public:
  ScanECModel() EddyTry {} EddyCatch
  ScanECModel(const NEWMAT::ColumnVector& ep) EddyTry : _ep(ep) {} EddyCatch
  virtual ~ScanECModel() {}
  /// Returns which model it is
  virtual ECModel WhichModel() const = 0;
  /// Indicates if a field offset is modeled or not.
  virtual bool HasFieldOffset() const = 0;
  /// Returns the field offset.
  virtual double GetFieldOffset() const = 0;
  /// Set field offset.
  virtual void SetFieldOffset(double ofst) = 0;
  /// Return the total number of parameters
  unsigned int NParam() const { return(_ep.Nrows()); }
  /// Get all parameters.
  NEWMAT::ColumnVector GetParams() const EddyTry { return(_ep); } EddyCatch
  /// Set all parameters.
  void SetParams(const NEWMAT::ColumnVector& ep) EddyTry {
    if (ep.Nrows() != _ep.Nrows()) throw EddyException("ScanECModel::SetParams: Wrong number of parameters");
    _ep = ep;
  } EddyCatch
  /// Return the number of parameters that are updated as part of the estimation.
  virtual unsigned int NDerivs() const = 0;
  /// Returns the number of compound derivatives
  virtual unsigned int NCompoundDerivs() const = 0;
  /// Returns instructions for the dindx'th compound derivative
  virtual EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const = 0;
  // The following get/set routines indexes parameters from
  // 0 - NDerivs()-1  i.e. it ignores any parameters that are
  // not estimated for the particular model.
  // 6/5-2020 I added a flag that allows one to index one past NDerivs-1 for the case where
  // NDerivs() < NParam(), i.e. when no susceptibility field was passed to eddy. This is to allow
  // estimation of derivatives using the modulation trick for those cases. It is not elegant, and
  // it might be worth considering a complete redesign as a consquence of the modulated derivatives.
  /// Get parameter dindx of 0 to NDerivs-1
  virtual double GetDerivParam(unsigned int dindx, bool allow_field_offset=false) const = 0;
  /// Set parameter dindx of 0 to NDerivs-1
  virtual void SetDerivParam(unsigned int dindx, double p, bool allow_field_offset=false) = 0;
  /// Returns sutiable scales for evaluating numerical derivatives
  virtual double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const = 0;
  /// Used to create a polymorphic copy of self
  virtual std::shared_ptr<ScanECModel> Clone() const = 0;
  /// Return eddy current-induced field in Hz. Denoted e(\mathbf{b}) in paper.
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const = 0;
protected:
  NEWMAT::ColumnVector _ep;
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class PolynomialScanECModel (Polynomial Scan Eddy Current Model)
//
// A virtual base class for polynomial models (linear, quadratic etc).
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class PolynomialScanECModel : public ScanECModel
{
public:
  PolynomialScanECModel(bool field=false) EddyTry : ScanECModel() {
    _nepd = 0;
  } EddyCatch
  PolynomialScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : ScanECModel(ep), _nepd(0) {} EddyCatch
  virtual ~PolynomialScanECModel() {}
  virtual ECModel WhichModel() const = 0;
  virtual bool HasFieldOffset() const = 0;
  virtual double GetFieldOffset() const EddyTry { if (HasFieldOffset()) return(_ep(_nepd)); else return(0.0); } EddyCatch
  virtual void SetFieldOffset(double ofst) EddyTry {
    if (!HasFieldOffset()) throw EddyException("PolynomialScanECModel::SetFieldOffset: Attempting to set offset for model without offset");
    _ep(_nepd) = ofst;
  } EddyCatch
  virtual NEWMAT::RowVector GetLinearParameters() const EddyTry { return(_ep.Rows(1,3).t()); } EddyCatch
  virtual unsigned int NDerivs() const { return(_nepd); }
  /// Returns the number of "compound" derivatives
  virtual unsigned int NCompoundDerivs() const = 0;
  /// Returns the instructions for the ith compound derivative
  virtual EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const = 0;
  virtual double GetDerivParam(unsigned int dindx, bool allow_field_offset=false) const EddyTry {
    if (dindx>=NDerivs() && !allow_field_offset) throw EddyException("PolynomialScanECModel::GetDerivParam(allow_field_offset=false): dindx out of range");
    else if (dindx>=NParam()) throw EddyException("PolynomialScanECModel::GetDerivParam(allow_field_offset=true): dindx out of range");
    return(_ep(dindx+1));
  } EddyCatch
  virtual void SetDerivParam(unsigned int dindx, double p, bool allow_field_offset=false) EddyTry {
    if (dindx>=NDerivs() && !allow_field_offset) throw EddyException("PolynomialScanECModel::SetDerivParam(allow_field_offset=false): dindx out of range");
    else if (dindx>=NParam()) throw EddyException("PolynomialScanECModel::SetDerivParam(allow_field_offset=true): dindx out of range");
    _ep(dindx+1) = p;
  } EddyCatch
  virtual double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const = 0;
  virtual std::shared_ptr<ScanECModel> Clone() const = 0;
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const = 0;
protected:
  unsigned int         _nepd;  // Number of Eddy Parameter Derivatives (might depend on if field was set)
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class LinearScanECModel (Linear Scan Eddy Current Model)
//
// This class models the eddy curents as a linear combination of
// linear gradients in the x-, y- and z-directions. The assumption
// behind this is that the eddy currents resides mainly in the
// gradient coils and also that the gradient coils are close to
// linear.
// The _ep field contains:
// dfdx (shear), dfdy (zoom), dfdz (z-dependent trans) all in Hz/mm and df (trans in Hz)
// Note that the translations (e.g. dfdx (EC gradient in x-direction) to shear)
// only makes sense if the phase-encode is in the y-direction.
// The df field is there to model any difference between the centre
// of the FOV and the iso-centre of the scanner.
//
// If no susceptibility off-resonance field is specified df is strictly
// speaking redundant since it is identical to a subject movement
// (y-translation). It is however useful to retain as a parameter
// for the sake of modelling EC parameters as a function of
// diffusion gradients. In the updates it will not have a derivative
// and will hence not be updated. It will instead be set by the
// higher level modelling of the parameters.
//
// If a susceptibility off-resonance field is specified df will enter
// into the transform compared to subject y-translation and it should
// be possible to directly separate the two. In this case it will
// have a derivative and will be updated as part of the first level
// estimation.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class LinearScanECModel : public PolynomialScanECModel
{
public:
  LinearScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(4); _ep=0.0;
    _nepd = (field) ? 4 : 3;
  } EddyCatch
  LinearScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep) {
    _nepd = (field) ? 4 : 3;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("LinearScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~LinearScanECModel() {}
  ECModel WhichModel() const { return(Linear); }
  bool HasFieldOffset() const { return(_nepd==4); }
  virtual std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new LinearScanECModel(*this))); } EddyCatch
  virtual unsigned int NCompoundDerivs() const { return(2); }
  virtual EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const;
  virtual double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const EddyTry {
    if (dindx>=NDerivs() && !allow_field_offset) throw EddyException("LinearScanECModel::GetDerivScale(allow_field_offset=false): dindx out of range");
    else if (dindx>=NParam()) throw EddyException("LinearScanECModel::GetDerivScale(allow_field_offset=true): dindx out of range");
    if (dindx < 3) return(1e-3);
    else return(1e-2);
    throw EddyException("LinearScanECModel::GetDerivScale: This should not be possible");
  } EddyCatch
  virtual NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class QuadraticScanECModel (Quadratic Scan Eddy Current Model)
//
// This class models the eddy curents as a second order polynomial
// expansion of  combination of gradients in the x-, y- and z-directions.
//
// The _ep field contains:
// dfdx (shear), dfdy (zoom), dfdz (z-dependent trans) all in Hz/mm
// followed by dfdx^2, dfdy^2, dfdz^2, dfdx*dfdy, dfdx*dfdz and dfdy*dfdz
// and finally df (trans in Hz).
// The quadratic components are (arbitrarily) scaled to have ~the same value (in Hz)
// at the edge of the FOV as has the linear terms. This is done to ensure an
// update matrix with reasonable condition number.
// Note that the translations (e.g. dfdx (EC gradient in x-direction) to shear)
// only makes sense if the phase-encode is in the y-direction.
// The df field is there to model any difference between the centre
// of the FOV and the iso-centre of the scanner.
//
// If no susceptibility off-resonance field is specified df is strictly
// speaking redundant since it is identical to a subject movement
// (y-translation). It is however useful to retain as a parameter
// for the sake of modelling EC parameters as a function of
// diffusion gradients. In the updates it will not have a derivative
// and will hence not be updated. It will instead be set by the
// higher level modelling of the parameters.
//
// If a susceptibility off-resonance field is specified df will enter
// into the transform compared to subject y-translation and it should
// be possible to directly separate the two. In this case it will
// have a derivative and will be updated as part of the first level
// estimation.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class QuadraticScanECModel : public PolynomialScanECModel
{
public:
  QuadraticScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(10); _ep=0.0;
    _nepd = (field) ? 10 : 9;
  } EddyCatch
  QuadraticScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep,field) {
    _nepd = (field) ? 10 : 9;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("QuadraticScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~QuadraticScanECModel() {}
  ECModel WhichModel() const { return(Quadratic); }
  bool HasFieldOffset() const { return(_nepd==10); }
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new QuadraticScanECModel(*this))); } EddyCatch
  virtual unsigned int NCompoundDerivs() const { return(3); }
  virtual EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const;
  virtual double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const EddyTry {
    if (dindx>=NDerivs() && !allow_field_offset) throw EddyException("QuadraticScanECModel::GetDerivScale(allow_field_offset=false): dindx out of range");
    else if (dindx>=NParam()) throw EddyException("QuadraticScanECModel::GetDerivScale(allow_field_offset=true): dindx out of range");
    if (dindx < 3) return(1e-3);
    else if (dindx < 9) return(1e-5);
    else return(1e-2);
    throw EddyException("QuadraticScanECModel::GetDerivScale: This should not be possible");
  } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class CubicScanECModel (Cubic Scan Eddy Current Model)
//
// This class models the eddy curents as a third order polynomial
// expansion of  combination of gradients in the x-, y- and z-directions.
//
// The _ep field contains:
// dfdx (shear), dfdy (zoom), dfdz (z-dependent trans) all in Hz/mm
// followed by dfdx^2, dfdy^2, dfdz^2, dfdx*dfdy, dfdx*dfdz and dfdy*dfdz,
// followed by dfdx^3, dfdy^3, dfdz^3, dfdx^2*dfdy, dfdx^2*dfdz,
// dfdy^2*dfdx, dfdy^2*dfdz, dfdz^2*dfdx, dfdz^2*dfdy, dfdx*dfdy*dfdz
//  and finally df (trans in Hz).
// The quadratic and cubic components are (arbitrarily) scaled to have ~the
// same value (in Hz) at the edge of the FOV as has the linear terms.
// This is done to ensure an update matrix with reasonable condition number.
// Note that the translations (e.g. dfdx (EC gradient in x-direction) to shear)
// only makes sense if the phase-encode is in the y-direction.
// The df field is there to model any difference between the centre
// of the FOV and the iso-centre of the scanner.
//
// If no susceptibility off-resonance field is specified df is strictly
// speaking redundant since it is identical to a subject movement
// (y-translation). It is however useful to retain as a parameter
// for the sake of modelling EC parameters as a function of
// diffusion gradients. In the updates it will not have a derivative
// and will hence not be updated. It will instead be set by the
// higher level modelling of the parameters.
//
// If a susceptibility off-resonance field is specified df will enter
// into the transform compared to subject y-translation and it should
// be possible to directly separate the two. In this case it will
// have a derivative and will be updated as part of the first level
// estimation.
//
// {{{ @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class CubicScanECModel : public PolynomialScanECModel
{
public:
  CubicScanECModel(bool field=false) EddyTry : PolynomialScanECModel(field) {
    _ep.ReSize(20); _ep=0.0;
    _nepd = (field) ? 20 : 19;
  } EddyCatch
  CubicScanECModel(const NEWMAT::ColumnVector& ep, bool field=false) EddyTry : PolynomialScanECModel(ep) {
    _nepd = (field) ? 20 : 19;
    if (_ep.Nrows() != int(_nepd)) throw EddyException("CubicScanECModel: Wrong number of elements for ep");
  } EddyCatch
  virtual ~CubicScanECModel() {}
  ECModel WhichModel() const EddyTry { return(Cubic); } EddyCatch
  bool HasFieldOffset() const EddyTry { return(_nepd==20); } EddyCatch
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new CubicScanECModel(*this))); } EddyCatch
  virtual unsigned int NCompoundDerivs() const { return(4); }
  virtual EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const;
  virtual double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const EddyTry {
    if (dindx>=NDerivs() && !allow_field_offset) throw EddyException("CubicScanECModel::GetDerivScale(allow_field_offset=false): dindx out of range");
    else if (dindx>=NParam()) throw EddyException("CubicScanECModel::GetDerivScale(allow_field_offset=true): dindx out of range");
    if (dindx < 3) return(1e-3);
    else if (dindx < 9) return(1e-5);
    else if (dindx < 19) return(1e-7);
    else return(1e-2);
    throw EddyException("CubicScanECModel::GetDerivScale: This should not be possible");
  } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const;
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class NoECScanECModel (Movement Scan Eddy Current Model)
//
// This class doesn't model the eddy curents at all, and simply
// uses the rigid body model. This is done to create a polymorphism
// so we can use the same basic code for the b0 scans as for the
// diffusion weighted ones.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class NoECScanECModel : public ScanECModel
{
public:
  NoECScanECModel() EddyTry : ScanECModel() {} EddyCatch
  NoECScanECModel(const NEWMAT::ColumnVector& ep) EddyTry {
    if (_ep.Nrows()) throw EddyException("NoECScanScanECModel: ep must have 0 elements");
  } EddyCatch
  virtual ~NoECScanECModel() {}
  ECModel WhichModel() const { return(NoEC); }
  bool HasFieldOffset() const { return(false); }
  double GetFieldOffset() const { return(0.0); }
  void SetFieldOffset(double ofst) { }
  unsigned int NDerivs() const { return(0); }
  unsigned int NCompoundDerivs() const { return(0); }
  EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, const std::vector<unsigned int>& pev) const { throw EddyException("NoECScanECModel::GetCompoundDerivInstructions: Model has no EC parameters"); };
  double GetDerivParam(unsigned int dindx, bool allow_field_offset=false) const {
    throw EddyException("NoECScanECModel::GetDerivParam: Model has no EC parameters");
  }
  void SetDerivParam(unsigned int dindx, double p, bool allow_field_offset=false) {
    throw EddyException("NoECScanECModel::SetDerivParam: Model has no EC parameters");
  }
  double GetDerivScale(unsigned int dindx, bool allow_field_offset=false) const {
    throw EddyException("NoECScanECModel::GetDerivScale: Model has no EC parameters");
  }
  std::shared_ptr<ScanECModel> Clone() const EddyTry { return(std::shared_ptr<ScanECModel>( new NoECScanECModel(*this))); } EddyCatch
  NEWIMAGE::volume<float> ECField(const NEWIMAGE::volume<float>& scan) const EddyTry { NEWIMAGE::volume<float> field=scan; field=0.0; return(field); } EddyCatch
};

} // End namespace EDDY

#endif // End #ifndef ECModels_h
