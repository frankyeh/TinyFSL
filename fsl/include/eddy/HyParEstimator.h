/*! \file HyParEstimator.h
    \brief Contains declaration of class for estimating hyper parameters for a given GP model and data.

    \author Jesper Andersson
    \version 1.0b, Nov., 2013.
*/
// Contains declaration of class for estimating
// hyper parameters for a given GP model and data.
//
// HyParEstimator.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2013 University of Oxford
//

#ifndef HyParEstimator_h
#define HyParEstimator_h

#include <cstdlib>
#include <string>
#include <exception>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "EddyHelperClasses.h"
#include "KMatrix.h"

namespace EDDY {

class DataSelector
{
public:
  DataSelector(const std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >& idata,
	       const NEWIMAGE::volume<float>&                                 mask,
	       unsigned int                                                   nvox,
	       int                                                            rndinit=0) EddyTry { common_constructor(idata,mask,nvox,0.0,rndinit); } EddyCatch
  DataSelector(const std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >& idata,
	       const NEWIMAGE::volume<float>&                                 mask,
	       unsigned int                                                   nvox,
	       float                                                          fwhm,
	       int                                                            rndinit=0) EddyTry { common_constructor(idata,mask,nvox,fwhm,rndinit); } EddyCatch
  ~DataSelector() {}
  const std::vector<NEWMAT::ColumnVector>& GetData() const EddyTry { return(_data); } EddyCatch
  NEWMAT::Matrix AsNEWMAT() const;
  NEWMAT::Matrix CoordsAsNEWMAT() const;
  void Write(const std::string& fname) const EddyTry { MISCMATHS::write_ascii_matrix(AsNEWMAT(),fname); } EddyCatch
  void WriteCoords(const std::string& fname) const EddyTry { MISCMATHS::write_ascii_matrix(CoordsAsNEWMAT(),fname); } EddyCatch
private:
  void common_constructor(const std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >& idata,
			  const NEWIMAGE::volume<float>&                                 mask,
			  unsigned int                                                   nvox,
			  float                                                          fwhm,
			  int                                                            rndinit);
  float get_smooth(const NEWIMAGE::volume<float>&           ima,
		   const NEWIMAGE::volume<float>&           mask,
		   const std::vector<unsigned int>&         coords,
		   const std::vector<NEWMAT::ColumnVector>& kernels);
  std::vector<NEWMAT::ColumnVector>        _data;   //
  std::vector<std::vector<unsigned int> >  _coords; //
};

class HyParCF : public MISCMATHS::NonlinCF
{
public:
  HyParCF() {}
  virtual ~HyParCF() {}
  virtual std::shared_ptr<HyParCF> Clone() const = 0;
  virtual void SetData(const std::vector<NEWMAT::ColumnVector>& data) EddyTry { _data = data; } EddyCatch
  virtual void SetKMatrix(std::shared_ptr<const KMatrix> K) EddyTry { _K = K->Clone(); } EddyCatch
  virtual double cf(const NEWMAT::ColumnVector& p) const = 0;
protected:
  std::vector<NEWMAT::ColumnVector>          _data; // Data
  std::shared_ptr<KMatrix>                   _K;    //
  double sqr(double a) const { return(a*a); }
  float sqr(float a) const { return(a*a); }
};

class MMLHyParCF : public HyParCF
{
public:
  MMLHyParCF() EddyTry : HyParCF() {} EddyCatch
  virtual ~MMLHyParCF()  {}
  std::shared_ptr<HyParCF> Clone() const EddyTry { return(std::shared_ptr<MMLHyParCF>(new MMLHyParCF(*this))); } EddyCatch
  double cf(const NEWMAT::ColumnVector& p) const;
};

class CVHyParCF : public HyParCF
{
public:
  CVHyParCF() EddyTry : HyParCF() {} EddyCatch
  virtual ~CVHyParCF()  {}
  std::shared_ptr<HyParCF> Clone() const EddyTry { return(std::shared_ptr<CVHyParCF>(new CVHyParCF(*this))); } EddyCatch
  double cf(const NEWMAT::ColumnVector& p) const;
};

class GPPHyParCF : public HyParCF
{
public:
  GPPHyParCF() EddyTry : HyParCF() {} EddyCatch
  virtual ~GPPHyParCF()  {}
  std::shared_ptr<HyParCF> Clone() const EddyTry { return(std::shared_ptr<GPPHyParCF>(new GPPHyParCF(*this))); } EddyCatch
  double cf(const NEWMAT::ColumnVector& p) const;
};

class HyParEstimator
{
public:
  HyParEstimator() {}
  HyParEstimator(const NEWMAT::ColumnVector& hpar) EddyTry : _hpar(hpar) {} EddyCatch
  virtual ~HyParEstimator() {}
  virtual std::shared_ptr<HyParEstimator> Clone() const = 0;
  std::vector<double> GetHyperParameters() const EddyTry { return(newmat_2_stl(_hpar)); } EddyCatch
  virtual unsigned int GetNVox() const { return(0); }
  virtual int RndInit() const { return(0); }
  virtual void SetData(const std::vector<NEWMAT::ColumnVector>&    data) {}
  virtual void Estimate(std::shared_ptr<const KMatrix>  K, bool verbose) = 0;
protected:
  const NEWMAT::ColumnVector& get_hpar() const EddyTry { return(_hpar); } EddyCatch
  void set_hpar(const NEWMAT::ColumnVector& hpar) EddyTry { _hpar = hpar; } EddyCatch
  std::vector<double> newmat_2_stl(const NEWMAT::ColumnVector& nm) const EddyTry {
    std::vector<double> stl(nm.Nrows());
    for (unsigned int i=0; i<stl.size(); i++) stl[i] = nm(i+1);
    return(stl);
  } EddyCatch
  NEWMAT::ColumnVector stl_2_newmat(const std::vector<double>& stl) const EddyTry {
    NEWMAT::ColumnVector nm(stl.size());
    for (unsigned int i=0; i<stl.size(); i++) nm(i+1) = stl[i];
    return(nm);
  } EddyCatch
private:
  NEWMAT::ColumnVector         _hpar;  // Hyper parameters
};


class FixedValueHyParEstimator : public HyParEstimator
{
public:
  FixedValueHyParEstimator(const NEWMAT::ColumnVector& hpar) EddyTry : HyParEstimator(hpar) {} EddyCatch
  virtual ~FixedValueHyParEstimator() {}
  std::shared_ptr<HyParEstimator> Clone() const EddyTry { return(std::shared_ptr<FixedValueHyParEstimator>(new FixedValueHyParEstimator(*this))); } EddyCatch
  virtual void Estimate(std::shared_ptr<const KMatrix>  K, bool verbose) EddyTry {
    if (verbose) std::cout << "Hyperparameters set to user specified values: " << get_hpar() << std::endl;
  } EddyCatch
};

class CheapAndCheerfulHyParEstimator : public HyParEstimator
{
public:
  CheapAndCheerfulHyParEstimator(unsigned int                      nvox=1000,
				 int                               ir=0) EddyTry : _nvox(nvox), _ir(ir) {} EddyCatch
  virtual ~CheapAndCheerfulHyParEstimator() {}
  std::shared_ptr<HyParEstimator> Clone() const EddyTry { return(std::shared_ptr<CheapAndCheerfulHyParEstimator>(new CheapAndCheerfulHyParEstimator(*this))); } EddyCatch
  virtual unsigned int GetNVox() const { return(_nvox); }
  virtual int RndInit() const { return(_ir); }
  virtual void SetData(const std::vector<NEWMAT::ColumnVector>&    data) EddyTry { _data = data; } EddyCatch
  virtual void Estimate(std::shared_ptr<const KMatrix>  K, bool verbose);
private:
  unsigned int                       _nvox;
  int                                _ir;    // Initialise rand?
  std::vector<NEWMAT::ColumnVector>  _data;
};

class FullMontyHyParEstimator : public HyParEstimator
{
public:
  FullMontyHyParEstimator(std::shared_ptr<const HyParCF>   hpcf,
			  double                           evff=1.0,
			  unsigned int                     nvox=1000,
			  int                              ir=0,
			  bool                             verbose=false) EddyTry : _cf(hpcf->Clone()), _evff(evff), _nvox(nvox), _ir(ir), _v(verbose) {} EddyCatch
  virtual ~FullMontyHyParEstimator() {}
  std::shared_ptr<HyParEstimator> Clone() const EddyTry { return(std::shared_ptr<FullMontyHyParEstimator>(new FullMontyHyParEstimator(*this))); } EddyCatch
  virtual unsigned int GetNVox() const { return(_nvox); }
  virtual int RndInit() const { return(_ir); }
  virtual void SetData(const std::vector<NEWMAT::ColumnVector>&    data) EddyTry { _data = data; } EddyCatch
  virtual void Estimate(std::shared_ptr<const KMatrix>  K, bool verbose);
private:
  static const unsigned int          MITER=500;

  std::shared_ptr<HyParCF>           _cf;    // Cost-function object for use with nonlin
  double                             _evff;  // Error variance fudge factor
  unsigned int                       _nvox;
  int                                _ir;
  bool                               _v;     // Verbose flag
  std::vector<NEWMAT::ColumnVector>  _data;
};

} // End namespace EDDY

#endif // end #ifndef HyParEstimator_h
