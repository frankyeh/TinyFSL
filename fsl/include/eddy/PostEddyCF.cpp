// Definitions of classes and functions that
// perform a post-hoc registration of the shells
// for the eddy project/.
//
// post_registration.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "miscmaths/nonlin.h"
#include "warpfns/warpfns.h"
#include "topup/topup_file_io.h"
#include "EddyUtils.h"
#include "PostEddyCF.h"

namespace EDDY {

class PostEddyCFImpl
{
public:
  PostEddyCFImpl(const NEWIMAGE::volume<float>&  ref,
		 const NEWIMAGE::volume<float>&  ima,
		 const NEWIMAGE::volume<float>&  mask) EddyTry : _ref(ref), _ima(ima), _mask(mask) {} EddyCatch
  ~PostEddyCFImpl() {}
  double cf(const NEWMAT::ColumnVector&    p,
	    const EDDY::MutualInfoHelper&  fwd_mih,
	    const EDDY::MutualInfoHelper&  bwd_mih,
	    int                            pe_dir) const;
  NEWIMAGE::volume<float> GetTransformedIma(const NEWMAT::ColumnVector& p,
                                            int                         pe_dir) const;
private:
  NEWIMAGE::volume<float>   _ref;
  NEWIMAGE::volume<float>   _ima;
  NEWIMAGE::volume<float>   _mask;
};

PostEddyCF::PostEddyCF(const NEWIMAGE::volume<float>&  ref,
		       const NEWIMAGE::volume<float>&  ima,
	               const NEWIMAGE::volume<float>&  mask,
		       unsigned int                    nbins) EddyTry
: _fwd_mih(nbins,ref.robustmin(),ref.robustmax(),ima.robustmin(),ima.robustmax()),
  _bwd_mih(nbins,ima.robustmin(),ima.robustmax(),ref.robustmin(),ref.robustmax())
{
  _pimpl = new PostEddyCFImpl(ref,ima,mask);
  _pe_dir = -1; // Disallowed value
} EddyCatch

PostEddyCF::PostEddyCF(const NEWIMAGE::volume<float>&  ref,
		       const NEWIMAGE::volume<float>&  ima,
	               const NEWIMAGE::volume<float>&  mask,
		       unsigned int                    nbins,
		       unsigned int                    pe_dir) EddyTry
: _fwd_mih(nbins,ref.robustmin(),ref.robustmax(),ima.robustmin(),ima.robustmax()),
  _bwd_mih(nbins,ima.robustmin(),ima.robustmax(),ref.robustmin(),ref.robustmax())
{
  _pimpl = new PostEddyCFImpl(ref,ima,mask);
  if (pe_dir > 1) throw EddyException("EDDY::PostEddyCF::PostEddyCF: pe_dir must be 0 or 1");
  else _pe_dir = static_cast<int>(pe_dir);
} EddyCatch

PostEddyCF::~PostEddyCF() { delete _pimpl; }

double PostEddyCF::cf(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->cf(p,_fwd_mih,_bwd_mih,_pe_dir)); } EddyCatch

NEWMAT::ReturnMatrix PostEddyCF::grad(const NEWMAT::ColumnVector& p) const EddyTry
{
  NEWMAT::ColumnVector tp = p;
  NEWMAT::ColumnVector gradv(p.Nrows());
  static const double dscale[] = {1e-2, 1e-2, 1e-2, 1e-5, 1e-5, 1e-5}; // Works both for Nrows = 1 and Nrows = 6
  double base = _pimpl->cf(tp,_fwd_mih,_bwd_mih,_pe_dir);
  for (int i=0; i<p.Nrows(); i++) {
    tp(i+1) += dscale[i];
    gradv(i+1) = (_pimpl->cf(tp,_fwd_mih,_bwd_mih,_pe_dir) - base) / dscale[i];
    tp(i+1) -= dscale[i];
  }
  gradv.Release();
  return(gradv);
} EddyCatch

NEWIMAGE::volume<float> PostEddyCF::GetTransformedIma(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->GetTransformedIma(p,_pe_dir)); } EddyCatch

double PostEddyCFImpl::cf(const NEWMAT::ColumnVector&    p,
			  const EDDY::MutualInfoHelper&  fwd_mih,
			  const EDDY::MutualInfoHelper&  bwd_mih,
			  int                            pe_dir) const EddyTry
{
  NEWIMAGE::volume<float> rima = _ima; rima = 0.0;
  NEWIMAGE::volume<char> mask1(_ima.xsize(),_ima.ysize(),_ima.zsize());
  NEWIMAGE::copybasicproperties(_ima,mask1); mask1 = 1;
  NEWIMAGE::volume<float> mask2 = _ima; mask2 = 0.0;
  NEWMAT::Matrix A;
  if (p.Nrows() == 1) {
    if(pe_dir<0 || pe_dir>1) EddyException("EDDY::PostEddyCFImpl::cf: invalid pe_dir value");
    NEWMAT::ColumnVector tmp(6); tmp = 0.0;
    tmp(pe_dir + 1) = p(1);
    A = TOPUP::MovePar2Matrix(tmp,_ima);
  }
  else if (p.Nrows() == 6) A = TOPUP::MovePar2Matrix(p,_ima);
  else throw EddyException("EDDY::PostEddyCFImpl::cf: size of p must be 1 or 6");
  // Transform image
  NEWIMAGE::affine_transform(_ima,A,rima,mask1);
  // Transform mask in same way
  NEWIMAGE::affine_transform(_mask,A,mask2);
  // mask2.binarise(0.99); // Value above (arbitrary) 0.99 implies valid voxels
  // Combine masks
  mask2 *= EddyUtils::ConvertMaskToFloat(mask1);
  // double rval = - mih.MI(_ref,rima,mask2);
  double rval = - fwd_mih.SoftMI(_ref,rima,mask2);

  // Backward transform image
  rima = 0.0; mask1 = 1.0; mask2 = 0.0;
  NEWIMAGE::affine_transform(_ref,A.i(),rima,mask1);
  // Backward transform mask
  NEWIMAGE::affine_transform(_mask,A.i(),mask2);
  // mask2.binarise(0.99);
  mask2 *= EddyUtils::ConvertMaskToFloat(mask1);
  // rval += - mih.MI(_ima,rima,mask2); rval /= 2.0;
  rval += - bwd_mih.SoftMI(_ima,rima,mask2); rval /= 2.0;

  //cout << "rval = " << rval << endl;
  //cout << "p = " << p << endl;

  return(rval);
} EddyCatch

NEWIMAGE::volume<float> PostEddyCFImpl::GetTransformedIma(const NEWMAT::ColumnVector& p,
							  int                         pe_dir) const EddyTry
{
  NEWIMAGE::volume<float> rima = _ima; rima = 0.0;
  NEWMAT::Matrix A;
  if (p.Nrows() == 1) {
    if(pe_dir<0 || pe_dir>1) EddyException("EDDY::PostEddyCFImpl::GetTransformedIma: invalid pe_dir value");
    NEWMAT::ColumnVector tmp(6); tmp = 0.0;
    tmp(pe_dir + 1) = p(1);
    A = TOPUP::MovePar2Matrix(tmp,_ima);
  }
  else if (p.Nrows() == 6) A = TOPUP::MovePar2Matrix(p,_ima);
  else throw EddyException("EDDY::PostEddyCFImpl::GetTransformedIma: size of p must be 1 or 6");
  NEWIMAGE::affine_transform(_ima,A,rima);
  return(rima);
} EddyCatch

} // End namespace EDDY
