// Definitions of classes and functions that
// estimates the derivative fields for a
// movement-by-susceptibility model for the
// eddy project.
//
// This file contins all the code for both the
// CPU and the GPU implementations. The code
// generation is goverened #include:s and
// the COMPILE_GPU macro.
//
// MoveBySuscCF.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford
//
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "utils/FSLProfiler.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "EddyHelperClasses.h"
#include "DiffusionGP.h"
#include "b0Predictor.h"
#include "EddyUtils.h"
#include "ECScanClasses.h"
#include "MoveBySuscCF.h"
#include "eddy.h"
#ifdef COMPILE_GPU
#include "cuda/EddyGpuUtils.h"
#include "cudabasisfield/CBFSplineField.cuh"
#endif

using namespace std;
using namespace MISCMATHS;

namespace EDDY {

class MoveBySuscCFImpl
{
public:
  MoveBySuscCFImpl(EDDY::ECScanManager&                 sm,
		   const EDDY::EddyCommandLineOptions&  clo,
		   const std::vector<unsigned int>&     b0s,
		   const std::vector<unsigned int>&     dwis,
		   const std::vector<unsigned int>&     mps,
		   unsigned int                         order,
		   bool                                 ujm,
		   double                               ksp);
  ~MoveBySuscCFImpl() {}
  double cf(const NEWMAT::ColumnVector&    p);
  NEWMAT::ReturnMatrix grad(const NEWMAT::ColumnVector&    p);
  std::shared_ptr<BFMatrix> hess(const NEWMAT::ColumnVector& p,
				   std::shared_ptr<BFMatrix> iptr=std::shared_ptr<BFMatrix>());
  void SetLambda(double lambda) { _lmbd = lambda; }
  unsigned int NPar() const EddyTry { return(this->total_no_of_parameters()); } EddyCatch
  NEWMAT::ReturnMatrix Par() const EddyTry { return(this->get_field_parameters()); } EddyCatch
  void WriteFirstOrderFields(const std::string& fname) const;
  void WriteSecondOrderFields(const std::string& fname) const;
  /// Used to force recalculation of sum images the next time cf, grad or hess is called
  void ResetCache() { _utd = false; _m_utd = false; }
private:
  EDDY::ECScanManager&                                                   _sm;    ///< Ref to ScanManger. Means that external object will change.
  EDDY::EddyCommandLineOptions                                           _clo;   ///< Internal copy of Command Line Options.
  std::vector<unsigned int>                                              _b0s;   ///< Indicies for st=B0 of b0 volumes to use
  std::vector<unsigned int>                                              _dwis;  ///< Indicies for st=DWI of dwi volumes to use
  std::vector<unsigned int>                                              _mps;   ///< Zero-offset indicies to the movement parameters to model
  unsigned int                                                           _order; ///< Order, should be 1 or 2
  bool                                                                   _ujm;   ///< If true, Jacobian modulation is used for derivatives and Hessian
  double                                                                 _lmbd;  ///< Lambda (weight of regularisation).
  std::vector<unsigned int>                                              _ksp;   ///< Knot-spacing of spline-fields
#ifdef COMPILE_GPU
  std::vector<std::shared_ptr<CBF::CBFSplineField> >                     _sp1;   ///< Modelled first order fields
  std::vector<std::vector<std::shared_ptr<CBF::CBFSplineField> > >       _sp2;   ///< Modelled 2nd order fields, organised as sub-diagonal matrix
#else
  std::vector<std::shared_ptr<BASISFIELD::splinefield> >                 _sp1;   ///< Modelled first order fields
  std::vector<std::vector<std::shared_ptr<BASISFIELD::splinefield> > >   _sp2;   ///< Modelled 2nd order fields, organised as sub-diagonal matrix
#endif
  NEWMAT::ColumnVector                                                   _cp;    ///< Current p (parameters)
  NEWMAT::ColumnVector                                                   _hypar; ///< Hyper-parameters for GP
  NEWIMAGE::volume<float>                                                _mask;  ///< Latest calculated mask with data for all volumes
  NEWIMAGE::volume<char>                                                 _cmask; ///< char representation of mask
  unsigned int                                                           _nvox;  ///< Number of non-zero voxels in mask
  unsigned int                                                           _omp_num_threads; ///< Value of OMP_NUM_THREADS when object is created
  // Here I declare variables that are initialised
  MISCMATHS::BFMatrixPrecisionType  _hp = MISCMATHS::BFMatrixDoublePrecision;     ///< Precision (double or float) of Hessian
  bool                                                            _bno = true;    ///< Brand New Object?
  bool                                                            _utd = false;   ///< Are all the sum images up-to-date?
  bool                                                            _m_utd = false; ///< Is mask up to date or not?

  bool                                                            _chgr = false;  ///< Check Gradient, write gradient checking info
  bool                                                            _chH = false;   ///< Check Hessian, write Hessian checking info
  unsigned int                                                    _grad_cnt = 0;  ///< Counter to see how many times .grad() has been called
  unsigned int                                                    _hess_cnt = 0;  ///< Counter to see how many times .hess() has been called

  // Here I declare sums-of-products of image volumes. It "looks like there would
  // be a lot of these sum images, but for the "normal" case of a first-order model
  // for pitch and roll with a single PE-orientation there is only one for .cf(),
  // four for .grad() and six for .hess(). So the memory footprint is small.

  // These two typedefs are used to make subsequent declarations a little saner
  typedef std::vector<NEWIMAGE::volume<float> > ImageVec;    ///< Vector of images
  typedef std::vector<unsigned int>             BpeVec;      ///< Binarised PE-vector
  // Used by .cf()
  NEWIMAGE::volume<float>                                    _sum_sqr_diff;      ///< Sum of squared differences between prediction and observation
  // Used by .grad()
  ImageVec                                                   _sum_fo_deriv_diff; ///< First order Sum(deriv*diff)
  std::map<BpeVec,ImageVec>                                  _sum_fo_ima_diff;   ///< First order Sum(ima*diff)
  std::vector<ImageVec>                                      _sum_so_deriv_diff; ///< Second order Sum(deriv*diff)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_ima_diff;   ///< Second order Sum(ima*diff)
  // Used by .hess()
  std::vector<ImageVec>                                      _sum_fo_deriv_deriv;///< First order Sum(deriv*deriv)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_fo_ima_ima;    ///< First order Sum(ima*ima)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_fo_deriv_ima;  ///< First order Sum(deriv*ima)
  std::vector<ImageVec>                                      _sum_so_deriv_deriv;///< Second order Sum(deriv*deriv)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_ima_ima;    ///< Second order Sum(ima*ima)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_so_deriv_ima;  ///< Second order Sum(deriv*ima)
  std::vector<ImageVec>                                      _sum_cross_deriv_deriv; ///< Cross-term between 1st and 2nd order Sum(deriv*deriv)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_cross_ima_ima;     ///< Cross-term between 1st and 2nd order Sum(ima*ima)
  std::map<BpeVec,std::vector<ImageVec> >                    _sum_cross_deriv_ima;   ///< Cross-term between 1st and 2nd order Sum(deriv*ima)


  // These private functions are all candidates for removal

  unsigned int no_of_parameters_per_field() const EddyTry { return(_sp1[0]->CoefSz_x() * _sp1[0]->CoefSz_y() * _sp1[0]->CoefSz_z()); } EddyCatch
  unsigned int no_of_first_order_fields() const EddyTry { return(_mps.size()); } EddyCatch
  unsigned int no_of_second_order_fields() const EddyTry { return((_order == 1) ? 0 : _mps.size()*(_mps.size()+1)/2); } EddyCatch
  unsigned int no_of_fields() const EddyTry { return(no_of_first_order_fields() + no_of_second_order_fields()); } EddyCatch
  unsigned int total_no_of_parameters() const EddyTry { return(no_of_parameters_per_field()*no_of_fields()); } EddyCatch
  unsigned int no_of_b0s() const EddyTry { return(_b0s.size()); } EddyCatch
  unsigned int no_of_dwis() const EddyTry { return(_dwis.size()); } EddyCatch
  unsigned int nvox() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_nvox); } EddyCatch
  double lambda() const { return(_lmbd); }
  double pi() const { return(3.141592653589793); }
  MISCMATHS::BFMatrixPrecisionType hessian_precision() const { return(_hp); }
  NEWMAT::ColumnVector get_field_parameters() const EddyTry { return(_cp); } EddyCatch
  void set_field_parameters(EDDY::ECScanManager&        sm,
			    const NEWMAT::ColumnVector& p);
  const NEWIMAGE::volume<float>& mask() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_mask); } EddyCatch
  const NEWIMAGE::volume<char>& cmask() EddyTry { if (!_m_utd) recalculate_sum_images(); return(_cmask); } EddyCatch
  void resize_containers_for_sum_images_for_grad(const std::vector<std::vector<unsigned int> >& indicies,
						 const std::vector<EDDY::ScanType>&             imtypes);
  void resize_containers_for_sum_images_for_hess(const std::vector<std::vector<unsigned int> >& indicies,
						 const std::vector<EDDY::ScanType>&             imtypes);
  void set_sum_images_to_zero(const NEWIMAGE::volume<float>& ima);
  void recalculate_sum_images();
  void recalculate_sum_so_imas_for_hess(const NEWMAT::ColumnVector&           p,
					const BpeVec&                         bpe,
					const NEWIMAGE::volume<float>&        ima,
					const NEWIMAGE::volume<float>&        deriv);
  void recalculate_sum_cross_imas_for_hess(const NEWMAT::ColumnVector&           p,
					   const BpeVec&                         bpe,
					   const NEWIMAGE::volume<float>&        ima,
					   const NEWIMAGE::volume<float>&        deriv);
  void calculate_first_order_subm(// Input
				  const std::vector<ImageVec>&                                        sop_dfdf,
				  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
				  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
				  const NEWIMAGE::volume<char>&                                       lmask,
				  unsigned int                                                        nvox,
				  // Output
				  std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;
  void calculate_second_order_subm(// Input
				   const std::vector<ImageVec>&                                        sop_dfdf,
				   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
				   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
				   const NEWIMAGE::volume<char>&                                       lmask,
				   unsigned int                                                        nvox,
				   // Output
				   std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;
  void calculate_cross_subm(// Input
			    const std::vector<ImageVec>&                                        sop_dfdf,
			    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
			    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
			    const NEWIMAGE::volume<char>&                                       lmask,
			    unsigned int                                                        nvox,
			    // Output
			    std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const;

  NEWMAT::ColumnVector stl2newmat(const std::vector<double>& stl) const;
  NEWMAT::Matrix linspace(const NEWMAT::Matrix& inmat) const;
  std::shared_ptr<MISCMATHS::BFMatrix> concatenate_subdiag_subm(const std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
								  unsigned int                                                              n) const;
  std::shared_ptr<MISCMATHS::BFMatrix> concatenate_rect_subm(const std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
							       unsigned int                                                              m,
							       unsigned int                                                              n) const;
  bool is_first_index_pair(unsigned int                                                 i,
			   unsigned int                                                 j,
			   const std::vector<unsigned int>&                             ivec,
			   const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const;
  std::pair<unsigned int,unsigned int> find_first_index_pair(const std::vector<unsigned int>&                             ivec,
							     const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const;
  std::vector<std::vector<std::vector<unsigned int> > > build_second_order_pmap() const;
  std::vector<std::vector<std::vector<unsigned int> > > build_cross_pmap() const;
  std::vector<unsigned int> make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
					      const std::pair<unsigned int,unsigned int>& p2,
					      unsigned int                                np) const;
  std::vector<unsigned int> make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
					      unsigned int                                i,
					      unsigned int                                np) const;
  std::pair<unsigned int,unsigned int> get_second_order_index_pair(unsigned int i,
								   unsigned int np) const;
};

MoveBySuscCF::MoveBySuscCF(EDDY::ECScanManager&                 sm,
			   const EDDY::EddyCommandLineOptions&  clo,
			   const std::vector<unsigned int>&     b0s,
			   const std::vector<unsigned int>&     dwis,
			   const std::vector<unsigned int>&     mps,
			   unsigned int                         order,
			   double                               ksp) EddyTry { _pimpl = new MoveBySuscCFImpl(sm,clo,b0s,dwis,mps,order,true,ksp); } EddyCatch

MoveBySuscCF::~MoveBySuscCF() { delete _pimpl; }

double MoveBySuscCF::cf(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->cf(p)); } EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCF::grad(const NEWMAT::ColumnVector& p) const EddyTry { return(_pimpl->grad(p)); } EddyCatch

std::shared_ptr<BFMatrix> MoveBySuscCF::hess(const NEWMAT::ColumnVector& p,
					       std::shared_ptr<BFMatrix> iptr) const EddyTry { return(_pimpl->hess(p,iptr)); } EddyCatch

void MoveBySuscCF::SetLambda(double lambda) EddyTry { _pimpl->SetLambda(lambda); } EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCF::Par() const EddyTry { return(_pimpl->Par()); } EddyCatch

unsigned int MoveBySuscCF::NPar() const EddyTry { return(_pimpl->NPar()); } EddyCatch

void MoveBySuscCF::WriteFirstOrderFields(const std::string& fname) const EddyTry { _pimpl->WriteFirstOrderFields(fname); } EddyCatch

void MoveBySuscCF::WriteSecondOrderFields(const std::string& fname) const EddyTry { _pimpl->WriteSecondOrderFields(fname); } EddyCatch

void MoveBySuscCF::ResetCache() EddyTry { _pimpl->ResetCache(); } EddyCatch

MoveBySuscCFImpl::MoveBySuscCFImpl(EDDY::ECScanManager&                 sm,
				   const EDDY::EddyCommandLineOptions&  clo,
				   const std::vector<unsigned int>&     b0s,
				   const std::vector<unsigned int>&     dwis,
				   const std::vector<unsigned int>&     mps,
				   unsigned int                         order,
				   bool                                 ujm,
				   double                               ksp) EddyTry : _sm(sm), _clo(clo), _b0s(b0s), _dwis(dwis), _mps(mps), _order(order), _ujm(ujm), _lmbd(50)
{
  static Utilities::FSLProfiler prof("_"+string(__FILE__)+"_"+string(__func__));
  double total_key = prof.StartEntry("Total");

  if (order != 1 && order != 2) throw EddyException("MoveBySuscCFImpl::MoveBySuscCFImpl: order must be 1 or 2");
  std::vector<unsigned int> isz(3,0);
  isz[0] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().xsize());
  isz[1] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().ysize());
  isz[2] = static_cast<unsigned int>(_sm.Scan(0,ANY).GetIma().zsize());
  std::vector<double> vxs(3,0);
  vxs[0] = static_cast<double>(_sm.Scan(0,ANY).GetIma().xdim());
  vxs[1] = static_cast<double>(_sm.Scan(0,ANY).GetIma().ydim());
  vxs[2] = static_cast<double>(_sm.Scan(0,ANY).GetIma().zdim());
  _ksp.resize(3);
  _ksp[0] = static_cast<unsigned int>((ksp / vxs[0]) + 0.5);
  _ksp[1] = static_cast<unsigned int>((ksp / vxs[1]) + 0.5);
  _ksp[2] = static_cast<unsigned int>((ksp / vxs[2]) + 0.5);

  _sp1.resize(_mps.size(),nullptr);
  for (unsigned int i=0; i<_mps.size(); i++) {
#ifdef COMPILE_GPU
    _sp1[i] = std::shared_ptr<CBF::CBFSplineField>(new CBF::CBFSplineField(isz,vxs,_ksp));
#else
    _sp1[i] = std::shared_ptr<BASISFIELD::splinefield>(new BASISFIELD::splinefield(isz,vxs,_ksp));
#endif
  }
  if (order == 2) {
    _sp2.resize(_mps.size());
    for (unsigned int i=0; i<_mps.size(); i++) {
      _sp2[i].resize(i+1,nullptr);
      for (unsigned int j=0; j<(i+1); j++) {
#ifdef COMPILE_GPU
        _sp2[i][j] = std::shared_ptr<CBF::CBFSplineField>(new CBF::CBFSplineField(isz,vxs,_ksp));
#else
        _sp2[i][j] = std::shared_ptr<BASISFIELD::splinefield>(new BASISFIELD::splinefield(isz,vxs,_ksp));
#endif
      }
    }
  }
  _cp.ReSize(this->total_no_of_parameters()); _cp = 0.0;
  char *ont = getenv("OMP_NUM_THREADS");
  if (ont == nullptr) _omp_num_threads = 1;
  else if (sscanf(ont,"%u",&_omp_num_threads) != 1) throw EddyException("MoveBySuscCFImpl::MoveBySuscCFImpl: problem reading environment variable OMP_NUM_THREADS");

  prof.EndEntry(total_key);

} EddyCatch

void MoveBySuscCFImpl::WriteFirstOrderFields(const std::string& fname) const EddyTry
{
  const NEWIMAGE::volume<float>& tmp=_sm.Scan(0,ANY).GetIma();
  NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),no_of_first_order_fields());
  NEWIMAGE::copybasicproperties(tmp,ovol);
  NEWIMAGE::volume<float> dfield = tmp;
  for (unsigned int i=0; i<_mps.size(); i++) {
    _sp1[i]->AsVolume(dfield);
    if (_mps[i] > 2 && _mps[i] < 6) { // If it is a rotation
      dfield *= static_cast<float>(this->pi() / 180.0);
    }
    ovol[i] = dfield;
  }
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch

void MoveBySuscCFImpl::WriteSecondOrderFields(const std::string& fname) const EddyTry
{
  const NEWIMAGE::volume<float>& tmp=_sm.Scan(0,ANY).GetIma();
  NEWIMAGE::volume4D<float> ovol(tmp.xsize(),tmp.ysize(),tmp.zsize(),no_of_second_order_fields());
  NEWIMAGE::copybasicproperties(tmp,ovol);
  NEWIMAGE::volume<float> dfield = tmp;
  unsigned int cnt=0;
  for (unsigned int i=0; i<_mps.size(); i++) {
    for (unsigned int j=0; j<(i+1); j++) {
      _sp2[i][j]->AsVolume(dfield);
      ovol[cnt++] = dfield;
    }
  }
  NEWIMAGE::write_volume(ovol,fname);
} EddyCatch

void MoveBySuscCFImpl::set_field_parameters(EDDY::ECScanManager&        sm,
					    const NEWMAT::ColumnVector& p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::set_field_parameters: mismatch between p and total no of parameters");
  // Check if these parameters have been used before
  if (static_cast<unsigned int>(_cp.Nrows()) == total_no_of_parameters() && _cp == p) return;
  else {
    _m_utd = false;
    _utd = false;
    _cp = p;
    // Set first order fields
    unsigned int pindx=1;
    for (unsigned int i=0; i<_mps.size(); i++) {
      _sp1[i]->SetCoef(p.Rows(pindx,pindx+no_of_parameters_per_field()-1));
      NEWIMAGE::volume<float> dfield = sm.Scan(0,ANY).GetIma(); dfield=0.0;
      _sp1[i]->AsVolume(dfield);
      sm.SetDerivSuscField(_mps[i],dfield);
      pindx += no_of_parameters_per_field();
    }
    // And second order fields if requested
    if (_order == 2) {
      for (unsigned int i=0; i<_mps.size(); i++) {
	for (unsigned int j=0; j<(i+1); j++) {
	  _sp2[i][j]->SetCoef(p.Rows(pindx,pindx+no_of_parameters_per_field()-1));
	  NEWIMAGE::volume<float> dfield = sm.Scan(0,ANY).GetIma(); dfield=0.0;
	  _sp2[i][j]->AsVolume(dfield);
	  sm.Set2ndDerivSuscField(_mps[i],_mps[j],dfield);
	  pindx += no_of_parameters_per_field();
	}
      }
    }
  }
} EddyCatch

double MoveBySuscCFImpl::cf(const NEWMAT::ColumnVector&  p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::cf: mismatch between p and total no of parameters");
  // Set derivative fields according to p
  // cout << "Setting field parameters" << endl;
  set_field_parameters(_sm,p);
  recalculate_sum_images();
  // Sum of squared differences
  double ssd = _sum_sqr_diff.sum(mask()) / static_cast<double>(nvox());
  // Add contribution from regularisation
  double reg = 0;
  for (unsigned int i=0; i<_mps.size(); i++) {
    reg += lambda() * _sp1[i]->BendEnergy() / static_cast<double>(nvox());
  }
  if (_order == 2) {
    for (unsigned int i=0; i<_mps.size(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	reg += lambda() * _sp2[i][j]->BendEnergy() / static_cast<double>(nvox());
      }
    }
  }
  reg /= no_of_fields(); // Average over all fields.
  ssd += reg;

  return(ssd);
} EddyCatch

NEWMAT::ReturnMatrix MoveBySuscCFImpl::grad(const NEWMAT::ColumnVector& p) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::grad: mismatch between p and total no of parameters");
  // Set derivative fields according to p
  set_field_parameters(_sm,p);
  recalculate_sum_images();

  // Use the pre-calculated sum images to calculate the gradient
  // cout << "Starting the actual calculation of the gradient" << endl;
  NEWMAT::ColumnVector gradient(total_no_of_parameters()); gradient=0.0;
  unsigned int fr = 1;
  unsigned int lr = no_of_parameters_per_field();
  NEWIMAGE::volume<float> ones = _sm.Scan(0,ANY).GetIma(); ones = 1.0;
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    // Contribution from voxel-translations
    gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp1[pi]->Jte(_sum_fo_deriv_diff[pi],ones,&cmask());
    // Contribution from Jacobian modulation
    if (_ujm) {
      for (const auto& elem : _sum_fo_ima_diff) { // Loop over all binarised PE-vectors in sum_fo_ima_diff
	std::vector<unsigned int> dvec = elem.first;
	gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp1[pi]->Jte(dvec,elem.second[pi],ones,&cmask());
      }
    }
    fr += no_of_parameters_per_field();
    lr += no_of_parameters_per_field();
  }
  if (_order == 2) {
    for (unsigned int pi=0; pi<_mps.size(); pi++) {
      for (unsigned int pj=0; pj<(pi+1); pj++) {
	// Contribution from voxel-translations
	gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp2[pi][pj]->Jte(_sum_so_deriv_diff[pi][pj],ones,&cmask());
	// Contribution from Jacobian Modulation
	if (_ujm) {
	  for (const auto& elem: _sum_so_ima_diff) {
	    std::vector<unsigned int> dvec = elem.first;
	    gradient.Rows(fr,lr) += (2.0/static_cast<double>(nvox())) * _sp2[pi][pj]->Jte(dvec,elem.second[pi][pj],ones,&cmask());
	  }
	}
      }
    }
  }

  // Finally the contributions from regularisation
  fr = 1;
  lr = no_of_parameters_per_field();
  for (unsigned int i=0; i<_mps.size(); i++) {
    gradient.Rows(fr,lr) += (lambda()/static_cast<double>(nvox()*no_of_fields())) * _sp1[i]->BendEnergyGrad();
    fr += no_of_parameters_per_field();
    lr += no_of_parameters_per_field();
  }
  if (_order == 2) {
    for (unsigned int i=0; i<_mps.size(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	gradient.Rows(fr,lr) += (lambda()/static_cast<double>(nvox()*no_of_fields())) * _sp2[i][j]->BendEnergyGrad();
	fr += no_of_parameters_per_field();
	lr += no_of_parameters_per_field();
      }
    }
  }

  if (_chgr) { // If we are to check gradient
    char fname[256]; sprintf(fname,"gradient_%02d.txt",_grad_cnt);
    MISCMATHS::write_ascii_matrix(gradient,fname);

    NEWMAT::ColumnVector tmp_p = p;
    tmp_p(10000) += 0.0001;
    double cf0 = this->cf(tmp_p);
    tmp_p(10000) -= 0.0001;
    cf0 = this->cf(tmp_p);
    NEWMAT::ColumnVector numgrad(total_no_of_parameters()); numgrad=0.0;
    unsigned int no_of_values = 20;
    unsigned int step = total_no_of_parameters() / (no_of_values + 1);
    double delta = 0.001;
    for (unsigned int i=step/2; i<total_no_of_parameters(); i+=step) {
      if (i<total_no_of_parameters()) {
	tmp_p(i) += delta;
	double cf1 = this->cf(tmp_p);
	numgrad(i) = (cf1-cf0) / delta;
	tmp_p(i) -= delta;
      }
    }
    sprintf(fname,"numgrad_delta_001_%02d.txt",_grad_cnt);
    MISCMATHS::write_ascii_matrix(numgrad,fname);
  }
  _grad_cnt++;

  gradient.Release(); return(gradient);
} EddyCatch

/*!
 * Calculates and returns the Hessian. It is rather complicated for the following reasons
 * 1. Each sub-Hessian is a sum of four components. The direct translation and Jacobian components and
 *    the cross components (one is the transpose of the other).
 * 2. The Hessian is a tiling of NxN sub-Hessians, where N is the total number of fields that are
 *    modelled. The fields divide into first and second order fields, and the Hessian into three
 *    blocks of sub-Hessian. The first is the first order fields and their interactions. The second
 *    is the second order field and their interactions. The third is the interactions between the
 *    first and second order fields. The two latter contain repetitions of the same sub-matrix, and
 *    this should be taken into account for efficiency. The first two are represented as sub-diagonal
 *    matrices of sub-matrices. The last is represented as a rectangular matrix of sub-matrices.
 * \param[in] p The parameters that, when convolved with B-splines, define all modelled fields
 * \param[in] iptr The previous Hessian. Allows us to explicitly free up the memory before allocating new.
 * \return A boost shared ptr to a sparse matrix with the Gauss-Newton approximation to the Hessian.
 */
std::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::hess(const NEWMAT::ColumnVector& p,
							      std::shared_ptr<BFMatrix> iptr) EddyTry
{
  if (static_cast<unsigned int>(p.Nrows()) != total_no_of_parameters()) throw EddyException("MoveBySuscCFImpl::hess: mismatch between p and total no of parameters");
  // Set derivative fields according to p
  set_field_parameters(_sm,p);
  recalculate_sum_images();
  // Reclaim memory from old Hessian if there is one
  if (iptr) iptr->Clear();
  // Allocate matrices of pointers to sub-matrices
  std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > > first_order_subm(this->no_of_first_order_fields());
  std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > > second_order_subm(this->no_of_second_order_fields());
  std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > > cross_subm(this->no_of_second_order_fields());
  for (unsigned int i=0; i<no_of_first_order_fields(); i++) {
    first_order_subm[i].resize(i+1);
  }
  if (_order == 2) { // Only allocate if actually used
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      second_order_subm[i].resize(i+1);
      cross_subm[i].resize(this->no_of_first_order_fields());
    }
  }

  // Then calculate the actual components (sub-matrices) of the Hessian
  calculate_first_order_subm(_sum_fo_deriv_deriv,_sum_fo_ima_ima,_sum_fo_deriv_ima,cmask(),nvox(),first_order_subm);
  if (_order == 2) {
    calculate_second_order_subm(_sum_so_deriv_deriv,_sum_so_ima_ima,_sum_so_deriv_ima,cmask(),nvox(),second_order_subm);
    calculate_cross_subm(_sum_cross_deriv_deriv,_sum_cross_ima_ima,_sum_cross_deriv_ima,cmask(),nvox(),cross_subm);
  }

  // Add regularisation to diagonal blocks
  std::shared_ptr<MISCMATHS::BFMatrix> reg = _sp1[0]->BendEnergyHess(hessian_precision());
  for (unsigned int i=0; i<this->no_of_first_order_fields(); i++) {
    first_order_subm[i][i]->AddToMe(*reg,lambda()/static_cast<double>(nvox()*no_of_fields()));
  }
  for (unsigned int i=0; i<this->no_of_second_order_fields(); i++) {
    second_order_subm[i][i]->AddToMe(*reg,lambda()/static_cast<double>(nvox()*no_of_fields()));
  }

  // Finally concatenate them together
  std::shared_ptr<MISCMATHS::BFMatrix> first_order_subm_cat = concatenate_subdiag_subm(first_order_subm,no_of_first_order_fields());
  first_order_subm.clear(); // Free up memory
  std::shared_ptr<MISCMATHS::BFMatrix> rval = first_order_subm_cat;
  // rval->Print("hessian.txt");
  // exit(1);
  if (_order == 2) {
    std::shared_ptr<MISCMATHS::BFMatrix> second_order_subm_cat = concatenate_subdiag_subm(second_order_subm,no_of_second_order_fields());
    second_order_subm.clear(); // Free up memory
    std::shared_ptr<MISCMATHS::BFMatrix> cross_subm_cat = concatenate_rect_subm(cross_subm,no_of_second_order_fields(),no_of_first_order_fields());
    cross_subm.clear(); // Free up memory
    rval->HorConcat2MyRight(*(cross_subm_cat->Transpose()));
    cross_subm_cat->HorConcat2MyRight(*second_order_subm_cat);
    rval->VertConcatBelowMe(*cross_subm_cat);
  }

  if (_chH) { // If we are to check Hessian
    bool old_chgr = false;
    if (_chgr) { _chgr = false; old_chgr = true; }
    char fname[256]; sprintf(fname,"hessian_%02d.txt",_hess_cnt);
    rval->Print(fname);

    NEWMAT::ColumnVector tmp_p = p;
    NEWMAT::ColumnVector grad0 = this->grad(tmp_p);
    unsigned int no_of_values = 20;
    NEWMAT::Matrix numhess(no_of_values*total_no_of_parameters(),3); numhess=0.0;
    unsigned int step = total_no_of_parameters() / (no_of_values + 1);
    double delta = 0.01;
    unsigned int ii = 0;
    for (unsigned int i=step/2; i<total_no_of_parameters(); i+=step) {
      if (ii<no_of_values) {
	tmp_p(i) += delta;
	NEWMAT::ColumnVector grad1 = this->grad(tmp_p);
	unsigned int fr = (ii)*total_no_of_parameters()+1;
	unsigned int lr = (ii+1)*total_no_of_parameters();
	numhess.SubMatrix(fr,lr,3,3) = (grad1 - grad0) / delta;
	numhess.SubMatrix(fr,lr,2,2) = i;
	numhess.SubMatrix(fr,lr,1,1) = this->linspace(numhess.SubMatrix(fr,lr,1,1));
	tmp_p(i) -= delta;
	ii++;
      }
    }
    sprintf(fname,"numhess_delta_01_%02d.txt",_hess_cnt);
    MISCMATHS::write_ascii_matrix(numhess,fname);
    _chgr = old_chgr;
  }
  _hess_cnt++;

  return(rval);
} EddyCatch

std::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::concatenate_subdiag_subm(const std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
										  unsigned int                                                              n) const EddyTry
{
  // Concatenate each row of submatrices left->right
  std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > tmp(n);
  for (unsigned int i=0; i<n; i++) {
    for (unsigned int j=0; j<n; j++) {
      if (!j) {
	if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	  const SparseBFMatrix<float>& tmp2 = dynamic_cast<SparseBFMatrix<float>& >(*subm[i][0]);
	  tmp[i] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp2));
	}
	else {
	  const SparseBFMatrix<double>& tmp2 = dynamic_cast<SparseBFMatrix<double>& >(*subm[i][0]);
	  tmp[i] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp2));
	}
      }
      else {
	if (j <= i) tmp[i]->HorConcat2MyRight(*subm[i][j]);
	else tmp[i]->HorConcat2MyRight(*(subm[j][i]->Transpose()));
      }
    }
  }
  // Concatenate the rows top->bottom
  std::shared_ptr<MISCMATHS::BFMatrix> rval = tmp[0];
  for (unsigned int i=1; i<n; i++) rval->VertConcatBelowMe(*tmp[i]);

  return(rval);
} EddyCatch

std::shared_ptr<MISCMATHS::BFMatrix> MoveBySuscCFImpl::concatenate_rect_subm(const std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm,
									       unsigned int                                                              m,
									       unsigned int                                                              n) const EddyTry
{
  // Concatenate each row of submatrices left->right
  std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > tmp(m);
  for (unsigned int i=0; i<m; i++) {
    for (unsigned int j=0; j<n; j++) {
      if (!j) {
	if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	  const SparseBFMatrix<float>& tmp2 = dynamic_cast<SparseBFMatrix<float>& >(*subm[i][0]);
	  tmp[i] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp2));
	}
	else {
	  const SparseBFMatrix<double>& tmp2 = dynamic_cast<SparseBFMatrix<double>& >(*subm[i][0]);
	  tmp[i] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp2));
	}
      }
      else tmp[i]->HorConcat2MyRight(*subm[i][j]);
    }
  }
  // Concatenate the rows top->bottom
  std::shared_ptr<MISCMATHS::BFMatrix> rval = tmp[0];
  for (unsigned int i=1; i<m; i++) rval->VertConcatBelowMe(*tmp[i]);

  return(rval);
} EddyCatch

void MoveBySuscCFImpl::calculate_first_order_subm(// Input
						  const std::vector<ImageVec>&                                        sop_dfdf,
						  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
						  const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
						  const NEWIMAGE::volume<char>&                                       lmask,
						  unsigned int                                                        nvox,
						  // Output
						  std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  std::vector<unsigned int> noderiv(3,0); // No differentiation
  NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
  for (unsigned int i=0; i<no_of_first_order_fields(); i++) {
    for (unsigned int j=0; j<(i+1); j++) {
      subm[i][j] = _sp1[i]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
      subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
      if (_ujm) {
	for (const auto& elem : sop_ff) {
	  subm[i][j]->AddToMe(*_sp1[i]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),2.0/static_cast<double>(nvox));
	}
	for (const auto& elem: sop_fdf) {
	  std::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp1[i]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	  subm[i][j]->AddToMe(*tmp,2.0/static_cast<double>(nvox));
	  subm[i][j]->AddToMe(*(tmp->Transpose()),2.0/static_cast<double>(nvox));
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::calculate_second_order_subm(// Input
						   const std::vector<ImageVec>&                                        sop_dfdf,
						   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
						   const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
						   const NEWIMAGE::volume<char>&                                       lmask,
						   unsigned int                                                        nvox,
						   // Output
						   std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  if (_order == 2)
  {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_second_order_pmap();
    std::vector<unsigned int> noderiv(3,0); // No differentiation
    NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
    // First pass to calculate all unique sub-matrices
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { // If this is the first instance of this index combination
	  subm[i][j] = _sp2[i][j]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
	  subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
	  if (_ujm) {
	    for (const auto& elem : sop_ff) {
	      subm[i][j]->AddToMe(*_sp2[i][j]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),(2.0/static_cast<double>(nvox)));
	    }
	    for (const auto& elem : sop_fdf) {
	      std::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp2[i][j]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	      subm[i][j]->AddToMe(*tmp,(2.0/static_cast<double>(nvox)));
	      subm[i][j]->AddToMe(*(tmp->Transpose()),(2.0/static_cast<double>(nvox)));
	    }
	  }
	}
      }
    }
    // Second pass to copy pointers into non-unique locations
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (!is_first_index_pair(i,j,pindx,pmap)) { // If this is NOT the first instance of this index combination
	  std::pair<unsigned int,unsigned int> iijj = find_first_index_pair(pindx,pmap);
	  if (i==j || iijj.first==iijj.second) { // Make deep copy if on diagonal
	    if (_hp == MISCMATHS::BFMatrixFloatPrecision) {
	      const SparseBFMatrix<float>& tmp = dynamic_cast<SparseBFMatrix<float>& >(*(subm[iijj.first][iijj.second]));
	      subm[i][j] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<float>(tmp));
	    }
	    else {
	      const SparseBFMatrix<double>& tmp = dynamic_cast<SparseBFMatrix<double>& >(*(subm[iijj.first][iijj.second]));
	      subm[i][j] = std::shared_ptr<MISCMATHS::BFMatrix>(new MISCMATHS::SparseBFMatrix<double>(tmp));
	    }
	  }
	  else subm[i][j] = subm[iijj.first][iijj.second];
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::calculate_second_order_subm: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::calculate_cross_subm(// Input
					    const std::vector<ImageVec>&                                        sop_dfdf,
					    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_ff,
					    const std::map<BpeVec,std::vector<ImageVec> >&                      sop_fdf,
					    const NEWIMAGE::volume<char>&                                       lmask,
					    unsigned int                                                        nvox,
					    // Output
					    std::vector<std::vector<std::shared_ptr<MISCMATHS::BFMatrix> > >& subm) const EddyTry
{
  if (_order == 2)
  {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_cross_pmap();
    std::vector<unsigned int> noderiv(3,0); // No differentiation
    NEWIMAGE::volume<float> ones = sop_dfdf[0][0]; ones = 1.0;
    // First pass to calculate all unique sub-matrices
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { // If this is the first instance of this index combination
	  subm[i][j] = _sp2[i][j]->JtJ(sop_dfdf[i][j],ones,&lmask,hessian_precision());
	  subm[i][j]->MulMeByScalar(2.0/static_cast<double>(nvox));
	  if (_ujm) {
	    for (const auto& elem : sop_ff) {
	      subm[i][j]->AddToMe(*_sp2[i][j]->JtJ(elem.first,elem.second[i][j],ones,&lmask,hessian_precision()),2.0/static_cast<double>(nvox));
	    }
	    for (const auto& elem : sop_fdf) {
	      std::shared_ptr<MISCMATHS::BFMatrix> tmp = _sp2[i][j]->JtJ(elem.first,elem.second[i][j],noderiv,ones,&lmask,hessian_precision());
	      subm[i][j]->AddToMe(*tmp,2.0/static_cast<double>(nvox));
	      subm[i][j]->AddToMe(*(tmp->Transpose()),2.0/static_cast<double>(nvox));
	    }
	  }
	}
      }
    }
    // Second pass to copy pointers into non-unique locations
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (!is_first_index_pair(i,j,pindx,pmap)) { // If this is NOT the first instance of this index combination
	  std::pair<unsigned int,unsigned int> iijj = find_first_index_pair(pindx,pmap);
	  subm[i][j] = subm[iijj.first][iijj.second];
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::calculate_cross_subm: I should not be here.");
  return;
} EddyCatch

bool MoveBySuscCFImpl::is_first_index_pair(unsigned int                                                 i,
					   unsigned int                                                 j,
					   const std::vector<unsigned int>&                             ivec,
					   const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const EddyTry
{
  if (pmap[i][j] != ivec) throw EddyException("MoveBySuscCFImpl::is_first_index_pair: ivec is not the i-j'th member of pmap.");
  std::pair<unsigned int,unsigned int> first = find_first_index_pair(ivec,pmap);
  if (first.first==i && first.second==j) return(true);
  return(false);
} EddyCatch

std::pair<unsigned int,unsigned int> MoveBySuscCFImpl::find_first_index_pair(const std::vector<unsigned int>&                             ivec,
									     const std::vector<std::vector<std::vector<unsigned int> > >& pmap) const EddyTry
{
  std::pair<unsigned int,unsigned int> first(0,0);
  for ( ; first.first<pmap.size(); first.first++) {
    for ( ; first.second<pmap[first.first].size(); first.second++) {
      if (pmap[first.first][first.second] == ivec) return(first);
    }
  }
  throw EddyException("MoveBySuscCFImpl::find_first_index_pair: ivec is not a member of pmap.");
} EddyCatch

std::vector<std::vector<std::vector<unsigned int> > > MoveBySuscCFImpl::build_second_order_pmap() const EddyTry
{
  std::vector<std::vector<std::vector<unsigned int> > > pmap(no_of_second_order_fields());
  for (unsigned int i=0; i<pmap.size(); i++) {
    std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
    pmap[i].resize(i+1);
    for (unsigned j=0; j<(i+1); j++) {
      std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
      pmap[i][j] = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
    }
  }
  return(pmap);
} EddyCatch

std::vector<std::vector<std::vector<unsigned int> > > MoveBySuscCFImpl::build_cross_pmap() const EddyTry
{
  std::vector<std::vector<std::vector<unsigned int> > > pmap(no_of_second_order_fields());
  for (unsigned int i=0; i<pmap.size(); i++) {
    std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
    pmap[i].resize(no_of_first_order_fields());
    for (unsigned j=0; j<no_of_first_order_fields(); j++) {
      pmap[i][j] = make_index_vector(first_pair,j,no_of_first_order_fields());
    }
  }
  return(pmap);
} EddyCatch

std::vector<unsigned int> MoveBySuscCFImpl::make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
							      const std::pair<unsigned int,unsigned int>& p2,
							      unsigned int                                np) const EddyTry
{
  std::vector<unsigned int> rval(np,0);
  rval[p1.first]++; rval[p1.second]++; rval[p2.first]++; rval[p2.second]++;
  return(rval);
} EddyCatch

std::vector<unsigned int> MoveBySuscCFImpl::make_index_vector(const std::pair<unsigned int,unsigned int>& p1,
							      unsigned int                                i,
							      unsigned int                                np) const EddyTry
{
  std::vector<unsigned int> rval(np,0);
  rval[p1.first]++; rval[p1.second]++; rval[i]++;
  return(rval);
} EddyCatch

std::pair<unsigned int,unsigned int> MoveBySuscCFImpl::get_second_order_index_pair(unsigned int i,
										   unsigned int np) const EddyTry
{
  std::pair<unsigned int,unsigned int> indx;
  for (indx.first = 0; indx.first<np; indx.first++) {
    for (indx.second = 0; indx.second<(indx.first+1); indx.second++) {
      if (indx.first + indx.second == i) return(indx);
    }
  }
  throw EddyException("MoveBySuscCFImpl::get_second_order_index_pair: I should not be here");
} EddyCatch

NEWMAT::ColumnVector MoveBySuscCFImpl::stl2newmat(const std::vector<double>& stl) const EddyTry
{
  NEWMAT::ColumnVector nm(stl.size());
  for (unsigned int i=0; i<stl.size(); i++) nm(i+1) = stl[i];
  return(nm);
} EddyCatch

NEWMAT::Matrix MoveBySuscCFImpl::linspace(const NEWMAT::Matrix& inmat) const EddyTry
{
  NEWMAT::Matrix rval(inmat.Nrows(),inmat.Ncols());
  for (int r=1; r<=inmat.Nrows(); r++) rval(r,1) = r;
  return(rval);
} EddyCatch

void MoveBySuscCFImpl::resize_containers_for_sum_images_for_grad(const std::vector<std::vector<unsigned int> >& indicies,
								 const std::vector<EDDY::ScanType>&             imtypes) EddyTry
{
  _sum_fo_deriv_diff.resize(_mps.size());
  if (_order==2) {
    _sum_so_deriv_diff.resize(_mps.size());
  }
  if (_ujm) {
    for (unsigned int ti=0; ti<indicies.size(); ti++) {
      for (unsigned int i=0; i<indicies[ti].size(); i++) {
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	if (_sum_fo_ima_diff.find(bpe) == _sum_fo_ima_diff.end()) { // If this acqp has not been seen before
	  _sum_fo_ima_diff[bpe].resize(_mps.size());
	  if (_order==2) {
	    _sum_so_ima_diff[bpe].resize(_mps.size());
	    for (unsigned int pi=0; pi<_mps.size(); pi++) {
	      _sum_so_ima_diff[bpe][pi].resize(pi+1);
	    }
	  }
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::resize_containers_for_sum_images_for_hess(const std::vector<std::vector<unsigned int> >& indicies,
								 const std::vector<EDDY::ScanType>&             imtypes) EddyTry
{
  // Start with the ones used for a first-order expansion
  _sum_fo_deriv_deriv.resize(_mps.size());
  for (unsigned int pi=0; pi<_mps.size(); pi++) _sum_fo_deriv_deriv[pi].resize(pi+1);
  if (_ujm) {
    for (unsigned int ti=0; ti<indicies.size(); ti++) {
      for (unsigned int i=0; i<indicies[ti].size(); i++) {
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	if (_sum_fo_ima_ima.find(bpe) == _sum_fo_ima_ima.end()) { // If this acqp has not been seen before
	  _sum_fo_ima_ima[bpe].resize(_mps.size());
	  _sum_fo_deriv_ima[bpe].resize(_mps.size());
	  for (unsigned int pi=0; pi<_mps.size(); pi++) {
	    _sum_fo_ima_ima[bpe][pi].resize(pi+1);
	    _sum_fo_deriv_ima[bpe][pi].resize(pi+1);
	  }
	}
      }
    }
  }
  // Then do the ones used for a second-order expansion
  if (_order == 2) {
    _sum_so_deriv_deriv.resize(no_of_second_order_fields());
    _sum_cross_deriv_deriv.resize(no_of_second_order_fields());
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      _sum_so_deriv_deriv[i].resize(i+1);
      _sum_cross_deriv_deriv.resize(no_of_first_order_fields());
    }
    if (_ujm) {
      for (unsigned int ti=0; ti<indicies.size(); ti++) {
	for (unsigned int i=0; i<indicies[ti].size(); i++) {
	  NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
	  BpeVec bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
	  if (_sum_so_ima_ima.find(bpe) == _sum_so_ima_ima.end()) { // If this acqp has not been seen before
	    _sum_so_ima_ima[bpe].resize(no_of_second_order_fields());
	    _sum_so_deriv_ima[bpe].resize(no_of_second_order_fields());
	    _sum_cross_ima_ima[bpe].resize(no_of_second_order_fields());
	    _sum_cross_deriv_ima[bpe].resize(no_of_second_order_fields());
	    for (unsigned j=0; j<no_of_second_order_fields(); j++) {
	      _sum_so_ima_ima[bpe][j].resize(j+1);
	      _sum_so_deriv_ima[bpe][j].resize(j+1);
	      _sum_cross_ima_ima[bpe][j].resize(no_of_first_order_fields());
	      _sum_cross_deriv_ima[bpe][j].resize(no_of_first_order_fields());
	    }
	  }
	}
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::set_sum_images_to_zero(const NEWIMAGE::volume<float>& ima) EddyTry
{
  NEWIMAGE::volume<float> zeros = ima; zeros = 0.0;
  // Field for .cf()
  _sum_sqr_diff = zeros;
  // Fields for .grad()
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    _sum_fo_deriv_diff[pi] = zeros;
    for (auto& elem : _sum_fo_ima_diff) elem.second[pi] = zeros;
    if (_order == 2) {
      for (unsigned int pj=0; pj<(pi+1); pj++) {
	_sum_so_deriv_diff[pi][pj] = zeros;
	for (auto& elem : _sum_so_ima_diff) elem.second[pi][pj] = zeros;
      }
    }
  }
  // Fields for .hess()
  for (unsigned int pi=0; pi<_mps.size(); pi++) {
    for (unsigned int pj=0; pj<(pi+1); pj++) {
      _sum_fo_deriv_deriv[pi][pj] = zeros;
      for (auto& elem : _sum_fo_ima_ima) elem.second[pi][pj] = zeros;
      for (auto& elem : _sum_fo_deriv_ima) elem.second[pi][pj] = zeros;
    }
  }
  if (_order == 2) {
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      for (unsigned int j=0; j<(i+1); j++) {
	_sum_so_deriv_deriv[i][j] = zeros;
	for (auto& elem : _sum_so_ima_ima) elem.second[i][j] = zeros;
	for (auto& elem : _sum_so_deriv_ima) elem.second[i][j] = zeros;
      }
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	_sum_cross_deriv_deriv[i][j] = zeros;
	for (auto& elem : _sum_cross_ima_ima) elem.second[i][j] = zeros;
	for (auto& elem : _sum_cross_deriv_ima) elem.second[i][j] = zeros;
      }
    }
  }
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_so_imas_for_hess(const NEWMAT::ColumnVector&           p,
							const BpeVec&                         bpe,
							const NEWIMAGE::volume<float>&        ima,
							const NEWIMAGE::volume<float>&        deriv) EddyTry
{
  if (!_utd && _order==2) {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_second_order_pmap();
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<(i+1); j++) {
	std::pair<unsigned int,unsigned int> second_pair = get_second_order_index_pair(j,no_of_second_order_fields());
	std::vector<unsigned int> pindx = make_index_vector(first_pair,second_pair,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { // If this is the first instance of this index combination
	  float factor=1.0;
	  for (unsigned int ii=0; ii<pindx.size(); ii++) {
	    for (unsigned int jj=0; jj<pindx[ii]; jj++) factor *= p(_mps[ii]+1);
	  }
	  // Voxel displacement term
	  _sum_so_deriv_deriv[i][j] += factor*deriv*deriv;
	  if (_ujm) {
	    // Jacobian modulation term
	    _sum_so_ima_ima[bpe][i][j] += factor*ima*ima;
	    // Cross term
	    _sum_so_deriv_ima[bpe][i][j] += factor*ima*deriv;
	  }
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::recalculate_sum_so_imas_for_hess: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_cross_imas_for_hess(const NEWMAT::ColumnVector&           p,
							   const BpeVec&                         bpe,
							   const NEWIMAGE::volume<float>&        ima,
							   const NEWIMAGE::volume<float>&        deriv) EddyTry
{
  if (!_utd && _order==2) {
    std::vector<std::vector<std::vector<unsigned int> > > pmap = build_cross_pmap();
    for (unsigned int i=0; i<no_of_second_order_fields(); i++) {
      std::pair<unsigned int,unsigned int> first_pair = get_second_order_index_pair(i,no_of_second_order_fields());
      for (unsigned int j=0; j<no_of_first_order_fields(); j++) {
	std::vector<unsigned int> pindx = make_index_vector(first_pair,j,no_of_first_order_fields());
	if (is_first_index_pair(i,j,pindx,pmap)) { // If this is the first instance of this index combination
	  float factor=1.0;
	  for (unsigned int ii=0; ii<pindx.size(); ii++) {
	    for (unsigned int jj=0; jj<pindx[ii]; jj++) factor *= p(_mps[ii]+1);
	  }
	  // Voxel displacement term
	  _sum_cross_deriv_deriv[i][j] += factor*deriv*deriv;
	  if (_ujm) {
	    // Jacobian modulation term
	    _sum_cross_ima_ima[bpe][i][j] += factor*ima*ima;
	    // Cross term
	    _sum_cross_deriv_ima[bpe][i][j] += factor*ima*deriv;
	  }
	}
      }
    }
  }
  else throw EddyException("MoveBySuscCFImpl::recalculate_sum_cross_imas_for_hess: I should not be here.");
  return;
} EddyCatch

void MoveBySuscCFImpl::recalculate_sum_images() EddyTry
{
  if (!_utd || !_m_utd) {
    // Make some temporary vectors to allow us to use the same code for B0 and DWI.
    std::vector<std::vector<unsigned int> > indicies = {_b0s, _dwis};
    std::vector<EDDY::ScanType> imtypes = {B0, DWI};
    std::vector<NEWIMAGE::volume<float> > masks = {_sm.Scan(0,ANY).GetIma(), _sm.Scan(0,ANY).GetIma()};
    std::vector<std::shared_ptr<EDDY::DWIPredictionMaker> > pmps(2,nullptr);
    if (_bno) { // If Brand New Object
      resize_containers_for_sum_images_for_grad(indicies,imtypes);
      resize_containers_for_sum_images_for_hess(indicies,imtypes);
      // Load up prediction maker and estimate (and save) hyper-parameters
      if (_clo.NVoxHp() < 10000) _clo.SetNVoxHp(10000); // Make sure it is done well
#ifdef COMPILE_GPU // If it shall be compiled for GPU
      if (indicies[0].size()) pmps[0] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); // B0 predictor
      if (indicies[1].size()) {  // DWI predictor
	pmps[1] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]);
	_hypar = this->stl2newmat(pmps[1]->GetHyperPar());
      }
#else
      if (indicies[0].size()) pmps[0] = EDDY::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); // B0 predictor
      if (indicies[1].size()) {  // DWI predictor
	pmps[1] = EDDY::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]);
	_hypar = this->stl2newmat(pmps[1]->GetHyperPar());
      }
#endif
      _bno = false;
    }
    else {
      // Load up prediction maker and use saved hyper-parameters
      _clo.SetHyperParValues(_hypar); // Clunk!
#ifdef COMPILE_GPU // If it shall be compiled for GPU
      if (indicies[0].size()) pmps[0] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); // B0 predictor
      if (indicies[1].size()) pmps[1] = EddyGpuUtils::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]); // DWI predictor
#else
      if (indicies[0].size()) pmps[0] = EDDY::LoadPredictionMaker(_clo,imtypes[0],_sm,0,0.0,masks[0]); // B0 predictor
      if (indicies[1].size()) pmps[1] = EDDY::LoadPredictionMaker(_clo,imtypes[1],_sm,0,0.0,masks[1]); // DWI predictor
#endif
    }

    // Make mask
    _mask = masks[0] * masks[1];
    _nvox = static_cast<unsigned int>(std::round(_mask.sum()));
    _cmask.reinitialize(_mask.xsize(),_mask.ysize(),_mask.zsize());
    NEWIMAGE::copybasicproperties(_mask,_cmask);
    std::vector<int> tsz = {static_cast<int>(_mask.xsize()), static_cast<int>(_mask.ysize()), static_cast<int>(_mask.zsize())};
    for (int k=0; k<tsz[2]; k++) for (int j=0; j<tsz[1]; j++) for (int i=0; i<tsz[0]; i++) _cmask(i,j,k) = (_mask(i,j,k) > 0.0) ? 1 : 0;

    // This is where we start to generate the sum images
    // First set them all to zero
    set_sum_images_to_zero(_mask);

    // And then start doing the actual summing
    NEWIMAGE::volume4D<float> deriv(_mask.xsize(),_mask.ysize(),_mask.zsize(),3);
    NEWIMAGE::copybasicproperties(_mask,deriv);
    NEWIMAGE::volume<float> sderiv = _mask;  // Scalar "derivative" image
    std::vector<double> vxs = { _mask.xdim(), _mask.ydim(), _mask.zdim() };
    for (unsigned int ti=0; ti<indicies.size(); ti++) {  // Loop over b0 and DWI
      for (unsigned int i=0; i<indicies[ti].size(); i++) { // Loop over all images
	NEWMAT::ColumnVector p = _sm.Scan(indicies[ti][i],imtypes[ti]).GetParams(EDDY::ZERO_ORDER_MOVEMENT);
        NEWMAT::ColumnVector hz2mm = _sm.Scan(indicies[ti][i],imtypes[ti]).GetHz2mmVector();
	std::vector<unsigned int> bpe = _sm.Scan(indicies[ti][i],imtypes[ti]).GetAcqPara().BinarisedPhaseEncodeVector();
#ifdef COMPILE_GPU // If it shall be compiled for GPU
	NEWIMAGE::volume<float> vol = EddyGpuUtils::GetVolumetricUnwarpedScan(_sm.Scan(indicies[ti][i],imtypes[ti]),_sm.GetSuscHzOffResField(indicies[ti][i],imtypes[ti]),_sm.GetBiasField(),false,nullptr,&deriv);
#else
	NEWIMAGE::volume<float> vol = _sm.Scan(indicies[ti][i],imtypes[ti]).GetVolumetricUnwarpedIma(_sm.GetSuscHzOffResField(indicies[ti][i],imtypes[ti]),deriv);
#endif
	NEWIMAGE::volume<float> diff = vol - pmps[ti]->Predict(indicies[ti][i]);
	_sum_sqr_diff += diff * diff;
	sderiv = static_cast<float>(hz2mm(1)/vxs[0])*deriv[0] + static_cast<float>(hz2mm(2)/vxs[1])*deriv[1] + static_cast<float>(hz2mm(3)/vxs[2])*deriv[2];
        if (_ujm) vol *= static_cast<float>(hz2mm(1)/vxs[0] + hz2mm(2)/vxs[1] + hz2mm(3)/vxs[2]); // Assumes only one of hz2mm is non-zero
        // First sum the images used for .grad()
	for (unsigned int pi=0; pi<_mps.size(); pi++) {
	  _sum_fo_deriv_diff[pi] += static_cast<float>(p(_mps[pi]+1))*sderiv*diff;
	  if (_ujm) _sum_fo_ima_diff[bpe][pi] += static_cast<float>(p(_mps[pi]+1))*vol*diff;
	  if (_order == 2) {
	    for (unsigned int pj=0; pj<(pi+1); pj++) {
	      _sum_so_deriv_diff[pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*diff;
	      if (_ujm) _sum_so_ima_diff[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*vol*diff;
	    }
	  }
	}
	// Then sum the images used for .hess()
	for (unsigned int pi=0; pi<_mps.size(); pi++) {
	  for (unsigned int pj=0; pj<(pi+1); pj++) {
	    _sum_fo_deriv_deriv[pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*sderiv;
	    if (_ujm) {
	      _sum_fo_ima_ima[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*vol*vol;
	      _sum_fo_deriv_ima[bpe][pi][pj] += static_cast<float>(p(_mps[pi]+1)*p(_mps[pj]+1))*sderiv*vol;
	    }
	  }
	}
	if (_order == 2) { // Farm out 2nd order summation
	  recalculate_sum_so_imas_for_hess(p,bpe,vol,sderiv);     // Second-order sub-matrix of total Hessian
	  recalculate_sum_cross_imas_for_hess(p,bpe,vol,sderiv);  // Cross-term between first and second order sub-matrix
	}
      }
    }
    _m_utd = true;
    _utd = true;
  }

  return;
} EddyCatch

} // End namespace EDDY
