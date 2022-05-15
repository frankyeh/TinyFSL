/////////////////////////////////////////////////////////////////////
/// \file ECScanClasses.h
/// \brief Declarations of classes that implements a scan or a collection of scans within the EC project.
///
/// \author Jesper Andersson
/// \version 1.0b, Sep., 2012.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////
//
// Copyright (C) 2011 University of Oxford
//
#pragma GCC diagnostic ignored "-Wunknown-pragmas" // Ignore the OpenMP pragmas

#ifndef ECScanClasses_h
#define ECScanClasses_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

namespace EDDY {

/////////////////////////////////////////////////////////////////////
///
/// \brief This class manages one diffusion weighted or b=0 scan
///  for the eddy project.
///
/////////////////////////////////////////////////////////////////////
class ECScan
{
public:
  /// Constructs an object from volume, acquisition parameters, diffusion parameters, EC-Model and session number.
  ECScan(const NEWIMAGE::volume<float>&   ima,
         const AcqPara&                   acqp,
         const DiffPara&                  diffp,
	 const ScanMovementModel&         mp,
	 const MultiBandGroups&           mbg,
         std::shared_ptr<ScanECModel>     ecp,
	 double                           mrl=1.0) EddyTry : _ima(ima), _ols(ima.zsize(),0ul), _acqp(acqp), _diffp(diffp), _mp(mp), _mbg(mbg), _ecp(ecp->Clone()), _pp(EDDY::PolationPara()), _mrl(mrl) {} EddyCatch
  /// Copy constructor.
  ECScan(const ECScan& inp) EddyTry : _ima(inp._ima), _ols(inp._ols), _acqp(inp._acqp), _diffp(inp._diffp), _mp(inp._mp), _mbg(inp._mbg), _ecp(inp._ecp->Clone()), _pp(inp._pp), _mrl(inp._mrl) {
    for (int sl=0; sl<_ima.zsize(); sl++) {
      if (_ols[sl]) { _ols[sl] = new float[_ima.xsize()*_ima.ysize()]; std::memcpy(_ols[sl],inp._ols[sl],_ima.xsize()*_ima.ysize()*sizeof(float)); }
    }
  } EddyCatch
  virtual ~ECScan() { for (int sl=0; sl<_ima.zsize(); sl++) { if (_ols[sl]) delete[] _ols[sl]; } }
  /// Assignment.
  ECScan& operator=(const ECScan& rhs) EddyTry {
    if (this == &rhs) return(*this);
    _ima=rhs._ima; _ols=rhs._ols; _acqp=rhs._acqp; _diffp=rhs._diffp; _mp=rhs._mp; _mbg=rhs._mbg; _ecp=rhs._ecp->Clone(); _pp=rhs._pp; _mrl=rhs._mrl;
    for (int sl=0; sl<_ima.zsize(); sl++) {
      if (_ols[sl]) { _ols[sl] = new float[_ima.xsize()*_ima.ysize()]; std::memcpy(_ols[sl],rhs._ols[sl],_ima.xsize()*_ima.ysize()*sizeof(float)); }
    }
    return(*this);
  } EddyCatch
  /// Returns the EC model being used.
  ECModel Model() const EddyTry { return(_ecp->WhichModel()); } EddyCatch
  /// Set movement model order (0 for rigid body model).
  void SetMovementModelOrder(unsigned int order) EddyTry { if (int(order)>_ima.zsize()) throw EddyException("ECScan::SetMovementModelOrder: order too high"); else _mp.SetOrder(order); } EddyCatch
  /// Get movement model order (0 for rigid body model)
  unsigned int GetMovementModelOrder() const EddyTry { return(_mp.Order()); } EddyCatch
  /// Get read/write reference to movement model
  ScanMovementModel& GetMovementModel() EddyTry { return(_mp); } EddyCatch
  /// Returns true if model order > 0
  bool IsSliceToVol() const EddyTry { return(GetMovementModelOrder()); } EddyCatch
  /// Returns across slice/group std of movement parameter
  double GetMovementStd(unsigned int mi) const EddyTry { std::vector<unsigned int> empty; return(this->GetMovementStd(mi,empty)); } EddyCatch
  /// Returns across slice std of movement parameter. Only valid to call with non-empty icsl if no multi-band
  double GetMovementStd(unsigned int mi, std::vector<unsigned int> icsl) const;
  /// Returns multi-band info
  const MultiBandGroups& GetMBG() const EddyTry { return(_mbg); } EddyCatch
  /// Returns true if any slices has been replaced
  bool HasOutliers() const EddyTry { for (unsigned int i=0; i<_ols.size(); i++) { if (_ols[i]) return(true); } return(false); } EddyCatch
  /// Returns true if slice sl has been replaced
  bool IsOutlier(unsigned int sl) const EddyTry { if (_ols.at(sl)) return(true); else return(false); } EddyCatch
  /// Will return true if an offset (DC part) is included in the EC model.
  bool HasFieldOffset() const EddyTry { return(_ecp->HasFieldOffset()); } EddyCatch
  /// Returns the field offset (in Hz).
  double GetFieldOffset() const EddyTry { return(_ecp->GetFieldOffset()); } EddyCatch
  /// Sets the field offset. The value of ofst should be in Hz.
  void SetFieldOffset(double ofst) EddyTry { _ecp->SetFieldOffset(ofst); } EddyCatch
  /// Get inter/extrapolation options
  PolationPara GetPolation() const EddyTry { return(_pp); } EddyCatch
  /// Set inter/extrapolation options
  void SetPolation(const PolationPara& pp);
  /// Checks for the presence of empty planes in the x- and y-directions.
  bool HasEmptyPlane(std::vector<unsigned int>&  pi) const;
  /// Fills empty planes in the x- and y-directions.
  void FillEmptyPlane(const std::vector<unsigned int>&  pi);
  /// Returns the original ("unreplaced" and untransformed) volume.
  NEWIMAGE::volume<float> GetOriginalIma() const;
  /// Returns the original ("unreplaced") volume after correction for rigid-body movement.
  NEWIMAGE::volume<float> GetMotionCorrectedOriginalIma(NEWIMAGE::volume<float>& omask) const EddyTry { return(motion_correct(GetOriginalIma(),&omask)); } EddyCatch
  /// Returns the original ("unreplaced") volume after correction for rigid-body movement.
  NEWIMAGE::volume<float> GetMotionCorrectedOriginalIma() const EddyTry { return(motion_correct(GetOriginalIma(),NULL)); } EddyCatch
  /// Returns the original (unsmoothed) volume after correction for rigid-body movement and EC distortions.
  NEWIMAGE::volume<float> GetUnwarpedOriginalIma(// Input
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 const NEWIMAGE::volume<float>&                    pred,
						 // Output
						 NEWIMAGE::volume<float>&                          omask) const;
/// Returns the original (unsmoothed) volume after correction for rigid-body movement and EC distortions.
  NEWIMAGE::volume<float> GetUnwarpedOriginalIma(// Input
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						 // Output
						 NEWIMAGE::volume<float>&                          omask) const;
  /// Returns the original (unsmoothed) volume after correction for rigid-body movement and EC distortions.
  NEWIMAGE::volume<float> GetUnwarpedOriginalIma(// Input
						 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const;
  /// Returns the volume (with replacements) in the original space (no corrections)
  const NEWIMAGE::volume<float>& GetIma() const EddyTry { return(_ima); } EddyCatch
  /// Returns the ("outlier replaced") volume after correction for rigid-body movement.
  NEWIMAGE::volume<float> GetMotionCorrectedIma(NEWIMAGE::volume<float>& omask) const EddyTry { return(motion_correct(GetIma(),&omask)); } EddyCatch
  /// Returns the ("outlier replaced") volume after correction for rigid-body movement.
  NEWIMAGE::volume<float> GetMotionCorrectedIma() const EddyTry { return(motion_correct(GetIma(),NULL)); } EddyCatch
  /// Returns the smoothed and corrected (for movement and EC distortions) volume.
  NEWIMAGE::volume<float> GetUnwarpedIma(// Input
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					 const NEWIMAGE::volume<float>                     pred,
					 // Output
					 NEWIMAGE::volume<float>&                          omask) const;
  /// Returns the smoothed and corrected (for movement and EC distortions) volume.
  NEWIMAGE::volume<float> GetUnwarpedIma(// Input
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
					 // Output
					 NEWIMAGE::volume<float>&                          omask) const;
  /// Returns the smoothed and corrected (for movement and EC distortions) volume.
  NEWIMAGE::volume<float> GetUnwarpedIma(// Input
					 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) const;
  /// Return volumetrically unwarped ima (for when s2v is implemented on the CPU) with optional derivative images
  NEWIMAGE::volume<float> GetVolumetricUnwarpedIma(// Input
						   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						   // Output
						   NEWIMAGE::volume4D<float>&                        deriv) const;
  /// Returns the acquisition parameters for the scan
  AcqPara GetAcqPara() const EddyTry { return(_acqp); } EddyCatch
  /// Returns the diffusion parameters for the scan
  DiffPara GetDiffPara(bool rot=false) const;
  /// Return the diffusion parameters for a slice within a scan
  DiffPara GetDiffPara(unsigned int sl, bool rot=true) const;
  /// Returns a vector of displacements (in mm) from a field offset of 1Hz
  NEWMAT::ColumnVector GetHz2mmVector() const;
  /// Number of parameters for the EC model (including rigid body movement).
  unsigned int NParam(EDDY::Parameters whichp=EDDY::ALL) const EddyTry {
    if (whichp==EDDY::ZERO_ORDER_MOVEMENT) return(static_cast<unsigned int>(_mp.GetZeroOrderParams().Nrows()));
    if (whichp==EDDY::MOVEMENT) return(_mp.NParam());
    else if (whichp==EDDY::EC) return(_ecp->NParam());
    else return(_mp.NParam()+_ecp->NParam());
  } EddyCatch
  /// Returns all the current parameters for the EC model.
  NEWMAT::ColumnVector GetParams(EDDY::Parameters whichp=EDDY::ALL) const EddyTry {
    if (whichp==EDDY::ZERO_ORDER_MOVEMENT) return(_mp.GetZeroOrderParams());
    else if (whichp==EDDY::MOVEMENT) return(_mp.GetParams());
    else if (whichp==EDDY::EC) return(_ecp->GetParams());
    else return(_mp.GetParams() & _ecp->GetParams());
  } EddyCatch
  /// Number of parameters that are being optimised in the current ScanECModel.
  unsigned int NDerivs(EDDY::Parameters whichp=EDDY::ALL) const EddyTry {
    if (whichp==EDDY::MOVEMENT) return(_mp.NDerivs());
    else if (whichp==EDDY::EC) return(_ecp->NDerivs());
    else return(_mp.NDerivs() + _ecp->NDerivs());
  } EddyCatch
  /// Number of "compound" derivatives. See ECModels.h for an explanation of "compound"
  unsigned int NCompoundDerivs(EDDY::Parameters whichp=EDDY::ALL) const EddyTry {
    if (whichp==EDDY::MOVEMENT) return(_mp.NCompoundDerivs());
    else if (whichp==EDDY::EC) return(_ecp->NCompoundDerivs());
    else return(_mp.NCompoundDerivs() + _ecp->NCompoundDerivs());
  } EddyCatch
  /// Set lambda (weight) of regularisation cost for DCT movement
  void SetRegLambda(double lambda) { _mrl=lambda; }
  /// Return lambda (weight) of regularisation cost for DCT movement
  double GetRegLambda() const { return(_mrl); }
  /// Return value of regularisation cost for DCT movement
  double GetReg(EDDY::Parameters whichp=EDDY::ALL) const;
  /// Return gradient of regularisation cost for DCT movement
  NEWMAT::ColumnVector GetRegGrad(EDDY::Parameters whichp=EDDY::ALL) const;
  /// Return Hessian of regularisation cost for DCT movement
  NEWMAT::Matrix GetRegHess(EDDY::Parameters whichp=EDDY::ALL) const;
  /// Get the parameter (of those being optimised) given by indx
  double GetDerivParam(unsigned int indx, EDDY::Parameters whichp=EDDY::ALL, bool allow_field_offset=false) const;
  /// Get the scale for the parameter (of those being optimised) given by indx
  double GetDerivScale(unsigned int indx, EDDY::Parameters whichp=EDDY::ALL, bool allow_field_offset=false) const;
  /// Get indx'th set of instructions for "compound" derivatives. See ECModels.h for an explanation of "compound"
  EDDY::DerivativeInstructions GetCompoundDerivInstructions(unsigned int indx, EDDY::Parameters whichp=EDDY::ALL) const;
  /// Set/Update the parameter (of those being optimised) given by indx
  void SetDerivParam(unsigned int indx, double p, EDDY::Parameters whichp=EDDY::ALL, bool allow_field_offset=false);
  /// Returns matrix denoted \f$\mathbf{R}\f$ in paper. Returns const part in the case of SliceToVol.
  NEWMAT::Matrix ForwardMovementMatrix() const EddyTry { return(_mp.ForwardMovementMatrix(_ima)); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}\f$ in paper, but only for group of slices indicated by grp
  NEWMAT::Matrix ForwardMovementMatrix(unsigned int grp) const EddyTry { return(_mp.ForwardMovementMatrix(_ima,grp,_mbg.NGroups())); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}\f$ in paper, but excludes parameters indicated by rindx. Returns const part in the case of SliceToVol.
  NEWMAT::Matrix RestrictedForwardMovementMatrix(const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedForwardMovementMatrix(_ima,rindx)); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}\f$ in paper, but only for group of slices indicated by grp. Excludes parameters indicated by rindx.
  NEWMAT::Matrix RestrictedForwardMovementMatrix(unsigned int grp, const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedForwardMovementMatrix(_ima,grp,_mbg.NGroups(),rindx)); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}^{-1}\f$ in paper. Returns const part in the case of SliceToVol.
  NEWMAT::Matrix InverseMovementMatrix() const EddyTry { return(_mp.InverseMovementMatrix(_ima)); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}^{-1}\f$ in paper, but only for group of slices indicated by grp
  NEWMAT::Matrix InverseMovementMatrix(unsigned int grp) const EddyTry { return(_mp.InverseMovementMatrix(_ima,grp,_mbg.NGroups())); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}^{-1}\f$ in paper, but excludes parameters indicated by rindx. Returns const part in the case of SliceToVol.
  NEWMAT::Matrix RestrictedInverseMovementMatrix(const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedInverseMovementMatrix(_ima,rindx)); } EddyCatch
  /// Returns matrix denoted \f$\mathbf{R}^{-1}\f$ in paper, but only for group of slices indicated by grp. Excludes parameters indicated by rindx.
  NEWMAT::Matrix RestrictedInverseMovementMatrix(unsigned int grp, const std::vector<unsigned int>& rindx) const EddyTry { return(_mp.RestrictedInverseMovementMatrix(_ima,grp,_mbg.NGroups(),rindx)); } EddyCatch
  /// Returns the actual sampling points of all the points in the model space
  EDDY::ImageCoordinates SamplingPoints() const;
  /// Returns the EC-field relevant for the Observation->Model transform.
  NEWIMAGE::volume<float> ECField() const EddyTry { return(_ecp->ECField(_ima)); } EddyCatch
  /// Returns the total field (in mm) relevant for the Observation->Model transform
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(// Input
							 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							 // Output
							 NEWIMAGE::volume<float>&                          omask,
							 NEWIMAGE::volume<float>&                          jac) const EddyTry {
    return(field_for_scan_to_model_transform(susc,&omask,&jac));
  } EddyCatch
  /// Returns the total field (in mm) relevant for the Observation->Model transform
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(// Input
							 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							 // Output
							 NEWIMAGE::volume<float>&                          omask) const EddyTry {
    return(field_for_scan_to_model_transform(susc,&omask,NULL));
  } EddyCatch
  /// Returns the total field relevant for the Observation->Model transform
  NEWIMAGE::volume4D<float> FieldForScanToModelTransform(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(field_for_scan_to_model_transform(susc,NULL,NULL));
  } EddyCatch
  /// Returns the total field relevant for the Model->Observation transform
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(// Input
							 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							 // Output
							 NEWIMAGE::volume<float>&                        omask,
							 NEWIMAGE::volume<float>&                        jac) const EddyTry {
    return(field_for_model_to_scan_transform(susc,&omask,&jac));
  } EddyCatch
  /// Returns the total field relevant for the Model->Observation transform
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(// Input
							 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							 // Output
							 NEWIMAGE::volume<float>&                        omask) const EddyTry {
    return(field_for_model_to_scan_transform(susc,&omask,NULL));
  } EddyCatch
  /// Returns the total field relevant for the Model->Observation transform
  NEWIMAGE::volume4D<float> FieldForModelToScanTransform(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(field_for_model_to_scan_transform(susc,NULL,NULL));
  } EddyCatch
  /// Returns the total field relevant for the Model->Observation transform and the Jacobian associated with it.
  NEWIMAGE::volume4D<float> FieldForModelToScanTransformWithJac(// Input
								std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								// Output
								NEWIMAGE::volume<float>&                        jac) const EddyTry {
    return(field_for_model_to_scan_transform(susc,NULL,&jac));
  } EddyCatch
  /// Returns the total displacements (including subject movement) relevant for the Observation->Model transform.
  NEWIMAGE::volume4D<float> TotalDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc));
  } EddyCatch
  /// Returns the movement induced displacements relevant for the Observation->Model transform.
  NEWIMAGE::volume4D<float> MovementDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc,true));
  } EddyCatch
  /// Returns the movement induced displacements relevant for the Observation->Model transform, excluding any translation in PE-direction.
  NEWIMAGE::volume4D<float> RestrictedMovementDisplacementToModelSpace(std::shared_ptr<const NEWIMAGE::volume<float> > susc) const EddyTry {
    return(total_displacement_to_model_space(GetOriginalIma(),susc,true,true));
  } EddyCatch
  /// Set/update the parameters for the EC model.
  void SetParams(const NEWMAT::ColumnVector& mpep, EDDY::Parameters whichp=EDDY::ALL);
  /// Set the movement trace (movement over time) for slice-to-vol
  void SetS2VMovement(const NEWMAT::Matrix& mt);
  /// Replace selected slices AND recycle previous outliers. rep should be in model space.
  void SetAsOutliers(const NEWIMAGE::volume<float>&                     rep,
		     std::shared_ptr<const NEWIMAGE::volume<float> >    susc,
		     const NEWIMAGE::volume<float>&                     inmask,
		     const std::vector<unsigned int>&                   ol);
  /// Replace selected slices AND recycle previous outliers. rep should be in observation space.
  void SetAsOutliers(const NEWIMAGE::volume<float>&                     rep,
		     const NEWIMAGE::volume<float>&                     mask,
		     const std::vector<unsigned int>&                   ol);
  /// Bring back previously discarded slices into _ima
  void RecycleOutliers();
  /// Return a volume with _only_ those slices set as outliers being non-zero
  NEWIMAGE::volume<float> GetOutliers() const;

private:
  NEWIMAGE::volume<float>         _ima;   ///< The original volume, possibly with (outlier) replacements.
  std::vector<float*>             _ols;   ///< Discarded (for now) slices.
  AcqPara                         _acqp;  ///< The acquisition parameters associated with the volume.
  DiffPara                        _diffp; ///< The diffusion parameters associated with the volume.
  ScanMovementModel               _mp;    ///< The movement model associated with the volume.
  EDDY::MultiBandGroups           _mbg;   ///< Multi-band structure
  std::shared_ptr<ScanECModel>    _ecp;   ///< The EC model.
  EDDY::PolationPara              _pp;    ///< Inter/extrapolation parameters for volumetric and slice-to-vol
  double                          _mrl;   ///< Lambda for slice-to-vol movement regularisation

  NEWIMAGE::volume<float> motion_correct(// Input
					 const NEWIMAGE::volume<float>&  inima,
					 // Output (optional)
					 NEWIMAGE::volume<float>         *omask) const;
  NEWIMAGE::volume<float> transform_to_model_space(// Input
						   const NEWIMAGE::volume<float>&                    inima,
						   std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
						   // Output
						   NEWIMAGE::volume<float>&                          omask,
						   // Optional input
						   bool                                              jacmod=true) const;
  NEWIMAGE::volume<float> transform_slice_to_vol_to_model_space(// Input
								const NEWIMAGE::volume<float>&                  inima,
								std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								const NEWIMAGE::volume<float>                   *pred_ptr,
								// Output
								NEWIMAGE::volume<float>&                        omask,
								// Optional input
								bool                                            jacmod=true) const;
  void get_slice_stack_and_zcoords(// Input
				   const NEWIMAGE::volume<float>&                  inima,
				   std::shared_ptr<const NEWIMAGE::volume<float> > susc,
				   // Output
				   NEWIMAGE::volume<float>&                        slice_stack,
				   NEWIMAGE::volume<float>&                        z_coord,
				   NEWIMAGE::volume<float>&                        stack_mask,
				   // Optional input
				   bool                                            jacmod=true) const;
  NEWIMAGE::volume<float> transform_volumetric_to_model_space(// Input
							      const NEWIMAGE::volume<float>&                    inima,
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      // Output
							      NEWIMAGE::volume<float>&                          omask,
							      NEWIMAGE::volume4D<float>&                        deriv,
							      // Optional input
							      bool                                              jacmod=true) const;
  NEWIMAGE::volume4D<float> total_displacement_to_model_space(// Input
							      const NEWIMAGE::volume<float>&                    inima,
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      // Optional input
							      bool                                              movement_only=false,
							      bool                                              exclude_PE_tr=false) const;
  NEWIMAGE::volume4D<float> field_for_scan_to_model_transform(// Input
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      // Output
							      NEWIMAGE::volume<float>                           *omask,
							      NEWIMAGE::volume<float>                           *jac) const;
  void resample_and_combine_ec_and_susc_for_s2v(// Input
						std::shared_ptr<const NEWIMAGE::volume<float> > susc,
						// Output
						NEWIMAGE::volume<float>&                        tot,
						NEWIMAGE::volume<float>                         *omask) const;
  NEWIMAGE::volume4D<float> field_for_model_to_scan_transform(// Input
							      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							      // Output
							      NEWIMAGE::volume<float>                           *omask,
							      NEWIMAGE::volume<float>                           *jac) const;
  bool in_list(unsigned int s, const std::vector<unsigned int>& l) const EddyTry { for (unsigned int i=0; i<l.size(); i++) { if (l[i]==s) return(true); } return(false); } EddyCatch
};

/////////////////////////////////////////////////////////////////////
///
/// \brief This class manages a collection of scans (of type ECScan)
///  for the eddy project.
///
/////////////////////////////////////////////////////////////////////
class ECScanManager
{
public:
  /// Constructor for the ECScanManager class
  ECScanManager(const std::string&               imafname,
		const std::string&               maskfname,
		const std::string&               acqpfname,
		const std::string&               topupfname,
		const std::string&               fieldfname,
		const std::string&               field_mat_fname,
		const std::string&               bvecsfname,
		const std::string&               bvalsfname,
		EDDY::ECModel                    ecmodel,
		EDDY::ECModel                    b0_ecmodel,
		const std::vector<unsigned int>& indicies,
		const EDDY::PolationPara&        pp,
		EDDY::MultiBandGroups            mbg,
		bool                             fsh);
  ~ECScanManager() {}
  /// Number of scans of type st
  unsigned int NScans(ScanType st=ANY) const;
  /// Get multi-band structure. Assumes the same MBG for all scans
  const MultiBandGroups& MultiBand() const EddyTry { return(Scan(0).GetMBG()); } EddyCatch
  /// Set movement model order for all scans
  void SetMovementModelOrder(unsigned int order) EddyTry { for (unsigned int i=0; i<NScans(); i++) Scan(i).SetMovementModelOrder(order); } EddyCatch
  /// Get movement model order
  unsigned int GetMovementModelOrder() const EddyTry { return(Scan(0).GetMovementModelOrder()); } EddyCatch
  /// Returns true if the movement model has order greater than 0
  bool IsSliceToVol() const EddyTry { return(Scan(0).IsSliceToVol()); } EddyCatch
  /// Set lambda for regularisation of movement-over-time
  void Set_S2V_Lambda(double lambda) EddyTry { for (unsigned int i=0; i<NScans(); i++) Scan(i).SetRegLambda(lambda); } EddyCatch
  /// Returns true if the scans fall on shells (as opposed to DSI).
  bool IsShelled() const;
  /// Returns # of shells that data fall on. If st==ANY it will count b0 as an extra shell
  unsigned int NoOfShells(ScanType st=ANY) const;
  /// Returns a vector of diffusion parameters
  std::vector<DiffPara> GetDiffParas(ScanType st=ANY) const;
  /// Returns a vector of indicies (valid for st=ANY) for b0-scans
  std::vector<unsigned int> GetB0Indicies() const;
  /// Returns a vector of vectors of indicies (valid for st=ANY) into b!=0 shells.
  std::vector<std::vector<unsigned int> > GetShellIndicies(std::vector<double>& bvals) const;
  /// Number of LSR pairs of type st
  unsigned int NLSRPairs(ScanType st=ANY) const;
  /// Return true if any scan has a PE component in the x-direction
  bool HasPEinX() const EddyTry { return(has_pe_in_direction(1)); } EddyCatch
  /// Return true if any scan has a PE component in the y-direction
  bool HasPEinY() const EddyTry { return(has_pe_in_direction(2)); } EddyCatch
  /// Return true if any scan has a PE component in the x- AND any scan in the y-direction
  bool HasPEinXandY() const EddyTry { return(HasPEinX() && HasPEinY()); } EddyCatch
  /// Returns true if the scan indicated by indx (zero-offset) is diffusion weighted.
  bool IsDWI(unsigned int indx) const EddyTry { if (indx>_fi.size()-1) throw EddyException("ECScanManager::IsDWI: index out of range"); else return(!_fi[indx].first); } EddyCatch
  /// Returns true if the scan indicated by indx (zero-offset) is a b=0 scan.
  bool IsB0(unsigned int indx) const EddyTry { return(!IsDWI(indx)); } EddyCatch
  /// Returns scale factor that has been applied to data prior to registration process.
  double ScaleFactor() const { return(_sf); }
  /// Sets inter- and extrapolation options
  void SetPolation(const PolationPara& pp) EddyTry { _pp=pp; set_polation(pp); } EddyCatch
  /// Returns inter- and extrapolation options
  PolationPara GetPolation() const EddyTry { return(_pp); } EddyCatch
  /// Fills empty planes in the x-y directions
  void FillEmptyPlanes();
  /// Returns vector of Global (file) indicies for all dwi indicies.
  std::vector<unsigned int> GetDwi2GlobalIndexMapping() const;
  /// Returns Global (file) index for specified dwi index.
  unsigned int GetDwi2GlobalIndexMapping(unsigned int dwindx) const;
  /// Returns vector of Global (file) indicies for all b0 indicies.
  std::vector<unsigned int> Getb02GlobalIndexMapping() const;
  /// Returns Global (file) index for specified b0 index.
  unsigned int Getb02GlobalIndexMapping(unsigned int b0indx) const;
  /// Returns DWI index for specified global index
  unsigned int GetGlobal2DWIIndexMapping(unsigned int gindx) const;
  /// Returns b0 index for specified global index
  unsigned int GetGlobal2b0IndexMapping(unsigned int gindx) const;
  /// Returns current inter- and extrapolation methods
  PolationPara GetPolationMethods();
  /// Sets inter- and extrapolation methods
  void SetPolationMethods(const PolationPara& pp);
  /// Returns true if b0's have been spaced out in a way that can help with dwi movements.
  bool B0sAreInterspersed() const;
  /// Returns true if we have enough interspersed B0s to base the Post Eddy Align Shells on it
  bool B0sAreUsefulForPEAS() const;
  /// Use b0 movement estimates as starting guess for dwis.
  void PolateB0MovPar();
  /// Returns true if we want to use B0 movement estimates as starting guess for dwis.
  bool UseB0sToInformDWIRegistration() const { return(_use_b0_4_dwi); }
  /// Sets if we want to use B0 movement estimates as starting guess for dwis or not.
  void SetUseB0sToInformDWIRegistration(bool use_b0_4_dwi) { _use_b0_4_dwi = use_b0_4_dwi; }
  /// Returns true if a susceptibilty induced off-resonance field has been set
  bool HasSuscHzOffResField() const { return(_has_susc_field); }
  /// Returns true if a derivative susceptibility field has been set
  bool HasSuscHzOffResDerivField() const EddyTry { return(this->has_move_by_susc_fields()); } EddyCatch
  /// Returns true if a recieve bias field has been set
  bool HasBiasField() const { return(_bias_field.IsValid()); }
  /// Returns true if the EC model includes a field offset
  bool HasFieldOffset(ScanType st) const EddyTry {
    if (st==B0 || st==DWI) return(Scan(0,st).HasFieldOffset());
    else return(Scan(0,B0).HasFieldOffset() || Scan(0,DWI).HasFieldOffset());
  } EddyCatch
  /// Returns true if data allows for LSR resampling
  bool CanDoLSRResampling() const;
  /// Returns the individual indicies for scans that constitute the i'th LSR pair
  std::pair<unsigned int,unsigned int> GetLSRPair(unsigned int i, ScanType st) const;
  /// Sets parameters for all scans
  void SetParameters(const NEWMAT::Matrix& pM, ScanType st=ANY);
  /// Sets parameters for all scans from values in file given by fname
  void SetParameters(const std::string& fname, ScanType st=ANY) EddyTry { NEWMAT::Matrix pM = MISCMATHS::read_ascii_matrix(fname); SetParameters(pM,st); } EddyCatch
  /// Sets S2V movement trace for all scans
  void SetS2VMovement(const NEWMAT::Matrix& s2v_pM, ScanType st=ANY);
  /// Sets S2V movement trace for all scans from values in file given by fname
  void SetS2VMovement(const std::string& fname, ScanType st=ANY) EddyTry { NEWMAT::Matrix s2v_pM = MISCMATHS::read_ascii_matrix(fname); SetS2VMovement(s2v_pM,st); } EddyCatch
  /// Returns a read-only reference to a scan given by indx
  const ECScan& Scan(unsigned int indx, ScanType st=ANY) const;
  /// Returns a read-write reference to a scan given by indx
  ECScan& Scan(unsigned int indx, ScanType st=ANY);
  // Returns an "original" (no smoothing) image "unwarped" into model space
  NEWIMAGE::volume<float> GetUnwarpedOrigScan(unsigned int                    indx,
					      const NEWIMAGE::volume<float>&  pred,
					      NEWIMAGE::volume<float>&        omask,
					      ScanType                        st=ANY) const;
  NEWIMAGE::volume<float> GetUnwarpedOrigScan(unsigned int              indx,
					      NEWIMAGE::volume<float>&  omask,
					      ScanType                  st=ANY) const;
  NEWIMAGE::volume<float> GetUnwarpedOrigScan(unsigned int indx,
					      ScanType     st=ANY) const EddyTry {
    NEWIMAGE::volume<float> mask=_scans[0].GetIma(); mask=1.0;
    return(GetUnwarpedOrigScan(indx,mask,st));
  } EddyCatch
  // Returns an image "unwarped" into model space
  NEWIMAGE::volume<float> GetUnwarpedScan(unsigned int                    indx,
					  const NEWIMAGE::volume<float>&  pred,
					  NEWIMAGE::volume<float>&        omask,
					  ScanType                        st=ANY) const;
  NEWIMAGE::volume<float> GetUnwarpedScan(unsigned int              indx,
					  NEWIMAGE::volume<float>&  omask,
					  ScanType                  st=ANY) const;
  NEWIMAGE::volume<float> GetUnwarpedScan(unsigned int indx,
					  ScanType     st=ANY) const EddyTry {
    NEWIMAGE::volume<float> mask=_scans[0].GetIma(); mask=1.0;
    return(GetUnwarpedScan(indx,mask,st));
  } EddyCatch
  /// Resamples a matching pair of images using least-squares resampling.
  NEWIMAGE::volume<float> LSRResamplePair(// Input
					  unsigned int              i,
					  unsigned int              j,
					  ScanType                  st,
					  // Output
					  NEWIMAGE::volume<float>&  omask) const;
  /// Sets movement and EC parameters for a scan given by indx
  void SetScanParameters(unsigned int indx, const NEWMAT::ColumnVector& p, ScanType st=ANY) EddyTry { Scan(indx,st).SetParams(p,ALL); } EddyCatch
  /// Add rotation to all scans (for testing rotation of b-vecs)
  void AddRotation(const std::vector<float>& rot);
  /// Returns the user defined mask (that was passed to the constructor).
  const NEWIMAGE::volume<float>& Mask() const EddyTry { return(_mask); } EddyCatch
  /// Returns a list of slices with more than nvox brain voxels (as assessed by _mask)
  std::vector<unsigned int> IntraCerebralSlices(unsigned int nvox) const;
  /// Returns a pointer to susceptibility induced off-resonance field (in Hz). Returns NULL when no field set.
  std::shared_ptr<const NEWIMAGE::volume<float> > GetSuscHzOffResField() const EddyTry { return(_susc_field); } EddyCatch
  /// Returns a susceptibility induced off-resonance field for scan indx. Will be different from above only when modelling susc-by-movement.
  std::shared_ptr<const NEWIMAGE::volume<float> > GetSuscHzOffResField(unsigned int indx, ScanType st=ANY) const;
  /// Returns a multiplicative bias field. Returns nullptr when no field set
  std::shared_ptr<const NEWIMAGE::volume<float> > GetBiasField() const EddyTry { return(_bias_field.GetField()); } EddyCatch
  /// Gets value of current offset used for bias-field
  float GetBiasFieldOffset() const EddyTry { return(_bias_field.GetOffset()); }	EddyCatch
  /// Set partial derivate field dfield w.r.t. movement parameter pi
  void SetDerivSuscField(unsigned int pi, const NEWIMAGE::volume<float>& dfield);
  /// Set second partial derivate field dfield w.r.t. movement parameters pi and pj
  void Set2ndDerivSuscField(unsigned int pi, unsigned int pj, const NEWIMAGE::volume<float>& dfield);
  /// Set bias field
  void SetBiasField(const NEWIMAGE::volume<float>& bfield) EddyTry { _bias_field.SetField(bfield); } EddyCatch
  /// Set offset for bias field. Will not allow an offset that yields negative field values.
  bool SetBiasFieldOffset(float offset) EddyTry { return(_bias_field.SetOffset(offset)); } EddyCatch
  /// Set a scalefactor for bias field by prescribing a mean
  void SetBiasFieldMean(float mean) EddyTry { _bias_field.SetMean(mean); } EddyCatch
  /// Set a scalefactor for bias field by prescribing a mean within a mask
  void SetBiasFieldMean(float mean, const NEWIMAGE::volume<float>& mask) EddyTry { _bias_field.SetMean(mean,mask); } EddyCatch
  /// Reset bias field, i.e. implicitly set it to one everywhere
  void ResetBiasField() EddyTry { _bias_field.Reset(); } EddyCatch
  /// Returns off-resonance field pertaining to EC only. Movements are not included and its main use is for visualiation/demonstration.
  NEWIMAGE::volume<float> GetScanHzECOffResField(unsigned int indx, ScanType st=ANY) const EddyTry { return(Scan(indx,st).ECField()); } EddyCatch
  /// Separate field offset (image FOV centre not coinciding with scanner iso-centre) from actual subject movement in PE direction.
  void SeparateFieldOffsetFromMovement(ScanType      st,
				       OffsetModel   m=LinearOffset);
  /// Recycle (bring back) all slices labeled as outliers
  void RecycleOutliers() EddyTry { for (unsigned int s=0; s<NScans(); s++) Scan(s).RecycleOutliers(); } EddyCatch
  /// Model parameters as a function of diffusion gradients
  void SetPredictedECParam(ScanType st, SecondLevelECModel slm);
  /// Set specified scan as overall reference for location.
  void SetLocationReference(unsigned int ref=0) EddyTry { _refs.SetLocationReference(ref); } EddyCatch
  /// Set specified scan as overall (of dwi scans) reference for location.
  void SetDWILocationReference(unsigned int ref) EddyTry { _refs.SetDWILocationReference(ref); } EddyCatch
  /// Set specified scan as b0 reference for location.
  void SetB0LocationReference(unsigned int ref) EddyTry { _refs.SetB0LocationReference(ref); } EddyCatch
  /// Set specified scan as shell location reference
  void SetShellLocationReference(unsigned int si, unsigned int ref) EddyTry { _refs.SetShellLocationReference(si,ref); } EddyCatch
  /// Set specified scan as b0 shape reference
  void SetB0ShapeReference(unsigned int ref) EddyTry { _refs.SetB0ShapeReference(ref); } EddyCatch
  /// Set specified scan as shell shape reference
  void SetShellShapeReference(unsigned int si, unsigned int ref) EddyTry { _refs.SetShellShapeReference(si,ref); } EddyCatch
  /// Apply b0 shape reference
  void ApplyB0ShapeReference() EddyTry { set_slice_to_vol_reference(_refs.GetB0ShapeReference(),B0); } EddyCatch
  /// Apply b0 location reference
  void ApplyB0LocationReference() EddyTry { set_reference(GetGlobal2b0IndexMapping(_refs.GetB0LocationReference()),B0); } EddyCatch
  /// Apply shell shape reference
  void ApplyShellShapeReference(unsigned int si) EddyTry { set_slice_to_vol_reference(_refs.GetShellShapeReference(si),DWI,si); } EddyCatch
  /// Apply dwi location reference
  void ApplyDWILocationReference() EddyTry { set_reference(GetGlobal2DWIIndexMapping(_refs.GetDWILocationReference()),DWI); } EddyCatch
  /// Apply overall location reference
  void ApplyLocationReference() EddyTry { set_reference(_refs.GetLocationReference(),ANY); } EddyCatch

  /// Writes distortion corrected images to disk.
  void WriteRegisteredImages(const std::string& fname, const std::string& maskfname, FinalResampling resmethod, double LSR_lambda, bool mask_output, ScanType st=ANY) EddyTry
  {
    if (resmethod==EDDY::LSR && !mask_output) throw EddyException("ECScanManager::WriteRegisteredImages: Must mask images when resampling method is LSR");
    PolationPara old_pp = this->GetPolation();
    PolationPara pp(NEWIMAGE::spline,NEWIMAGE::periodic,true,NEWIMAGE::spline);
    this->SetPolation(pp);
    if (resmethod==EDDY::JAC) write_jac_registered_images(fname,maskfname,mask_output,st);
    else if (resmethod==EDDY::LSR) {
      write_lsr_registered_images(fname,LSR_lambda,st);
    }
    else throw EddyException("ECScanManager::WriteRegisteredImages: Unknown resampling method");
    this->SetPolation(old_pp);
  } EddyCatch
  /// Writes distortion corrected images to disk, using predictions to fill in slice-to-vol gaps
  void WriteRegisteredImages(const std::string& fname, const std::string& maskfname, FinalResampling resmethod, double LSR_lambda, bool mask_output, const NEWIMAGE::volume4D<float>& pred, ScanType st=ANY) EddyTry
  {
    if (resmethod==EDDY::LSR && !mask_output) throw EddyException("ECScanManager::WriteRegisteredImages: Must mask images when resampling method is LSR");
    if (pred.tsize() != int(NScans(st))) throw EddyException("ECScanManager::WriteRegisteredImages: Size mismatch between pred and NScans");
    PolationPara old_pp = this->GetPolation();
    PolationPara pp(NEWIMAGE::spline,NEWIMAGE::periodic,true,NEWIMAGE::spline);
    this->SetPolation(pp);
    if (resmethod==EDDY::JAC) write_jac_registered_images(fname,maskfname,mask_output,pred,st);
    else if (resmethod==EDDY::LSR) {
      write_lsr_registered_images(fname,LSR_lambda,st);
    }
    else throw EddyException("ECScanManager::WriteRegisteredImages: Unknown resampling method");
    this->SetPolation(old_pp);
  } EddyCatch
  /// Writes file with movement and EC parameters.
  void WriteParameterFile(const std::string& fname, ScanType st=ANY) const;
  /// Writes file with one set of movement parameters per time-point (slice or MB-group)
  void WriteMovementOverTimeFile(const std::string& fname, ScanType st=ANY) const;
  /// Writes eddy-current induced fields to disc
  void WriteECFields(const std::string& fname, ScanType st=ANY) const;
  /// Writes rotated b-vecs (ascii) to disc
  void WriteRotatedBVecs(const std::string& fname, ScanType st=ANY) const;
  /// Writes an ascii file with movement induced RMS
  void WriteMovementRMS(const std::string& fname, ScanType st=ANY) const;
  /// Writes an ascii file with movement induced RMS, but excluding any translation in the PE-direction
  void WriteRestrictedMovementRMS(const std::string& fname, ScanType st=ANY) const;
  /// Writes 3D displacement fields (in mm).
  void WriteDisplacementFields(const std::string& basefname, ScanType st=ANY) const;
  /// Writes original data with outliers replaced by predictions
  void WriteOutlierFreeData(const std::string& fname, ScanType st=ANY) const;
  /// Writes out data currently labeled as outliers. For debugging only.
  void WriteOutliers(const std::string& fname, ScanType st=ANY) const;
  /// Nested class used for the bias-field
  class BiasField {
  public:
    BiasField() : _rawfield(nullptr), _field(nullptr), _offset(0.0), _scale(1.0) {}
    bool IsValid() const { return(_field != nullptr); }
    void SetField(const NEWIMAGE::volume<float>& field) {
      _rawfield = std::make_shared<NEWIMAGE::volume<float> >(field);
      _field = std::make_shared<NEWIMAGE::volume<float> >(field);
      _offset = 0.1 - _field->min(); // Ensure smallest value is 0.1
      (*_field) += _offset;
    }
    bool SetOffset(float offs) {
      if (!this->IsValid()) throw EddyException("ScanManager::BiasField::SetOffset: Invalid bias-field");
      if ((offs + _rawfield->min()) < 1e-6) return(false);
      else {
	_offset = offs;
	_scale = 1.0;
        (*_field) = _scale * ((*_rawfield) + offs);
        return(true);
      }
    }
    float GetOffset() const { return(_offset); }
    void SetMean(float mean) {
      if (!this->IsValid()) throw EddyException("ScanManager::BiasField::SetMean: Invalid bias-field");
      _scale = (mean / _field->mean());
      (*_field) *= _scale;
    }
    void SetMean(float mean, const NEWIMAGE::volume<float>& mask) {
      if (!this->IsValid()) throw EddyException("ScanManager::BiasField::SetMean: Invalid bias-field");
      if (!samesize((*_rawfield),mask)) throw EddyException("ScanManager::BiasField::SetMean: Size-mismatch between field and mask");
      _scale = (mean / _field->mean(mask));
      (*_field) *= _scale;
    }
    std::shared_ptr<const NEWIMAGE::volume<float> > GetField() const { return(_field); }
    void Reset() { if (_rawfield != nullptr) { _rawfield.reset(); _field.reset(); _offset=0.0; _scale=1.0;} }
  private:
    std::shared_ptr<NEWIMAGE::volume<float> >  _rawfield;
    std::shared_ptr<NEWIMAGE::volume<float> >  _field;
    float                                      _offset;
    float                                      _scale;
  }; // End of nested class BiasField
private:
  bool                                                                  _has_susc_field; ///< Is true if object contains a valid susceptibility field
  std::shared_ptr<NEWIMAGE::volume<float> >                             _susc_field;     ///< Susceptibility field (in Hz).
  std::vector<std::shared_ptr<NEWIMAGE::volume<float> > >               _susc_d1;        ///< First derivative susc fields in order xt yt zt xr yr zr
  std::vector<std::vector<std::shared_ptr<NEWIMAGE::volume<float> > > > _susc_d2;        ///< Second derivative susc fields. Organised as sub-diagonal matrix
  BiasField                                                             _bias_field;     ///< Multiplicative recieve bias field
  NEWIMAGE::volume<float>                                               _mask;           ///< User supplied mask
  double                                                                _sf;             ///< Scale factor applied to scans
  std::vector<std::pair<int,int> >                                      _fi;             ///< Used to keep track of index into file
  std::vector<ECScan>                                                   _scans;          ///< Vector of diffusion weighted scans
  std::vector<ECScan>                                                   _b0scans;        ///< Vector of b=0 scans.
  ReferenceScans                                                        _refs;           ///< Has info on location and shape references
  PolationPara                                                          _pp;             ///< Inter- and extrapolation settings
  bool                                                                  _fsh;            ///< If true, user guarantees shelling
  bool                                                                  _use_b0_4_dwi;   ///< Decides if b0s are to inform dwi registration

  /// Extracts everything that can potentially be a field offset scaled in Hz.
  NEWMAT::ColumnVector hz_vector_with_everything(ScanType st) const;
  /// Creates a design matrix modelling parameters as a linear function of diffusion gradients
  NEWMAT::Matrix linear_design_matrix(ScanType st) const;
  /// Creates a design matrix modelling parameters as a quadratic function of diffusion gradients
  NEWMAT::Matrix quadratic_design_matrix(ScanType st) const;
  /// Demeans (columnwise) a matrix. Can be replaced by MISCMATHS::detrend if bug fixed in that
  NEWMAT::Matrix demean_matrix(const NEWMAT::Matrix& X) const;
  /// Returns a vector of movement parameters for the b=0 scans inter/extrapolated onto the dwis.
  NEWMAT::Matrix get_b0_movement_vector(ScanType st=DWI) const;
  /// Sets scan indicated by ref as reference (position zero) for all scans of type st.
  void set_reference(unsigned int ref, ScanType st);
  /// Sets ref scan (as above) for shell si. Additionally also sets reference shape by ensuring ref has no higher order mp.
  void set_slice_to_vol_reference(unsigned int ref, ScanType st, int si=-1);
  double mean_of_first_b0(const NEWIMAGE::volume4D<float>&   vols,
                          const NEWIMAGE::volume<float>&     mask,
                          const NEWMAT::Matrix&              bvecs,
                          const NEWMAT::Matrix&              bvals) const;
  bool index_kosher(unsigned int indx, ScanType st) const EddyTry
  {
    if (st==DWI) return(indx<_scans.size());
    else if (st==B0) return(indx<_b0scans.size());
    else return(indx<_fi.size());
  } EddyCatch
  /// Sets inter/extrapolation parameters for all scans.
  void set_polation(const PolationPara& pp) EddyTry {
    for (unsigned int i=0; i<_scans.size(); i++) _scans[i].SetPolation(pp);
    for (unsigned int i=0; i<_b0scans.size(); i++) _b0scans[i].SetPolation(pp);
  } EddyCatch
  /// Writes registered images using Jacobian modulation. Has GPU implementation
  void write_jac_registered_images(const std::string& fname, const std::string& maskfname, bool mask_output, ScanType st) const;
  void write_jac_registered_images(const std::string& fname, const std::string& maskfname, bool mask_output, const NEWIMAGE::volume4D<float>& pred, ScanType st) const;
  void write_lsr_registered_images(const std::string& fname, double lambda, ScanType st) const;
  bool has_pe_in_direction(unsigned int dir, ScanType st=ANY) const;
  /// Brackets an index. A value of -1 of either element indicates end index.
  std::pair<int,int> bracket(unsigned int i, const std::vector<unsigned int>& ii) const;
  /// Will inter/extrapolate movement parameters.
  NEWMAT::ColumnVector interp_movpar(unsigned int i, const std::pair<int,int>& br) const;
  /// Reads ascii file with rigid body (RB) matrix, and makes sure it is RB
  NEWMAT::Matrix read_rb_matrix(const std::string& fname) const;
  bool indicies_clustered(const std::vector<unsigned int>& indicies,
			  unsigned int                     N) const;
  /// Checks if there are any derivative fields
  bool has_move_by_susc_fields() const;
};

} // End namespace EDDY

#endif // End #ifndef ECScanClasses_h

////////////////////////////////////////////////
//
// Here starts Doxygen documentation
//
////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
///
/// \var ECScan::_ecp
/// This is an object determining the model that is being used to model the eddy currents.
/// It is stored as a (safe) pointer to a virtual base class ScanECModel.
/// In any instantiation of an ECScan object it will contain an object of a class derived
/// from ScanECModel, for example LinearScanECModel. This means that ECScan do not need to
/// know which model is actually being used since it will only interact with _ecp through
/// the interface specified by ScanECModel.
///
/// \var ECScanManager::_fi
/// This is an array of pairs keeping track of the relationship between on the one hand
/// DWI and b0 indexing and on the other hand global indexing (corresponding to the indexing
/// on disk). So if we for example have a diffusion weighted volume that is the i'th volume
/// in the 4D file it was read from and that also is the j'th diffusion weighted volume then:
/// _fi[i-1].first==0 (to indicate that it is diffusion weighted) and _fi[i-1].second==j-1 to
/// indicate it is the j'th diffusion weighted volume. Correspondingly a b=0 volume that is
/// the l'th volume on disc and the m'th b=0 volume then: _fi[l-1].first==1 and
/// _fi[l-1].second==m-1.
///
/////////////////////////////////////////////////////////////////////
