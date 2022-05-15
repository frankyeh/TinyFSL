// Declarations of classes that implements useful
// utility functions for the eddy current project.
// They are collections of statically declared
// functions that have been collected into classes
// to make it explicit where they come from. There
// will never be any instances of theses classes.
//
// EddyUtils.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#ifndef EddyUtils_h
#define EddyUtils_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "EddyHelperClasses.h"
#include "ECScanClasses.h"

namespace EDDY {

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class EddyUtils
//
// Helper Class used to perform various useful tasks for
// the eddy current correction project.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class EddyUtils
{
private:
  /// b-values within this range considered equivalent
  static const int b_range = 100;

  /// bladibla
  static NEWIMAGE::volume4D<float> get_partial_derivatives_in_scan_space(// Input
									 const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
									 const EDDY::ECScan&                               scan,      // Scan space
									 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
                                                                         EDDY::Parameters                                  whichp);

  static NEWIMAGE::volume4D<float> get_direct_partial_derivatives_in_scan_space(// Input
										const NEWIMAGE::volume<float>&                    pred,     // Prediction in model space
										const EDDY::ECScan&                               scan,     // Scan space
										std::shared_ptr<const NEWIMAGE::volume<float> >   susc,     // Susceptibility off-resonance field
										EDDY::Parameters                                  whichp);

  static double param_update(// Input
			     const NEWIMAGE::volume<float>&                      pred,      // Prediction in model space
			     std::shared_ptr<const NEWIMAGE::volume<float> >     susc,      // Susceptibility off-resonance field
			     std::shared_ptr<const NEWIMAGE::volume<float> >     bias,      // Recieve bias field
			     const NEWIMAGE::volume<float>&                      pmask,     // "Data valid" mask in model space
			     Parameters                                          whichp,    // Which parameters to update
			     bool                                                cbs,       // Check (success of parameters) Before Set
			     float                                               fwhm,      // FWHM for Gaussian smoothing */
			     // These input parameters are for debugging only
			     unsigned int                                        scindx,    // Scan index
			     unsigned int                                        iter,      // Iteration
			     unsigned int                                        level,     // Determines how much gets written
			     // Input/output
			     EDDY::ECScan&                                       scan);     // Scan we want to register to pred

  static void write_debug_info_for_param_update(const EDDY::ECScan&                                scan,
						unsigned int                                       scindx,
						unsigned int                                       iter,
						unsigned int                                       level,
						bool                                               cbs,
						float                                              fwhm,
						const NEWIMAGE::volume4D<float>&                   derivs,
						const NEWIMAGE::volume<float>&                     mask,
						const NEWIMAGE::volume<float>&                     mios,
						const NEWIMAGE::volume<float>&                     pios,
						const NEWIMAGE::volume<float>&                     jac,
						std::shared_ptr<const NEWIMAGE::volume<float> >    susc,
						std::shared_ptr<const NEWIMAGE::volume<float> >    bias,
						const NEWIMAGE::volume<float>&                     pred,
						const NEWIMAGE::volume<float>&                     dima,
						const NEWIMAGE::volume<float>&                     sims,
						const NEWIMAGE::volume<float>&                     pmask,
						const NEWMAT::Matrix&                              XtX,
						const NEWMAT::ColumnVector&                        Xty,
						const NEWMAT::ColumnVector&                        update);

  static EDDY::ImageCoordinates transform_coordinates_from_model_to_scan_space(// Input
									       const NEWIMAGE::volume<float>&                    pred,
									       const EDDY::ECScan&                               scan,
									       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
									       // Output
									       NEWIMAGE::volume<float>                           *omask,
									       NEWIMAGE::volume<float>                           *jac);
  // Returns coordinates into f transformed with
  // displacement field d and affine matrix M
  static void transform_coordinates(// Input
				    const NEWIMAGE::volume<float>&    f,
				    const NEWIMAGE::volume4D<float>&  d,
				    const NEWMAT::Matrix&             M,
				    std::vector<unsigned int>         slices,
				    // Output
				    ImageCoordinates&                 c,
				    NEWIMAGE::volume<float>           *omask);

  // Calculates X.t()*X where X is a matrix where each column is one of the volumes in vols
  static NEWMAT::Matrix make_XtX(const NEWIMAGE::volume4D<float>& vols,
				 const NEWIMAGE::volume<float>&   mask);

  // Calculates X.t()*y where X is a matrix where each column is one of the volumes in Xvols
  // and where y is the volume in Yvol.
  static NEWMAT::ColumnVector make_Xty(const NEWIMAGE::volume4D<float>& Xvols,
				       const NEWIMAGE::volume<float>&   Yvol,
				       const NEWIMAGE::volume<float>&   mask);
  static bool get_groups(// Input
			 const std::vector<DiffPara>&             dpv,
			 // Output
			 std::vector<std::vector<unsigned int> >& grps,
			 std::vector<unsigned int>&               grpi,
			 std::vector<double>&                     grpb);

public:
  // This function has been temporarily moved into public space. Should probably be
  // moved back to private space at some stage.
  static NEWIMAGE::volume<float> transform_model_to_scan_space(// Input
							       const NEWIMAGE::volume<float>&                    pred,
							       const EDDY::ECScan&                               scan,
							       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							       bool                                              jacmod,
							       // Output
							       NEWIMAGE::volume<float>&                          omask,
							       NEWIMAGE::volume<float>                           *jac,
							       NEWIMAGE::volume4D<float>                         *grad);

  /*
  static NEWIMAGE::volume<float> transform_model_to_scan_space(// Input
							       const NEWIMAGE::volume<float>&                    pred,
							       const EDDY::ECScan&                               scan,
							       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
							       bool                                              jacmod,
							       // Output
							       NEWIMAGE::volume<float>&                          omask,
							       NEWIMAGE::volume<float>                           *jac,
							       NEWIMAGE::volume4D<float>                         *grad,
							       // Tmp
							       unsigned int                                      scindx,
							       unsigned int                                      iter,
							       unsigned int                                      level);
  */

  // Some functions for comparing diffusion parameters
  /// Returns true if the difference in b-value is less than EddyUtils::b_range
  static bool AreInSameShell(const DiffPara& dp1,
                             const DiffPara& dp2) EddyTry { return(fabs(dp1.bVal()-dp2.bVal())<double(b_range)); } EddyCatch
  static bool IsDiffusionWeighted(const DiffPara& dp) EddyTry { return(dp.bVal() > double(b_range)); } EddyCatch
  static bool Isb0(const DiffPara& dp) EddyTry { return(!IsDiffusionWeighted(dp)); } EddyCatch
  /// Returns true if the inner product of the b-vectors is greater than 0.999
  static bool HaveSameDirection(const DiffPara& dp1,
				const DiffPara& dp2) EddyTry { return(NEWMAT::DotProduct(dp1.bVec(),dp2.bVec())>0.999); } EddyCatch
  /// Returns true if a vector of DiffPara objects implies a "shelled" (i.e. non-DSI) design
  static bool IsShelled(const std::vector<DiffPara>& dpv);
  /// Returns true if a vector of DiffPara objects implies a multi-shell design
  static bool IsMultiShell(const std::vector<DiffPara>& dpv);
  /// Returns a vector of group indices (one for each element in dpv) and b-values for the different groups
  static bool GetGroups(// Input
			const std::vector<DiffPara>&             dpv,
			// Output
			std::vector<unsigned int>&               grpi,
			std::vector<double>&                     grpb);
  /// Returns n vectors of indicies into dpv, where n is the number of groups. Also b-values for each group.
  static bool GetGroups(// Input
			const std::vector<DiffPara>&             dpv,
			// Output
			std::vector<std::vector<unsigned int> >& grps,
			std::vector<double>&                     grpb);
  /// Returns group info in both the formats of the routines above.
  static bool GetGroups(// Input
			const std::vector<DiffPara>&             dpv,
			// Output
			std::vector<std::vector<unsigned int> >& grps,
			std::vector<unsigned int>&               grpi,
			std::vector<double>&                     grpb);
  // Random functions to set extrapolation and interpolation //
  template <class V>
  static void SetTrilinearInterp(V& vol) EddyTry {
    if (vol.getinterpolationmethod() != NEWIMAGE::trilinear) vol.setinterpolationmethod(NEWIMAGE::trilinear);
    if (vol.getextrapolationmethod() != NEWIMAGE::mirror) vol.setextrapolationmethod(NEWIMAGE::mirror);
  } EddyCatch
  template <class V>
  static void SetSplineInterp(V& vol) EddyTry {
    if (vol.getinterpolationmethod() != NEWIMAGE::spline) vol.setinterpolationmethod(NEWIMAGE::spline);
    if (vol.getsplineorder() != 3) vol.setsplineorder(3);
    if (vol.getextrapolationmethod() != NEWIMAGE::mirror) vol.setextrapolationmethod(NEWIMAGE::mirror);
  } EddyCatch

  // Check if a pair of ECScans can potentially be used in an LSR reconstruction
  static bool AreMatchingPair(const ECScan& s1, const ECScan& s2);

  // Get indicies for non-zero b-values
  static std::vector<unsigned int> GetIndiciesOfDWIs(const std::vector<DiffPara>& dpars);

  // Get vector of forward movement matrices, one per slice.
  static std::vector<NEWMAT::Matrix> GetSliceWiseForwardMovementMatrices(const EDDY::ECScan& scan);

  // Get vector of inverse movement matrices, one per slice.
  static std::vector<NEWMAT::Matrix> GetSliceWiseInverseMovementMatrices(const EDDY::ECScan& scan);

  // Removes bvecs associated with zero b-values.
  static std::vector<DiffPara> GetDWIDiffParas(const std::vector<DiffPara>&   dpars);

  // Reads all diffusion weighted images from 4D volume
  static int read_DWI_volume4D(NEWIMAGE::volume4D<float>&     dwivols,
			       const std::string&             fname,
			       const std::vector<DiffPara>&   dpars);

  // Converts char mask (from the general_transform functions) to a float mask
  static NEWIMAGE::volume<float> ConvertMaskToFloat(const NEWIMAGE::volume<char>& charmask);

  // 3D Smooth 3D/4D volume within mask
  static NEWIMAGE::volume<float> Smooth(const NEWIMAGE::volume<float>& ima,   // Image to smooth
					float                          fwhm,  // FWHM of Gaussian kernel
					const NEWIMAGE::volume<float>& mask); // Mask within which to smooth

  /// Checks for reasonable values in update. Based on empirical assessments.
  static bool UpdateMakesSense(const EDDY::ECScan&           scan,
			       const NEWMAT::ColumnVector&   update);

  // Make image with normal distributed noise
  static NEWIMAGE::volume<float> MakeNoiseIma(const NEWIMAGE::volume<float>&   ima,     // Template image
					      float                            mu,      // Mean of noise
					      float                            stdev);  // Stdev of noise

  // Calculates slice-wise statistics from the difference between observation and predicton in observation space
  static DiffStats GetSliceWiseStats(// Input
				     const NEWIMAGE::volume<float>&                  pred,      // Prediction in model space
				     std::shared_ptr<const NEWIMAGE::volume<float> > susc,      // Susceptibility induced off-resonance field
				     const NEWIMAGE::volume<float>&                  pmask,     // "Data valid" mask in model space
				     const NEWIMAGE::volume<float>&                  bmask,     // Brain mask in model space
				     const EDDY::ECScan&                             scan);     // Scan corresponding to pred

  // Performs an update of the movement parameters for one scan
  static double MovParamUpdate(// Input
			       const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
			       std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
			       std::shared_ptr<const NEWIMAGE::volume<float> >   bias,      // Recieve bias field
			       const NEWIMAGE::volume<float>&                    pmask,     // "Data valid" mask in model space
			       bool                                              cbs,       // Which parameters to update
                               float                                             fwhm,      // FWHM for Gaussian smoothing
			       // Input/output
			       EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,bias,pmask,MOVEMENT,cbs,fwhm,0,0,0,scan));
  } EddyCatch

  // Performs an update of the EC parameters for one scan
  static double ECParamUpdate(// Input
			      const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
			      std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
			      std::shared_ptr<const NEWIMAGE::volume<float> >   bias,      // Recieve bias field
			      const NEWIMAGE::volume<float>&                    pmask,     // "Data valid" mask in model space
			      bool                                              cbs,       // Which parameters to update
                              float                                             fwhm,      // FWHM for Gaussian smoothing
			      // Input/output
			      EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,bias,pmask,EC,cbs,fwhm,0,0,0,scan));
  } EddyCatch

  // Performs an update of the EC parameters for one scan
  // Does currently not use the bias parameter
  static double MovAndECParamUpdate(// Input
				    const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
				    std::shared_ptr<const NEWIMAGE::volume<float> >   bias,      // Recieve bias field
				    const NEWIMAGE::volume<float>&                    pmask,     // "Data valid" mask in model space
				    bool                                              cbs,       // Which parameters to update
				    float                                             fwhm,      // FWHM for Gaussian smoothing
				    // Input/output
				    EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,bias,pmask,ALL,cbs,fwhm,0,0,0,scan));
  } EddyCatch

  // Performs an update of the EC parameters and writes debug into for one scan
  // Does currently not use the bias parameter
  static double MovAndECParamUpdate(// Input
				    const NEWIMAGE::volume<float>&                    pred,      // Prediction in model space
				    std::shared_ptr<const NEWIMAGE::volume<float> >   susc,      // Susceptibility off-resonance field
				    std::shared_ptr<const NEWIMAGE::volume<float> >   bias,      // Recieve bias field
				    const NEWIMAGE::volume<float>&                    pmask,     // "Data valid" mask in model space
				    bool                                              cbs,       // Which parameters to update
				    float                                             fwhm,      // FWHM for Gaussian smoothing
				    // Parameters for debugging
				    unsigned int                                      scindx,    // Scan index
				    unsigned int                                      iter,      // Iteration
				    unsigned int                                      level,     // Determines how much gets written
				    // Input/output
				    EDDY::ECScan&                                     scan) EddyTry {
    return(param_update(pred,susc,bias,pmask,ALL,cbs,fwhm,scindx,iter,level,scan));
  } EddyCatch

  // Transforms an image from model/prediction space to observation space
  static NEWIMAGE::volume<float> TransformModelToScanSpace(// Input
							   const NEWIMAGE::volume<float>&                    pred,
							   const EDDY::ECScan&                               scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry {
    NEWIMAGE::volume<float> mask(pred.xsize(),pred.ysize(),pred.zsize());
    NEWIMAGE::volume<float> jac(pred.xsize(),pred.ysize(),pred.zsize());
    return(transform_model_to_scan_space(pred,scan,susc,true,mask,&jac,NULL));
  } EddyCatch
  static NEWIMAGE::volume<float> TransformScanToModelSpace(// Input
							   const EDDY::ECScan&                             scan,
							   std::shared_ptr<const NEWIMAGE::volume<float> > susc,
							   // Output
							   NEWIMAGE::volume<float>&                        omask);

  // The next two are alternate transformation routines that
  // performs the transforms in several resampling steps.
  // They are intended for debugging.
  static NEWIMAGE::volume<float> DirectTransformScanToModelSpace(// Input
								 const EDDY::ECScan&                             scan,
								 std::shared_ptr<const NEWIMAGE::volume<float> > susc,
								 // Output
								 NEWIMAGE::volume<float>&                        omask);
  /*
  static NEWIMAGE::volume<float> DirectTransformModelToScanSpace(// Input
								 const NEWIMAGE::volume<float>&                    ima,
								 const EDDY::ECScan&                               scan,
								 const EDDY::MultiBandGroups&                      mbg,
								 std::shared_ptr<const NEWIMAGE::volume<float> >   susc,
								 // Output
								 NEWIMAGE::volume<float>&                          omask);
  */
  static NEWIMAGE::volume4D<float> DerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                             scan,
									   const NEWIMAGE::volume<float>&                  mima,
									   std::shared_ptr<const NEWIMAGE::volume<float> > susc)
  {
    return(EddyUtils::get_partial_derivatives_in_scan_space(mima,scan,susc,EDDY::ALL));
  }

  static NEWIMAGE::volume4D<float> DirectDerivativesForModelToScanSpaceTransform(const EDDY::ECScan&                               scan,
										 const NEWIMAGE::volume<float>&                    mima,
										 std::shared_ptr<const NEWIMAGE::volume<float> >   susc) EddyTry
  {
    return(EddyUtils::get_direct_partial_derivatives_in_scan_space(mima,scan,susc,EDDY::ALL));
  } EddyCatch

};


//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class FieldUtils
//
// Helper Class used to perform various useful calculations
// on off-resonance and displacement fields.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class FieldUtils
{
public:
  // Rigid body transform of off-resonance field
  static NEWIMAGE::volume<float> RigidBodyTransformHzField(const NEWIMAGE::volume<float>& hzfield);

  // Some conversion routines off-resonance->displacements
  static NEWIMAGE::volume4D<float> Hz2VoxelDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                         const AcqPara&                 acqp);
  static NEWIMAGE::volume4D<float> Hz2MMDisplacements(const NEWIMAGE::volume<float>& hzfield,
                                                      const AcqPara&                 acqp);
  static NEWIMAGE::volume4D<float> Voxel2MMDisplacements(const NEWIMAGE::volume4D<float>& voxdisp) EddyTry {
    NEWIMAGE::volume4D<float> mmd=voxdisp; mmd[0] *= mmd.xdim(); mmd[1] *= mmd.ydim(); mmd[2] *= mmd.zdim(); return(mmd);
  } EddyCatch

  // Inverts 3D displacement field, ASSUMING it is really 1D (only non-zero displacements in one direction).
  static NEWIMAGE::volume4D<float> Invert3DDisplacementField(// Input
							     const NEWIMAGE::volume4D<float>& dfield,
							     const AcqPara&                   acqp,
							     const NEWIMAGE::volume<float>& inmask,
							     // Output
							     NEWIMAGE::volume<float>&       omask);

  // Inverts 1D displacement field. Input must be scaled to voxels (i.e. not mm).
  static NEWIMAGE::volume<float> Invert1DDisplacementField(// Input
							   const NEWIMAGE::volume<float>& dfield,
							   const AcqPara&                 acqp,
							   const NEWIMAGE::volume<float>& inmask,
							   // Output
							   NEWIMAGE::volume<float>&       omask);

  // Calculates Jacobian of a displacement field
  static NEWIMAGE::volume<float> GetJacobian(const NEWIMAGE::volume4D<float>& dfield,
                                             const AcqPara&                   acqp);

  // Calculates Jacobian of a 1D displacement field
  static NEWIMAGE::volume<float> GetJacobianFrom1DField(const NEWIMAGE::volume<float>& dfield,
							unsigned int                   dir);
private:
};

/****************************************************************//**
*
* \brief This class estimates amount of s2v movement
*
* This class estimates amount of s2v movement
*
********************************************************************/
class s2vQuant
{
public:
  s2vQuant(const ECScanManager&  sm) EddyTry : _sm(sm), _trth(0.3), _rotth(0.3) { common_construction(); } EddyCatch
  s2vQuant(const ECScanManager&  sm,
	   double                trth,
	   double                rotth) EddyTry : _sm(sm), _trth(trth), _rotth(rotth) { common_construction(); } EddyCatch
  /// Returns a vector of indicies of "still" volumes
  std::vector<unsigned int> FindStillVolumes(ScanType st, const std::vector<unsigned int>& mbsp) const;
private:
  /// Performs common construction tasks
  void common_construction();
  const ECScanManager&    _sm;     ///< Local copy of the scan manager
  NEWMAT::Matrix          _tr;     ///< Matrix with across-volume std of translations
  NEWMAT::Matrix          _rot;    ///< Matrix with across-volume std of rotations
  double                  _trth;   ///< Threshold for mean translation std for being considered "still"
  double                  _rotth;  ///< Threshold for mean rotation std for being considered "still"
};

} // End namespace EDDY

#endif // End #ifndef EddyUtils_h
