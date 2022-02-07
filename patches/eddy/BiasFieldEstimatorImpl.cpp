/*! \file BiasFieldEstimatorImpl.cpp
    \brief Contains one implementation of class for estimation of a bias field

    \author Jesper Andersson
    \version 1.0b, December, 2017.
*/
//
// BiasFieldEstimatorImpl.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2017 University of Oxford
//

#ifndef BiasFieldEstimatorImpl_h
#define BiasFieldEstimatorImpl_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <memory>
#include <time.h>
#include "armawrap/newmat.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "newimage/newimageall.h"
#include "miscmaths/SpMat.h"
#include "miscmaths/SpMatMatrices.h"
#include "EddyHelperClasses.h"
#include "BiasFieldEstimator.h"

namespace EDDY {

class BiasFieldEstimatorImpl
{
public:
  BiasFieldEstimatorImpl() : _nima(0) {}
  /// Set ref scan
  void SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  /// Add scan
  void AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed,
	       const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords);
  /// Calculate and return "direct" representation of the bias-field
  NEWIMAGE::volume<float> GetField(double lambda) const;
  /// Calculate and return spline basis-field representation of the bias-field
  NEWIMAGE::volume<float> GetField(double ksp, double lambda) const;
  /// Caclulate and return At matrix, for debug purposes
  MISCMATHS::SpMat<float> GetAtMatrix(const EDDY::ImageCoordinates&  coords,
				      const NEWIMAGE::volume<float>& predicted,
				      const NEWIMAGE::volume<float>& mask) const;
  /// Calculate and return At* matrix, for debug purposes
  MISCMATHS::SpMat<float> GetAtStarMatrix(const EDDY::ImageCoordinates&  coords, const NEWIMAGE::volume<float>& mask) const;
  /// Calculate and return A*b, where b is the field, for debug purposes
  NEWIMAGE::volume<float> GetATimesField(const EDDY::ImageCoordinates&  coords,
					 const NEWIMAGE::volume<float>& predicted,
					 const NEWIMAGE::volume<float>& mask,
					 const NEWIMAGE::volume<float>& field) const;
  /// Write out current state for debug purposes
  void Write(const std::string& basename) const EddyTry {
    MISCMATHS::write_ascii_matrix(_Aty,"BiasFieldEstimatorImpl_" + basename + "_Aty.txt");
    _AtA.Print("BiasFieldEstimatorImpl_" + basename + "_AtA.txt");
  } EddyCatch
private:
  unsigned int              _nima;    /// Number of loaded images
  std::vector<unsigned int> _isz;     /// Image matrix size
  std::vector<double>       _vxs;     /// Voxel size
  MISCMATHS::SpMat<float>   _rAt;     /// A^T matrix for reference volume
  MISCMATHS::SpMat<float>   _AtA;     /// \Sum A^T A in paper
  NEWMAT::ColumnVector      _Aty;     /// \Sum A^t f-\hat{f} in paper

  struct PointersForSpMat {
    PointersForSpMat(unsigned int N) : nz(0),
				       irp(std::unique_ptr<unsigned int[]>(new unsigned int[8*N])),
				       jcp(std::unique_ptr<unsigned int[]>(new unsigned int[N+1])),
				       sp(std::unique_ptr<double[]>(new double[8*N])) {}
    PointersForSpMat(PointersForSpMat&& rhs) : nz(rhs.nz),              // Moving is fine
					       irp(std::move(rhs.irp)),
					       jcp(std::move(rhs.jcp)),
					       sp(std::move(rhs.sp)) {}
    PointersForSpMat(const PointersForSpMat& rhs) = delete;             // No copying
    PointersForSpMat& operator=(const PointersForSpMat& rhs) = delete;  // No copying
    unsigned int                    nz;
    std::unique_ptr<unsigned int[]> irp;
    std::unique_ptr<unsigned int[]> jcp;
    std::unique_ptr<double[]>       sp;
  };

  unsigned int nvox() const { return(_isz[0]*_isz[1]*_isz[2]); }
  unsigned int ijk2indx(unsigned int i, unsigned int j, unsigned int k, const EDDY::ImageCoordinates& c) const {
    return(k*c.NX()*c.NY() + j*c.NX() + i);
  }
  void get_wgts(const double& dx, const double& dy, const double& dz,
		double& w000, double& w100, double& w010, double& w110,
		double& w001, double& w101, double& w011, double& w111) const
  {
    w000 = w100 = w010 = w110 = 1.0 - dz;
    w001 = w101 = w011 = w111 = dz;
    w000 *= 1.0 - dy; w100 *= 1.0 - dy; w001 *= 1.0 - dy; w101 *= 1.0 - dy;
    w010 *= dy; w110 *= dy; w011 *= dy; w111 *= dy;
    w000 *= 1.0 - dx; w010 *= 1.0 - dx; w001 *= 1.0 - dx; w011 *= 1.0 - dx;
    w100 *= dx; w110 *= dx; w101 *= dx; w111 *= dx;
  }
  PointersForSpMat make_At_star_CSC(// Input
				    const EDDY::ImageCoordinates&  coords,
				    const NEWIMAGE::volume<float>& mask) const;
  void multiply_At_star_CSC_by_image(const NEWIMAGE::volume<float>& ima,
				     const NEWIMAGE::volume<float>& mask,
				     unsigned int* const            jcp,
				     double* const                  sp) const;

};

BiasFieldEstimatorImpl::PointersForSpMat BiasFieldEstimatorImpl::make_At_star_CSC(// Input
										  const EDDY::ImageCoordinates&  coords,
										  const NEWIMAGE::volume<float>& mask) const EddyTry
{
  // Make A^T in a Compressed Column Storage representation
  PointersForSpMat ptrs(coords.N());
  unsigned int ii = 0; // Index intp irp and sp
  unsigned int ji = 0; // Index into jcp
  NEWMAT::ColumnVector vmask = mask.vec();
  for (unsigned int i=0; i<coords.N(); i++) {
    ptrs.jcp[ji++] = ii;
    if (vmask(i+1) > 0.0 && coords.IsInBounds(i)) { // If voxel falls within volume
      unsigned int xi = floor(coords.x(i));
      unsigned int yi = floor(coords.y(i));
      unsigned int zi = floor(coords.z(i));
      double dx = coords.x(i) - xi;
      double dy = coords.y(i) - yi;
      double dz = coords.z(i) - zi;
      if (dx < 1e-6 && dy < 1e-6 && dz < 1e-6) { // Voxel falls "exactly" on original voxel centre
	ptrs.irp[ii] = this->ijk2indx(xi,yi,zi,coords);
	ptrs.sp[ii++] = 1.0;
      }
      else {
	double w000, w100, w010, w110, w001, w101, w011, w111;
	get_wgts(dx,dy,dz,w000,w100,w010,w110,w001,w101,w011,w111);
	if (w000 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi,yi,zi,coords); ptrs.sp[ii++] = w000; }
	if (w100 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi+1,yi,zi,coords); ptrs.sp[ii++] = w100; }
	if (w010 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi,yi+1,zi,coords); ptrs.sp[ii++] = w010; }
	if (w110 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi+1,yi+1,zi,coords); ptrs.sp[ii++] = w110; }
	if (w001 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi,yi,zi+1,coords); ptrs.sp[ii++] = w001; }
	if (w101 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi+1,yi,zi+1,coords); ptrs.sp[ii++] = w101; }
	if (w011 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi,yi+1,zi+1,coords); ptrs.sp[ii++] = w011; }
	if (w111 > 1e-6) { ptrs.irp[ii] = this->ijk2indx(xi+1,yi+1,zi+1,coords); ptrs.sp[ii++] = w111; }
      }
    }
  }
  ptrs.jcp[ji] = ii;
  ptrs.nz = ii;
  return(ptrs);
} EddyCatch

void BiasFieldEstimatorImpl::multiply_At_star_CSC_by_image(const NEWIMAGE::volume<float>& ima,
							   const NEWIMAGE::volume<float>& mask,
							   unsigned int* const            jcp,
							   double* const                  sp) const EddyTry
{
  unsigned int indx = 0;
  for (int k=0; k<ima.zsize(); k++) {
    for (int j=0; j<ima.ysize(); j++) {
      for (int i=0; i<ima.xsize(); i++) {
	if (mask(i,j,k)) { for (unsigned int ii=jcp[indx]; ii<jcp[indx+1]; ii++) sp[ii] *= ima(i,j,k); }
	indx++;
      }
    }
  }
  return;
} EddyCatch

/*!
 * Creates what is effectively a "resampling matrix" (A*) for a "reference"
 * location. It is recommended that this is the average location of all volumes,
 * but it can in principle also be a "zero| position in which case the
 * resampling is simply an identity matrix.
 * \throw EDDY::EddyException if there is a mismatch between mask an previous reference
 */
void BiasFieldEstimatorImpl::SetRefScan(const NEWIMAGE::volume<float>& mask,    //<! [in] mask
					const EDDY::ImageCoordinates&  coords)  //<! [in] coordinates of desired reference
EddyTry
{
  if (!_rAt.Nrows()) { // If no ref scan has been set before
    _isz.resize(3);
    _isz[0] = static_cast<unsigned int>(mask.xsize());
    _isz[1] = static_cast<unsigned int>(mask.ysize());
    _isz[2] = static_cast<unsigned int>(mask.zsize());
    _vxs.resize(3);
    _vxs[0] = static_cast<double>(mask.xdim());
    _vxs[1] = static_cast<double>(mask.ydim());
    _vxs[2] = static_cast<double>(mask.zdim());
  }
  else { // Check that new ref scan is compatible with old
    if (static_cast<unsigned int>(mask.xsize()) != _isz[0] ||
	static_cast<unsigned int>(mask.ysize()) != _isz[1] ||
	static_cast<unsigned int>(mask.zsize()) != _isz[2]) {
      throw EddyException("BiasFieldEstimatorImpl::SetRefScan: Size mismatch between new and previously set ref image");
    }
  }
  PointersForSpMat ptrs = make_At_star_CSC(coords,mask);
  _rAt = MISCMATHS::SpMat<float>(coords.N(),coords.N(),ptrs.irp.get(),ptrs.jcp.get(),ptrs.sp.get());
  return;
} EddyCatch

/*!
 * Used to add a scan to _Aty and _AtA.
 * \throw EDDY::EddyException if there is a mismatch between input images
 */
void BiasFieldEstimatorImpl::AddScan(const NEWIMAGE::volume<float>& predicted, //<! [in] Predicted image
				     const NEWIMAGE::volume<float>& observed,  //<! [in] Observed image
				     const NEWIMAGE::volume<float>& mask,      //<! [in] Mask
				     const EDDY::ImageCoordinates&  coords)    //<! [in] Coordinates where observed image was "observed"
EddyTry
{
  static int cnt = 0;

  if (predicted.xsize() != observed.xsize() ||
      predicted.ysize() != observed.ysize() ||
      predicted.zsize() != observed.zsize()) {
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Size mismatch between predicted and observed image");
  }
  if (!_rAt.Nrows()) { // If no ref scan has been set yet
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Attempting to add scan before ref scan has been set");
  }
  if (static_cast<unsigned int>(predicted.xsize()) != _isz[0] ||
      static_cast<unsigned int>(predicted.ysize()) != _isz[1] ||
      static_cast<unsigned int>(predicted.zsize()) != _isz[2]) {
    throw EddyException("BiasFieldEstimatorImpl::AddScan: Size mismatch between predicted and previously set ref image");
  }

  NEWMAT::ColumnVector v_predicted = predicted.vec();
  NEWMAT::ColumnVector v_observed = observed.vec();
  NEWMAT::ColumnVector v_mask = mask.vec();
  _nima++;
  cnt++;
  PointersForSpMat ptrs = make_At_star_CSC(coords,mask);
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),ptrs.irp.get(),ptrs.jcp.get(),ptrs.sp.get());
  At -= _rAt;
  At.MultiplyColumns(NEWMAT::SP(v_predicted,v_mask));
  if (!_Aty.Nrows()) _Aty = At * NEWMAT::SP((v_observed - v_predicted),v_mask);
  else _Aty += At * NEWMAT::SP((v_observed - v_predicted),v_mask);
  if (!_AtA.Nrows()) _AtA = At*At.t();
  else _AtA += At*At.t();
} EddyCatch

/*!
 * Fits a "direct" representation of the bias-field to the data and return it in newimage format
 * \return A "direct" representation of the field in newimage format
 * \throw EDDY::EddyException if object not ready to estimate field
 */
NEWIMAGE::volume<float> BiasFieldEstimatorImpl::GetField(double lambda) const  //<! [in] Weight of Laplacian regularisation when fitting the field
EddyTry
{
  if (_nima==0) {
    throw EddyException("BiasFieldEstimatorImpl::GetField(double): The field cannot be estimated until images have been loaded");
  }
  // Make (approximate) Bending energy regularisation matrix
  MISCMATHS::SpMat<float> StS = MISCMATHS::Sparse3DBendingEnergyHessian(_isz,_vxs,MISCMATHS::PERIODIC);
  // Solve for field
  NEWMAT::ColumnVector field = (_AtA + (lambda/static_cast<double>(nvox())) * StS).SolveForx(_Aty,MISCMATHS::SYM_POSDEF,1.0e-6,1000);
  // Retreive field as NEWIMAGE volume
  NEWIMAGE::volume<float> fima(static_cast<int>(_isz[0]),static_cast<int>(_isz[1]),static_cast<int>(_isz[2]));
  fima.setxdim(_vxs[0]); fima.setydim(_vxs[1]); fima.setzdim(_vxs[2]);
  fima.insert_vec(field);
  return(fima);
} EddyCatch

/*!
 *  Fits a spline basis-field of the bias-field to the data and return it in newimage format
 *  \return A basis-field representation of the field in newimage format
 * \throw EDDY::EddyException if object not ready to estimate field
 */
NEWIMAGE::volume<float> BiasFieldEstimatorImpl::GetField(double ksp,           //<! [in] Knot-spacing of spline field in mm (approximate)
							 double lambda) const  //<! [in] Weight of Bending Energy regularisation when fitting the field
EddyTry
{
  if (_nima==0) {
    throw EddyException("BiasFieldEstimatorImpl::GetField(double, double): The field cannot be estimated until images have been loaded");
  }
  // knot-spacing mm->voxels
  std::vector<unsigned int> iksp(3);
  iksp[0] = static_cast<unsigned int>((ksp / _vxs[0]) + 0.5);
  iksp[1] = static_cast<unsigned int>((ksp / _vxs[1]) + 0.5);
  iksp[2] = static_cast<unsigned int>((ksp / _vxs[2]) + 0.5);
  // Make splinefield
  BASISFIELD::splinefield spf(_isz,_vxs,iksp);

  _AtA.Save("sum_AtA.txt");
  MISCMATHS::write_ascii_matrix(_Aty,"sum_Aty.txt");

  // Calculate B^T(\Sum A^T A)B and B^T(\Sum A^T \hat{f}) from paper
  std::shared_ptr<MISCMATHS::SpMat<float> > B = spf.J();
  MISCMATHS::SpMat<float> BtAtAB = B->t() * _AtA * *B;
  BtAtAB += (lambda/static_cast<double>(nvox())) * (*spf.BendEnergyHessAsSpMat());
  NEWMAT::ColumnVector BtAtf = B->t() * _Aty;
  // Solve for spline coefficients of field
  NEWMAT::ColumnVector b = BtAtAB.SolveForx(BtAtf,MISCMATHS::SYM_POSDEF,1.0e-6,200);
  // Set coefficients in field
  spf.SetCoef(b);
  // Retrieve field as NEWIMAGE volume
  NEWIMAGE::volume<float> field(static_cast<int>(_isz[0]),static_cast<int>(_isz[1]),static_cast<int>(_isz[2]));
  field.setxdim(_vxs[0]); field.setydim(_vxs[1]); field.setzdim(_vxs[2]);
  spf.AsVolume(field);

  return(field);
} EddyCatch

MISCMATHS::SpMat<float> BiasFieldEstimatorImpl::GetAtMatrix(const EDDY::ImageCoordinates&  coords,
							    const NEWIMAGE::volume<float>& predicted,
							    const NEWIMAGE::volume<float>& mask) const EddyTry
{
  PointersForSpMat ptrs = make_At_star_CSC(coords,mask);
  multiply_At_star_CSC_by_image(predicted,mask,ptrs.jcp.get(),ptrs.sp.get());
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),ptrs.irp.get(),ptrs.jcp.get(),ptrs.sp.get());
  return(At);
} EddyCatch

MISCMATHS::SpMat<float> BiasFieldEstimatorImpl::GetAtStarMatrix(const EDDY::ImageCoordinates&  coords,
								const NEWIMAGE::volume<float>& mask) const EddyTry
{
  PointersForSpMat ptrs = make_At_star_CSC(coords,mask);
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),ptrs.irp.get(),ptrs.jcp.get(),ptrs.sp.get());
  return(At);
} EddyCatch

NEWIMAGE::volume<float> BiasFieldEstimatorImpl::GetATimesField(const EDDY::ImageCoordinates&  coords,
							       const NEWIMAGE::volume<float>& predicted,
							       const NEWIMAGE::volume<float>& mask,
							       const NEWIMAGE::volume<float>& field) const EddyTry
{
  if (!_rAt.Nrows()) { // If no ref scan has been set yet
    throw EddyException("BiasFieldEstimatorImpl::GetATimesField: No ref scan set yet");
  }
  PointersForSpMat ptrs = make_At_star_CSC(coords,mask);
  MISCMATHS::SpMat<float> At(coords.N(),coords.N(),ptrs.irp.get(),ptrs.jcp.get(),ptrs.sp.get());
  At -= _rAt;
  At.MultiplyColumns(NEWMAT::SP(predicted.vec(),mask.vec()));
  MISCMATHS::SpMat<float> A = At.t();
  NEWMAT::ColumnVector fvec = field.vec();
  NEWMAT::ColumnVector Ab = A*fvec;
  NEWIMAGE::volume<float> Ab_ima = predicted;
  Ab_ima.insert_vec(Ab);
  return(Ab_ima);
} EddyCatch

BiasFieldEstimator::BiasFieldEstimator() EddyTry
{
  _pimpl = new BiasFieldEstimatorImpl();
} EddyCatch

BiasFieldEstimator::~BiasFieldEstimator()  { delete _pimpl; }

void BiasFieldEstimator::SetRefScan(const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords) EddyTry
{
  _pimpl->SetRefScan(mask,coords);
} EddyCatch

void BiasFieldEstimator::AddScan(const NEWIMAGE::volume<float>& predicted, const NEWIMAGE::volume<float>& observed,
				 const NEWIMAGE::volume<float>& mask, const EDDY::ImageCoordinates& coords) EddyTry
{
  _pimpl->AddScan(predicted,observed,mask,coords);
} EddyCatch

NEWIMAGE::volume<float> BiasFieldEstimator::GetField(double lambda) const EddyTry { return(_pimpl->GetField(lambda)); } EddyCatch

NEWIMAGE::volume<float> BiasFieldEstimator::GetField(double ksp, double lambda) const EddyTry { return(_pimpl->GetField(ksp,lambda)); } EddyCatch

MISCMATHS::SpMat<float> BiasFieldEstimator::GetAtMatrix(const EDDY::ImageCoordinates&  coords,
							const NEWIMAGE::volume<float>& predicted,
							const NEWIMAGE::volume<float>& mask) const EddyTry
{
  return(_pimpl->GetAtMatrix(coords,predicted,mask));
} EddyCatch

MISCMATHS::SpMat<float> BiasFieldEstimator::GetAtStarMatrix(const EDDY::ImageCoordinates&  coords,
							    const NEWIMAGE::volume<float>& mask) const EddyTry
{
  return(_pimpl->GetAtStarMatrix(coords,mask));
} EddyCatch

NEWIMAGE::volume<float> BiasFieldEstimator::GetATimesField(const EDDY::ImageCoordinates&  coords,
							   const NEWIMAGE::volume<float>& predicted,
							   const NEWIMAGE::volume<float>& mask,
							   const NEWIMAGE::volume<float>& field) EddyTry
{
  return(_pimpl->GetATimesField(coords,predicted,mask,field));
} EddyCatch

void BiasFieldEstimator::Write(const std::string& basename) const EddyTry { _pimpl->Write(basename); } EddyCatch

} // End namespace EDDY

#endif // End #ifndef BiasFieldEstimatorImpl_h
