/////////////////////////////////////////////////////////////////////
///
/// \file CudaVolume.h
/// \brief Declarations of class intended to mimic some of the functionality of Newimage, but on the GPU.
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2012.
/// \Copyright (C) 2012 University of Oxford
///
/////////////////////////////////////////////////////////////////////

#ifndef CudaVolume_h
#define CudaVolume_h

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <thrust/system_error.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/fill.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "armawrap/newmat.h"
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "EddyKernels.h"

namespace EDDY {

class CudaImageCoordinates;
class CudaVolume4D;
class CudaVolume3D_2_4D_Helper;
class CudaVolume;

/****************************************************************//**
*
* \brief Helper class that implements some tasks common to
* CudaVolume and CudaVolume4D.
*
********************************************************************/
class cuda_volume_utils
{
private:
  friend class CudaVolume;
  friend class CudaVolume4D;
  static const int threads_per_block_convolve_1D = 128;
  static float sqr(float a) { return(a*a); }
  static thrust::host_vector<float> gaussian_1D_kernel(float fwhm); // fwhm in voxels
  static void smooth(float                            fwhm, // fwhm in mm
		     const std::vector<unsigned int>& sz,
		     const NEWIMAGE::volume<float>&   hdr,
		     float                            *imaptr);
  static void divide_within_mask(const thrust::device_vector<float>& divisor,
				 const thrust::device_vector<float>& mask,
				 thrust::device_vector<float>::iterator  imbegin,
				 thrust::device_vector<float>::iterator  imend);
  static bool same_dim_size(const std::vector<int>&   sz1,
			    const std::vector<float>& vxs1,
			    const std::vector<int>&   sz2,
			    const std::vector<float>& vxs2);
};

/////////////////////////////////////////////////////////////////////
///
/// \brief Helper class for transfering NEWIMAGE volumes to and from
/// a CUDA device.
///
/////////////////////////////////////////////////////////////////////
class CudaVolume
{
public:
  /// Default constructor.
  CudaVolume() EddyTry : _spv(false), _sz(3,0) {} EddyCatch
  /// Construct a CudaVolume from another. ifcv determines if data or only the header is copied.
  CudaVolume(const CudaVolume& cv, bool ifcv=true) EddyTry : _spv(false), _hdr(cv._hdr), _sz(cv._sz) {
    if (ifcv) {_devec=cv._devec; _spcoef=cv._spcoef; _spv=cv._spv; } else _devec.resize(cv.Size());
  } EddyCatch
  /// Construct a CudaVolume from vol. ifvol determines if data or only the header is copied.
  CudaVolume(const NEWIMAGE::volume<float>& vol, bool ifvol=true) EddyTry : _spv(false), _sz(3,0) {
    common_assignment_from_newimage_vol(vol,ifvol);
  } EddyCatch
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const CudaVolume& cv) EddyTry {
    if (this != &cv) { _spv=false; _sz=cv._sz; _hdr=cv._hdr; _devec.resize(cv.Size()); _spcoef.clear(); }
  } EddyCatch
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const CudaVolume4D& cv);
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,false);
  } EddyCatch
  /// Sets header and GPU data to that in cv
  CudaVolume& operator=(const CudaVolume& cv) EddyTry {
    if (this != &cv) { _sz=cv._sz; _hdr=cv._hdr; _devec=cv._devec; _spcoef=cv._spcoef; _spv=cv._spv; } return(*this);
  } EddyCatch
  /// Sets header and GPU data to that in vol
  CudaVolume& operator=(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,true); return(*this);
  } EddyCatch
  /// Samples volume at points given by coord and returns it in smpl
  void Sample(const EDDY::CudaImageCoordinates& coord, CudaVolume& smpl) const;
  /// Samples volume at points given by coord and returns it in smpl with derivs in dsmpl
  void Sample(const EDDY::CudaImageCoordinates& coord, CudaVolume& smpl, CudaVolume4D& dsmpl) const;
  /// Check which coordinates falls outside FOV and for which extrapolation isn't valid
  void ValidMask(const EDDY::CudaImageCoordinates& coord, CudaVolume& mask) const;
  /// Do an interpolation in z given columns with varying z-ccordinates
  void ResampleStack(const CudaVolume& zcoord, const CudaVolume& inmask, CudaVolume oima) const;
  /// Adds GPU data in rhs to *this
  CudaVolume& operator+=(const CudaVolume& rhs);
  /// Subtracts GPU data in rhs from *this
  CudaVolume& operator-=(const CudaVolume& rhs);
  /// Multiplies GPU data in *this with rhs
  CudaVolume& operator*=(const CudaVolume& rhs);
  /// Divides GPU data in *this with scalar
  CudaVolume& operator/=(float a);
  /// + operator
  const CudaVolume operator+(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) += rhs); } EddyCatch
  /// - operator
  const CudaVolume operator-(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) -= rhs); } EddyCatch
  /// * operator
  const CudaVolume operator*(const CudaVolume& rhs) const EddyTry { return(CudaVolume(*this) *= rhs); } EddyCatch
  /// /scalar operator
  const CudaVolume operator/(float a) const EddyTry { return(CudaVolume(*this) /= a); } EddyCatch
  /// Smooths to requested FWHM
  void Smooth(float fwhm) EddyTry { cuda_volume_utils::smooth(fwhm,_sz,_hdr,this->GetPtr()); if (_spv) { _spcoef.clear(); _spv=false; } } EddyCatch
  /// Smooths to requested FWHM within mask
  void Smooth(float fwhm, const CudaVolume& mask);
  /// Performs += a*pv;
  void MultiplyAndAddToMe(const CudaVolume& pv, float a);
  /// Performs += a * (pv - nv);
  void SubtractMultiplyAndAddToMe(const CudaVolume& pv, const CudaVolume& nv, float a);
  /// Performs += pow(pv-nv,2);
  void SubtractSquareAndAddToMe(const CudaVolume& pv, const CudaVolume& nv);
  /// Divides one image with another for all voxels within mask
  void DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask);
  /// Set all voxels greater than val to one and the rest to zero
  CudaVolume& Binarise(float tv);
  /// Set all voxels > ll and < ul to one and the rest to zero
  CudaVolume& Binarise(float ll, float ul);
  /// Make volume with N(mu,sigma) distributed noise
  CudaVolume& MakeNormRand(float mu, float sigma);
  /// Returns sum of all voxel values inside mask
  double Sum(const CudaVolume& mask) const;
  /// Returns sum of all voxel values
  double Sum() const EddyTry { CudaVolume skrutt; return(Sum(skrutt)); } EddyCatch
  /// Returns sum-of-squares of all voxel values inside mask
  double SumOfSquares(const CudaVolume& mask) const;
  /// Returns sum-of-squares of all voxel values
  double SumOfSquares() const EddyTry { CudaVolume skrutt; return(SumOfSquares(skrutt)); } EddyCatch
  /// Returns max of all voxel values inside mask
  double Max(const CudaVolume& mask) const;
  /// Returns max of all voxel values
  double Max() const EddyTry { CudaVolume skrutt; return(Max(skrutt)); } EddyCatch
  /// Returns max of the absolute values all voxel values inside mask
  double MaxAbs(const CudaVolume& mask) const;
  /// Returns max of the absolute values all voxel values
  double MaxAbs() const EddyTry { CudaVolume skrutt; return(MaxAbs(skrutt)); } EddyCatch
  /// Assigns val to all voxels
  CudaVolume& operator=(float val);
  /// Returns true if basic image dimensions are the same. Does NOT consider the data.
  bool operator==(const CudaVolume& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const CudaVolume& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Returns true if basic image dimensions are the same. Does NOT consider the data.
  bool operator==(const NEWIMAGE::volume<float>& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const NEWIMAGE::volume<float>& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Returns true if basic image dimensions are the same. Does NOT consider the data.
  bool operator==(const CudaVolume4D& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const CudaVolume4D& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Writes some useful debug info to the screen. N.B. not a member function
  friend std::ostream& operator<<(std::ostream& out, const CudaVolume& cv) EddyTry {
    out << "Matrix size: " << cv._sz[0] << ", " << cv._sz[1] << ", " << cv._sz[2] << std::endl;
    out << "Voxel size: " << cv._hdr.xdim() << "mm, " << cv._hdr.ydim() << "mm, " << cv._hdr.zdim() << "mm" << std::endl;
    out << "_devec.size() = " << cv._devec.size() << ", _spv = " << cv._spv << ", _spcoef.size() = " << cv._spcoef.size();
    return(out);
  } EddyCatch
  /// Returns a pointer to the memory on the GPU
  float *GetPtr() EddyTry { _spv=false; return((Size()) ? thrust::raw_pointer_cast(_devec.data()) : 0); } EddyCatch
  /// Returns a const pointer to the memory on the GPU
  const float *GetPtr() const EddyTry { return((Size()) ? thrust::raw_pointer_cast(_devec.data()) : 0); } EddyCatch
  /// Returns an iterator to the start of the memory on the GPU
  thrust::device_vector<float>::iterator Begin() { _spv=false; return(_devec.begin()); }
  /// Returns an iterator to the end of the memory on the GPU
  thrust::device_vector<float>::iterator End() { _spv=false; return(_devec.end()); }
  /// Returns a const iterator to the start of the memory on the GPU
  thrust::device_vector<float>::const_iterator Begin() const { return(_devec.begin()); }
  /// Returns a const iterator to the end of the memory on the GPU
  thrust::device_vector<float>::const_iterator End() const { return(_devec.end()); }
  /// Returns the total size of the volume
  unsigned int Size() const { return(_sz[0]*_sz[1]*_sz[2]); }
  /// Returns the matrix size in direction indx, indx=0,1,2.
  unsigned int Size(unsigned int indx) const;
  /// Returns the voxel size (mm) in direction indx, indx=0,1,2.
  float Vxs(unsigned int indx) const;
  /// Returns image-to-world mapping matrix
  NEWMAT::Matrix Ima2WorldMatrix() const; //  { return(_hdr.sampling_mat()); } Actual definition in .cu
  /// Returns image-to-world mapping matrix
  NEWMAT::Matrix World2ImaMatrix() const; //  { return(_hdr.sampling_mat().i()); } Actual definition in .cu
  /// Returns interpolation method.
  NEWIMAGE::interpolation Interp() const EddyTry { return(_hdr.getinterpolationmethod()); } EddyCatch
  /// Returnd extrapolation method.
  NEWIMAGE::extrapolation Extrap() const EddyTry { return(_hdr.getextrapolationmethod()); } EddyCatch
  /// Returns a vector indicating in which directions extrapolation is valid (e.g. periodic in the PE direction).
  std::vector<bool> ExtrapValid() const EddyTry { return(_hdr.getextrapolationvalidity()); } EddyCatch
  /// Sets interpolation method
  void SetInterp(NEWIMAGE::interpolation im) EddyTry { _hdr.setinterpolationmethod(im); } EddyCatch
  /// Sets extrapolation method
  void SetExtrap(NEWIMAGE::extrapolation im) EddyTry { _hdr.setextrapolationmethod(im); } EddyCatch
  /// Copies the data from GPU into provided volume.
  void GetVolume(NEWIMAGE::volume<float>& ovol) const;
  /// Copies the data from GPU into returned volume.
  NEWIMAGE::volume<float> GetVolume() const EddyTry { NEWIMAGE::volume<float> ovol; GetVolume(ovol); return(ovol); } EddyCatch
  /// Writes image to disc
  void Write(const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetVolume(),fname); } EddyCatch
  /// Copies the spline coefficients from GPU into provided volume.
  void GetSplineCoefs(NEWIMAGE::volume<float>& ovol) const;
  /// Copies the spline coefficients from GPU into returned volume.
  NEWIMAGE::volume<float> GetSplineCoefs() const EddyTry { NEWIMAGE::volume<float> ovol; GetSplineCoefs(ovol); return(ovol); } EddyCatch
  /// Writes spline coefficients to disc
  void WriteSplineCoefs(const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetSplineCoefs(),fname); } EddyCatch
  friend class CudaVolume4D;              // To allow CudaVolume4D access to private members.
  friend class CudaVolume3D_2_4D_Helper;  // To allow CudaVolume3D_2_4D_Helper to access private members
private:
  static const int                       threads_per_block_interpolate = 128;
  static const int                       threads_per_block_deconv = 128;
  static const int                       threads_per_block_smaatm = 128;
  static const int                       threads_per_block_ssaatm = 128;

  thrust::device_vector<float>           _devec;
  mutable thrust::device_vector<float>   _spcoef;     // Spline coefficients for 3D deconv
  mutable bool                           _spv;        // True if spcoef valid
  NEWIMAGE::volume<float>                _hdr;
  std::vector<unsigned int>              _sz;

  void common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
					   bool                           ifvol);
  const float *sp_ptr() const EddyTry { return(thrust::raw_pointer_cast(_spcoef.data())); } EddyCatch
  void calculate_spline_coefs(const std::vector<unsigned int>&     sz,
			      const thrust::device_vector<float>&  ima,
			      thrust::device_vector<float>&        coef) const;
};

/////////////////////////////////////////////////////////////////////
///
/// \brief Helper class for transfering NEWIMAGE 4D volumes to and
/// from a CUDA device.
///
/////////////////////////////////////////////////////////////////////
class CudaVolume4D
{
public:
  CudaVolume4D() EddyTry : _sz(4,0) {} EddyCatch
  CudaVolume4D(const CudaVolume4D& cv, bool ifcv=true) EddyTry : _sz(cv._sz), _hdr(cv._hdr) {
    if (ifcv) _devec = cv._devec;
    else _devec.resize(cv._devec.size());
  } EddyCatch
  CudaVolume4D(const CudaVolume& cv, unsigned int nv, bool ifcv=true) EddyTry : _sz(4,0), _hdr(cv._hdr) {
    _sz[0]=cv._sz[0]; _sz[1]=cv._sz[1]; _sz[2]=cv._sz[2]; _sz[3]=nv;
    _devec.resize(_sz[3]*cv._devec.size());
    if (ifcv) for (int i=0; i<_sz[3]; i++) thrust::copy(cv._devec.begin(),cv._devec.end(),this->volbegin(i));
  } EddyCatch
  CudaVolume4D(const NEWIMAGE::volume<float>& vol, bool ifvol=true) EddyTry : _sz(4,0) {
    common_assignment_from_newimage_vol(vol,ifvol);
  } EddyCatch
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const CudaVolume4D& cv) EddyTry {
    if (this != &cv) { _sz=cv._sz; _hdr=cv._hdr; _devec.resize(cv._devec.size()); }
  } EddyCatch
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const CudaVolume& cv, unsigned int nv) EddyTry {
    _sz[0]=cv._sz[0]; _sz[0]=cv._sz[1]; _sz[2]=cv._sz[0]; _sz[3]=nv; _hdr=cv._hdr; _devec.resize(nv*cv._devec.size());
  } EddyCatch
  /// Sets header and allocates memory on GPU. Does NOT copy any data to the GPU.
  void SetHdr(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,false);
  } EddyCatch
  /// Sets header and GPU data to that in cv
  CudaVolume4D& operator=(const CudaVolume4D& cv) EddyTry {
    if (this != &cv) { _sz=cv._sz; _hdr=cv._hdr; _devec=cv._devec; } return(*this);
  } EddyCatch
  /// Sets header and GPU data to that in vol
  CudaVolume4D& operator=(const NEWIMAGE::volume<float>& vol) EddyTry {
    common_assignment_from_newimage_vol(vol,true); return(*this);
  } EddyCatch
  /// Allows for assignments of type FourD[i] = ThreeD;
  CudaVolume3D_2_4D_Helper operator[](unsigned int indx);
  /// Assigns a 3D CudaVolume to a "slot" in the 4D volume
  void SetVolume(unsigned int i, const CudaVolume& vol);
  /// Adds GPU data in cv to *this
  CudaVolume4D& operator+=(const CudaVolume4D& cv);
  /// Multiplies (masks) 4D data with 3D volume
  CudaVolume4D& operator*=(const CudaVolume& cv);
  /// Divides all volumes with another (3D volume) for all voxels within mask (3D).
  void DivideWithinMask(const CudaVolume& divisor, const CudaVolume& mask);
  /// Smooths (3D) to requested FWHM
  void Smooth(float fwhm) EddyTry { for (unsigned int i=0; i<_sz[3]; i++) cuda_volume_utils::smooth(fwhm,_sz,_hdr,this->GetPtr(i)); } EddyCatch
  /// Smooths (3D) to requested FWHM within mask
  void Smooth(float fwhm, const CudaVolume& mask);
    /// Assigns val to all voxels
  CudaVolume4D& operator=(float val);
  /// Returns true if basic image dimensions are the same. Does NOT consider the data or the fourth dimension.
  bool operator==(const CudaVolume4D& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const CudaVolume4D& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Returns true if basic image dimensions are the same. Does NOT consider the data or the fourth dimension.
  bool operator==(const CudaVolume& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const CudaVolume& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Returns true if basic image dimensions are the same. Does NOT consider the data or the fourth dimension.
  bool operator==(const NEWIMAGE::volume<float>& rhs) const;
  /// Same as !(lhs==rhs)
  bool operator!=(const NEWIMAGE::volume<float>& rhs) const EddyTry { return(!(*this==rhs)); } EddyCatch
  /// Returns a pointer to the memory on the GPU
  float *GetPtr() EddyTry { return(thrust::raw_pointer_cast(_devec.data())); } EddyCatch
  /// Returns a const pointer to the memory for a specific volume on the GPU
  const float *GetPtr() const EddyTry { return(thrust::raw_pointer_cast(_devec.data())); } EddyCatch
  /// Returns a pointer to the memory for a specific volume on the GPU
  float *GetPtr(unsigned int i) EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::GetPtr: index out of range");
    return(thrust::raw_pointer_cast(_devec.data())+i*this->Size());
  } EddyCatch
  /// Returns a const pointer to the memory for a specific volume on the GPU
  const float *GetPtr(unsigned int i) const EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::GetPtr: index out of range");
    return(thrust::raw_pointer_cast(_devec.data())+i*this->Size());
  } EddyCatch
  /// Returns an iterator to the start of the memory on the GPU
  thrust::device_vector<float>::iterator Begin(unsigned int i) EddyTry { return(this->volbegin(i)); } EddyCatch
  /// Returns an iterator to the end of the memory on the GPU
  thrust::device_vector<float>::iterator End(unsigned int i) EddyTry { return(this->volend(i)); } EddyCatch
  /// Returns a const iterator to the start of the memory on the GPU
  thrust::device_vector<float>::const_iterator Begin(unsigned int i) const EddyTry { return(this->volbegin(i)); } EddyCatch
  /// Returns a const iterator to the end of the memory on the GPU
  thrust::device_vector<float>::const_iterator End(unsigned int i) const EddyTry { return(this->volend(i)); } EddyCatch
  /// Returns the total size of one volume
  unsigned int Size() const { return(_sz[0]*_sz[1]*_sz[2]); }
  /// Returns the matrix size in direction indx, indx=0,1,2,3.
  unsigned int Size(unsigned int indx) const;
  /// Returns the voxel size (mm) in direction indx, indx=0,1,2.
  float Vxs(unsigned int indx) const;
  NEWIMAGE::interpolation Interp() const EddyTry { return(_hdr.getinterpolationmethod()); } EddyCatch
  NEWIMAGE::extrapolation Extrap() const EddyTry { return(_hdr.getextrapolationmethod()); } EddyCatch
  std::vector<bool> ExtrapValid() const EddyTry { return(_hdr.getextrapolationvalidity()); } EddyCatch
  /// Sets interpolation method
  void SetInterp(NEWIMAGE::interpolation im) EddyTry { _hdr.setinterpolationmethod(im); } EddyCatch
  /// Caclulates first derivative in direction i of volume i sampled trilinearly at voxel centres
  void SampleTrilinearDerivOnVoxelCentres(unsigned int dir, const CudaVolume& mask, CudaVolume& deriv, bool add_one=true) const;
  /// Copies the data from GPU into returned 4D volume.
  NEWIMAGE::volume4D<float> GetVolume() const EddyTry { NEWIMAGE::volume4D<float> ovol; GetVolume(ovol); return(ovol); } EddyCatch
  /// Copies the data from GPU into returned 4D volume.
  void GetVolume(NEWIMAGE::volume4D<float>& ovol) const;
  /// Copies the data from index'th volume into returned 3D volume
  NEWIMAGE::volume<float> GetVolume(unsigned int indx) const EddyTry { NEWIMAGE::volume<float> ovol; GetVolume(indx,ovol); return(ovol); } EddyCatch
  /// Copies the data from index'th volume into returned 3D volume
  void GetVolume(unsigned int indx, NEWIMAGE::volume<float>& ovol) const;
  /// Writes 4D volume to disc.
  void Write(const std::string& fname) const EddyTry { NEWIMAGE::write_volume4D(GetVolume(),fname); } EddyCatch
  /// Writes 3D volume to disc.
  void Write(unsigned int indx, const std::string& fname) const EddyTry { NEWIMAGE::write_volume(GetVolume(indx),fname); } EddyCatch
  friend class CudaVolume;                // To allow CudaVolume to access private members
  friend class CudaVolume3D_2_4D_Helper;  // To allow CudaVolume3D_2_4D_Helper to access private members
private:
  std::vector<unsigned int>                    _sz;
  NEWIMAGE::volume<float>                      _hdr;
  thrust::device_vector<float>                 _devec;

  void common_assignment_from_newimage_vol(const NEWIMAGE::volume<float>& vol,
					   bool                           ifvol);
  thrust::device_vector<float>::iterator volbegin(unsigned int i) EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::volbegin: index out of range");
    return(_devec.begin()+i*this->Size());
  } EddyCatch
  thrust::device_vector<float>::const_iterator volbegin(unsigned int i) const EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::volbegin:const: index out of range");
    return(_devec.begin()+i*this->Size());
  } EddyCatch
  thrust::device_vector<float>::iterator volend(unsigned int i) EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::volend: index out of range");
    if (i<_sz[3]-1) return(_devec.begin()+(i+1)*this->Size());
    else return(_devec.end());
  } EddyCatch
  thrust::device_vector<float>::const_iterator volend(unsigned int i) const EddyTry {
    if (i>=_sz[3]) throw EddyException("CudaVolume4D::End:const: index out of range");
    if (i<_sz[3]-1) return(_devec.begin()+(i+1)*this->Size());
    else return(_devec.end());
  } EddyCatch

};

/****************************************************************//**
*
* \brief Tiny helper class whos only purpose is to allow for
* skrutt[i] = plutt;
* where skrutt is of type CudaVolume4D and plutt of type CudaVolume.
*
********************************************************************/
class CudaVolume3D_2_4D_Helper
{
public:
  void operator=(const CudaVolume& threed);
  friend class CudaVolume4D; // To allow CudaVolume4D to access private members
private:
  CudaVolume3D_2_4D_Helper(CudaVolume4D& fourd, unsigned int indx) EddyTry : _fourd(fourd), _indx(indx) {} EddyCatch // N.B. Private
  CudaVolume4D& _fourd;
  unsigned int _indx;
};

/****************************************************************//**
*
* \brief Helper class that manages a set of image coordinates in a way that
* it enables calculation/implementation of partial derivatives of
* images w.r.t. transformation parameters.
*
********************************************************************/
class CudaImageCoordinates
{
public:
  CudaImageCoordinates() EddyTry : _xn(0), _yn(0), _zn(0), _init(false) {} EddyCatch
  CudaImageCoordinates(unsigned int xn, unsigned int yn, unsigned int zn, bool init=false) EddyTry
    : _xn(xn), _yn(yn), _zn(zn), _x(xn*yn*zn), _y(xn*yn*zn), _z(xn*yn*zn), _init(init) { if (init) init_coord(); } EddyCatch
  void Resize(unsigned int xn, unsigned int yn, unsigned int zn, bool init=false) EddyTry {
    _xn=xn; _yn=yn; _zn=zn;
    _x.resize(xn*yn*zn); _y.resize(xn*yn*zn); _y.resize(xn*yn*zn); _init=false;
    if (init) init_coord();
  } EddyCatch
  /// Affine transform
  void Transform(const NEWMAT::Matrix& A);
  /// Slice-wise affine transform
  void Transform(const std::vector<NEWMAT::Matrix>& A);
  /// General transform
  void Transform(const NEWMAT::Matrix& A, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& B);
  /// Slice-wise general transform
  void Transform(const std::vector<NEWMAT::Matrix>& A, const EDDY::CudaVolume4D& dfield, const std::vector<NEWMAT::Matrix>& B);
  /// Calculate x, y and z-cordinates for slice-to-vol (the tricky direction) transform
  void GetSliceToVolXYZCoord(const NEWMAT::Matrix& M1, const std::vector<NEWMAT::Matrix>& R, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& M2, EDDY::CudaVolume& zcoord);
  /// Calculate z-cordinates for slice-to-vol (the tricky direction) transform
  void GetSliceToVolZCoord(const NEWMAT::Matrix& M1, const std::vector<NEWMAT::Matrix>& R, const EDDY::CudaVolume4D& dfield, const NEWMAT::Matrix& M2);
  unsigned int Size() const { return(_xn*_yn*_zn); }
  unsigned int Size(unsigned int indx) const EddyTry {
    if (indx>2) throw EddyException("CudaImageCoordinates::Size: Index out of range.");
    return((!indx) ? _xn : ((indx==1) ? _yn : _zn));
  } EddyCatch
  CudaImageCoordinates& operator-=(const CudaImageCoordinates& rhs);
  const float *XPtr() const EddyTry { return(thrust::raw_pointer_cast(_x.data())); } EddyCatch
  const float *YPtr() const EddyTry { return(thrust::raw_pointer_cast(_y.data())); } EddyCatch
  const float *ZPtr() const EddyTry { return(thrust::raw_pointer_cast(_z.data())); } EddyCatch
  /// Returns coordinates as nx3 matrix. For debugging only.
  NEWMAT::Matrix AsMatrix() const;
  /// Writes list of coordinates to text-file
  void Write(const std::string& fname, unsigned int n=0) const;
private:
  float *XPtr() EddyTry { return(thrust::raw_pointer_cast(_x.data())); } EddyCatch
  float *YPtr() EddyTry { return(thrust::raw_pointer_cast(_y.data())); } EddyCatch
  float *ZPtr() EddyTry { return(thrust::raw_pointer_cast(_z.data())); } EddyCatch
  static const int             threads_per_block = 128;

  unsigned int                 _xn;
  unsigned int                 _yn;
  unsigned int                 _zn;
  thrust::device_vector<float> _x;
  thrust::device_vector<float> _y;
  thrust::device_vector<float> _z;
  bool                         _init;

  void init_coord();
  thrust::device_vector<float> repack_matrix(const NEWMAT::Matrix& A);
  thrust::device_vector<float> repack_vector_of_matrices(const std::vector<NEWMAT::Matrix>& A);
};


} // End namespace EDDY

#ifdef I_CUDAVOLUME_H_DEFINED_ET
#undef I_CUDAVOLUME_H_DEFINED_ET
#undef EXPOSE_TREACHEROUS   // Avoid exporting dodgy routines
#endif

#endif // End #ifndef CudaVolume_h
