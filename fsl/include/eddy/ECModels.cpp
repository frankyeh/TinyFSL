// Declarations of classes that implements a hirearchy
// of models for fields from eddy currents induced by
// diffusion gradients.
//
// ECModels.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2011 University of Oxford
//

#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <time.h>
#include "armawrap/newmat.h"
#ifndef EXPOSE_TREACHEROUS
#define EXPOSE_TREACHEROUS           // To allow us to use .set_sform etc
#endif
#include "newimage/newimageall.h"
#include "miscmaths/miscmaths.h"
#include "topup/topup_file_io.h"
#include "EddyHelperClasses.h"
#include "ECModels.h"

using namespace EDDY;

void DerivativeInstructions::SetSecondary(unsigned int               i,
					  unsigned int               index,
					  const SliceDerivModulator& sdm) EddyTry
{
  this->check_index(i,"SetSecondary");
  if (_mt != ModulationType::SliceWise) throw EddyException("DerivativeInstructions::SetSecondary: Wrong modulator for slice object");
  _scnd[i]._index = index;
  _scnd[i]._slmod = sdm;
  _set[i] = true;
  return;
} EddyCatch

void DerivativeInstructions::SetSecondary(unsigned int                 i,
					  unsigned int                 index,
					  const SpatialDerivModulator& sdm) EddyTry
{
  this->check_index(i,"SetSecondary");
  if (_mt != ModulationType::Spatial) throw EddyException("DerivativeInstructions::SetSecondary: Wrong modulator for spatial object");
  _scnd[i]._index = index;
  _scnd[i]._spmod = sdm;
  _set[i] = true;
  return;
} EddyCatch

EDDY::DerivativeInstructions ScanMovementModel::GetCompoundDerivInstructions(unsigned int                 indx,
									     const EDDY::MultiBandGroups& mbg) const EddyTry
{
  if (indx>5) throw EddyException("ScanMovementModel::GetCompoundDerivInstructions: indx out of range");
  float scale = (indx<3) ? 1e-2 : 1e-5;
  EDDY::DerivativeInstructions di(indx*(_order+1),scale,ModulationType::SliceWise,_order);
  NEWMAT::Matrix X = get_design(mbg.NGroups());
  for (unsigned int i=0; i<_order; i++) {
    std::vector<float> sw_wgt(mbg.NSlices()); // Slice wise weight
    for (unsigned int j=0; j<mbg.NGroups(); j++) {
      std::vector<unsigned int> satp = mbg.SlicesAtTimePoint(j);
      for (unsigned int k=0; k<satp.size(); k++) sw_wgt[satp[k]] = X(j+1,i+2);
    }
    di.SetSecondary(i,di.GetPrimaryIndex()+1+i,EDDY::SliceDerivModulator(sw_wgt));
  }
  return(di);
} EddyCatch

NEWMAT::Matrix ScanMovementModel::ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  if (_order) return(TOPUP::MovePar2Matrix(get_zero_order_mp(),scan));
  else return(TOPUP::MovePar2Matrix(_mp,scan));
} EddyCatch

NEWMAT::Matrix ScanMovementModel::ForwardMovementMatrix(const NEWIMAGE::volume<float>& scan,
							unsigned int                   grp,
							unsigned int                   ngrp) const EddyTry
{
  if (grp>=ngrp) throw EddyException("ScanMovementModel::ForwardMovementMatrix: grp has to be smaller than ngrp");
  NEWMAT::ColumnVector gmp = get_gmp(grp,ngrp);
  return(TOPUP::MovePar2Matrix(gmp,scan));
} EddyCatch

NEWMAT::Matrix ScanMovementModel::RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
								  const std::vector<unsigned int>&     rindx) const EddyTry
{
  NEWMAT::ColumnVector rmp;
  if (_order) rmp = get_zero_order_mp();
  else rmp = _mp;
  for (unsigned int i=0; i<rindx.size(); i++) {
    if (rindx[i] > 5) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: rindx has to be less than 6");
    else rmp(rindx[i]+1) = 0.0;
  }
  return(TOPUP::MovePar2Matrix(rmp,scan));
} EddyCatch

NEWMAT::Matrix ScanMovementModel::RestrictedForwardMovementMatrix(const NEWIMAGE::volume<float>&       scan,
								  unsigned int                         grp,
								  unsigned int                         ngrp,
								  const std::vector<unsigned int>&     rindx) const EddyTry
{
  if (grp>=ngrp) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: grp has to be smaller than ngrp");
  NEWMAT::ColumnVector rgmp = get_gmp(grp,ngrp);
  for (unsigned int i=0; i<rindx.size(); i++) {
    if (rindx[i] > 5) throw EddyException("ScanMovementModel::RestrictedForwardMovementMatrix: rindx has to be less than 6");
    else rgmp(rindx[i]+1) = 0.0;
  }
  return(TOPUP::MovePar2Matrix(rgmp,scan));
} EddyCatch

NEWIMAGE::volume<float> LinearScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(4);
  for (int k=0; k<field.zsize(); k++) {
    float zcomp = _ep(3)*field.zdim()*(k-(field.zsize()-1)/2.0);
    for (int j=0; j<field.ysize(); j++) {
      float ycomp = _ep(2)*field.ydim()*(j-(field.ysize()-1)/2.0);
      for (int i=0; i<field.xsize(); i++) {
        field(i,j,k) += _ep(1)*field.xdim()*(i-(field.xsize()-1)/2.0) + ycomp + zcomp;
      }
    }
  }
  return(field);
} EddyCatch

EDDY::DerivativeInstructions LinearScanECModel::GetCompoundDerivInstructions(unsigned int                     indx,
									     const std::vector<unsigned int>& pev) const EddyTry
{
  if (indx<1) { // This has the constant phase as primary and is modulated in two (non-PE) directions
    float scale = this->GetDerivScale(3,true);
    EDDY::DerivativeInstructions di(3,scale,ModulationType::Spatial,2);
    std::vector<unsigned int> zmod{0,0,1}; // Modulation in z
    di.SetSecondary(0,2,EDDY::SpatialDerivModulator(zmod));
    if (pev[0]) { // If phase encode in x
      std::vector<unsigned int> ymod{0,1,0}; // Modulation in y
      di.SetSecondary(1,1,EDDY::SpatialDerivModulator(ymod));
    }
    else if (pev[1]) { // If phase encode in y
      std::vector<unsigned int> xmod{1,0,0}; // Modulation in x
      di.SetSecondary(1,0,EDDY::SpatialDerivModulator(xmod));
    }
    return(di);
  }
  else if (indx<2) { // This is the linear phase in the PE direction
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(pedir);
    EDDY::DerivativeInstructions di(pedir,scale,ModulationType::Spatial,0);
    return(di);
  }
  else throw EddyException("LinearScanECModel::GetCompoundDerivInstructions: indx out of range");
} EddyCatch

NEWIMAGE::volume<float> QuadraticScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(10); // DC offset
  for (int k=0; k<field.zsize(); k++) {
    double z = field.zdim()*(k-(field.zsize()-1)/2.0);
    double zcomp = _ep(3)*z;
    double z2comp = _ep(6)*z*z;
    for (int j=0; j<field.ysize(); j++) {
      double y = field.ydim()*(j-(field.ysize()-1)/2.0);
      double ycomp = _ep(2)*y;
      double y2comp = _ep(5)*y*y;
      double yzcomp = _ep(9)*y*z;
      for (int i=0; i<field.xsize(); i++) {
        double x = field.xdim()*(i-(field.xsize()-1)/2.0);
        double xcomp = _ep(1)*x;
        double x2comp = _ep(4)*x*x;
	double xycomp = _ep(7)*x*y;
	double xzcomp = _ep(8)*x*z;
        field(i,j,k) += xcomp + ycomp + zcomp + x2comp + y2comp + z2comp + xycomp + xzcomp + yzcomp;
      }
    }
  }
  return(field);
} EddyCatch

EDDY::DerivativeInstructions QuadraticScanECModel::GetCompoundDerivInstructions(unsigned int                     indx,
										const std::vector<unsigned int>& pev) const EddyTry
{
  if (indx<1) { // This has the constant phase and is modulated linearly and quadratically in two directions.
    float scale = this->GetDerivScale(9,true);
    EDDY::DerivativeInstructions di(9,scale,ModulationType::Spatial,5);
    std::vector<unsigned int> mod{0,0,1}; // Linear moulation in z
    di.SetSecondary(0,2,EDDY::SpatialDerivModulator(mod));
    mod = {0,0,2}; // Quadratic modulation in z
    di.SetSecondary(1,5,EDDY::SpatialDerivModulator(mod));
    if (pev[0]) { // If phase encode in x
      mod = {0,1,0}; // Linear modulation in y
      di.SetSecondary(2,1,EDDY::SpatialDerivModulator(mod));
      mod = {0,2,0}; // Quadratic modulation in y
      di.SetSecondary(3,4,EDDY::SpatialDerivModulator(mod));
      mod = {0,1,1}; // Linear modulation in y _and_ z
      di.SetSecondary(4,8,EDDY::SpatialDerivModulator(mod));
    }
    else if (pev[1]) { // If phase encode in y
      mod = {1,0,0}; // Linear modulation in x
      di.SetSecondary(2,0,EDDY::SpatialDerivModulator(mod));
      mod = {2,0,0}; // Quadratic modulation in x
      di.SetSecondary(3,3,EDDY::SpatialDerivModulator(mod));
      mod = {1,0,1}; // Linear modulation in x _and_ z
      di.SetSecondary(4,7,EDDY::SpatialDerivModulator(mod));
    }
    return(di);
  }
  else if (indx<2) { // This has the linear phase in the PE-direction and is modulated linearly in two directions
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(pedir);
    EDDY::DerivativeInstructions di(pedir,scale,ModulationType::Spatial,2);
    if (pev[0]) { // If phase encode in x
      std::vector<unsigned int> mod{0,1,0}; // Linear moulation in y
      di.SetSecondary(0,6,EDDY::SpatialDerivModulator(mod));
      mod = {0,0,1}; // Linear modulation in z
      di.SetSecondary(1,7,EDDY::SpatialDerivModulator(mod));
    }
    else if (pev[1]) { // If phase encode in y
      std::vector<unsigned int> mod{1,0,0}; // Linear moulation in x
      di.SetSecondary(0,6,EDDY::SpatialDerivModulator(mod));
      mod = {0,0,1}; // Linear modulation in z
      di.SetSecondary(1,8,EDDY::SpatialDerivModulator(mod));
    }
    return(di);
  }
  else if (indx<3) { // This has the quadratic phase in the PE-direction and is not used for any other derivatives
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(3+pedir);
    EDDY::DerivativeInstructions di(3+pedir,scale,ModulationType::Spatial,0);
    return(di);
  }
  else throw EddyException("QuadraticScanECModel::GetCompoundDerivInstructions: indx out of range");
} EddyCatch

NEWIMAGE::volume<float> CubicScanECModel::ECField(const NEWIMAGE::volume<float>& scan) const EddyTry
{
  NEWIMAGE::volume<float> field = scan;
  field = _ep(20); // DC offset
  for (int k=0; k<field.zsize(); k++) {
    double z = field.zdim()*(k-(field.zsize()-1)/2.0);
    double z2 = z*z;
    double zcomp = _ep(3)*z;
    double z2comp = _ep(6)*z2;
    double z3comp = _ep(12)*z*z2;
    for (int j=0; j<field.ysize(); j++) {
      double y = field.ydim()*(j-(field.ysize()-1)/2.0);
      double y2 = y*y;
      double ycomp = _ep(2)*y;
      double y2comp = _ep(5)*y2;
      double y3comp = _ep(11)*y*y2;
      double yzcomp = _ep(9)*y*z;
      double y2zcomp = _ep(17)*y2*z;
      double yz2comp = _ep(19)*y*z2;
      for (int i=0; i<field.xsize(); i++) {
        double x = field.xdim()*(i-(field.xsize()-1)/2.0);
	double x2 = x*x;
        double xcomp = _ep(1)*x;
        double x2comp = _ep(4)*x2;
        double x3comp = _ep(10)*x*x2;
	double xycomp = _ep(7)*x*y;
	double xzcomp = _ep(8)*x*z;
	double x2ycomp = _ep(13)*x2*y;
	double x2zcomp = _ep(14)*x2*z;
	double xyzcomp = _ep(15)*x*y*z;
	double xy2comp = _ep(16)*x*y2;
	double xz2comp = _ep(18)*x*z2;
        field(i,j,k) += xcomp + ycomp + zcomp + x2comp + y2comp + z2comp + xycomp + xzcomp + yzcomp;
        field(i,j,k) += x3comp + y3comp + z3comp + x2ycomp + x2zcomp + xyzcomp + xy2comp + y2zcomp + xz2comp + yz2comp;
      }
    }
  }
  return(field);
} EddyCatch

EDDY::DerivativeInstructions CubicScanECModel::GetCompoundDerivInstructions(unsigned int                     indx,
									    const std::vector<unsigned int>& pev) const EddyTry
{
  if (indx<1) { // This has the constant phase and is modulated linearly, quadratically and cubically in two directions.
    float scale = this->GetDerivScale(19,true);
    EDDY::DerivativeInstructions di(19,scale,ModulationType::Spatial,9);
    std::vector<unsigned int> mod{0,0,1}; // Linear modulation in z
    di.SetSecondary(0,2,EDDY::SpatialDerivModulator(mod));
    mod = {0,0,2}; // Quadratic modulation in z
    di.SetSecondary(1,5,EDDY::SpatialDerivModulator(mod));
    mod = {0,0,3}; // Cubic modulation in z
    di.SetSecondary(2,11,EDDY::SpatialDerivModulator(mod));
    if (pev[0]) { // If phase encode in x
      mod = {0,1,0}; // Linear modulation in y
      di.SetSecondary(3,1,EDDY::SpatialDerivModulator(mod));
      mod = {0,2,0}; // Quadratic modulation in y
      di.SetSecondary(4,4,EDDY::SpatialDerivModulator(mod));
      mod = {0,3,0}; // Cubic modulation in y
      di.SetSecondary(5,10,EDDY::SpatialDerivModulator(mod));
      mod = {0,1,1}; // Linear modulation in y _and_ z
      di.SetSecondary(6,8,EDDY::SpatialDerivModulator(mod));
      mod = {0,2,1}; // Quadratic modulation in y and linear in z
      di.SetSecondary(7,16,EDDY::SpatialDerivModulator(mod));
      mod = {0,1,2}; // Linear modulation in y and quadratic in z
      di.SetSecondary(8,18,EDDY::SpatialDerivModulator(mod));
    }
    else if (pev[1]) { // If phase encode in y
      mod = {1,0,0}; // Linear modulation in x
      di.SetSecondary(3,0,EDDY::SpatialDerivModulator(mod));
      mod = {2,0,0}; // Quadratic modulation in x
      di.SetSecondary(4,3,EDDY::SpatialDerivModulator(mod));
      mod = {3,0,0}; // Cubic modulation in x
      di.SetSecondary(5,9,EDDY::SpatialDerivModulator(mod));
      mod = {1,0,1}; // Linear modulation in x _and_ z
      di.SetSecondary(6,7,EDDY::SpatialDerivModulator(mod));
      mod = {2,0,1}; // Quadratic modulation in x and linear in z
      di.SetSecondary(7,13,EDDY::SpatialDerivModulator(mod));
      mod = {1,0,2}; // Linear modulation in x and quadratic in z
      di.SetSecondary(8,17,EDDY::SpatialDerivModulator(mod));
    }
    return(di);
  }
  else if (indx<2) { // This has the linear phase in the PE-direction and is modulated linearly and quadratically in two directions
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(pedir);
    EDDY::DerivativeInstructions di(pedir,scale,ModulationType::Spatial,5);
    if (pev[0]) { // If phase encode in x
      std::vector<unsigned int> mod{0,1,0}; // Linear modulation in y
      di.SetSecondary(0,6,EDDY::SpatialDerivModulator(mod));
      mod = {0,0,1}; // Linear modulation in z
      di.SetSecondary(1,7,EDDY::SpatialDerivModulator(mod));
      mod = {0,2,0}; // Quadratic modulation in y
      di.SetSecondary(2,15,EDDY::SpatialDerivModulator(mod));
      mod ={0,1,1}; // Linear modulation in y _and_ z
      di.SetSecondary(3,14,EDDY::SpatialDerivModulator(mod));
      mod ={0,0,2}; // Quadratic modulation in z
      di.SetSecondary(4,17,EDDY::SpatialDerivModulator(mod));
    }
    else if (pev[1]) { // If phase encode in y
      std::vector<unsigned int> mod{1,0,0}; // Linear modulation in x
      di.SetSecondary(0,6,EDDY::SpatialDerivModulator(mod));
      mod ={0,0,1}; // Linear modulation in z
      di.SetSecondary(1,8,EDDY::SpatialDerivModulator(mod));
      mod ={2,0,0}; // Quadratic modulation in x
      di.SetSecondary(2,12,EDDY::SpatialDerivModulator(mod));
      mod ={1,0,1}; // Linear modulation in x _and_ z
      di.SetSecondary(3,14,EDDY::SpatialDerivModulator(mod));
      mod ={0,0,2}; // Quadratic modulation in z
      di.SetSecondary(4,18,EDDY::SpatialDerivModulator(mod));
    }
    return(di);
  }
  else if (indx<3) { // This has the quadratic phase in the PE-direction and is used for linear modulation in two directions
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(3+pedir);
    EDDY::DerivativeInstructions di(3+pedir,scale,ModulationType::Spatial,2);
    if (pev[0]) { // If phase encode in x
      std::vector<unsigned int> mod{0,1,0}; // Linear modulation in y
      di.SetSecondary(0,12,EDDY::SpatialDerivModulator(mod));
      mod = {0,0,1}; // Linear modulation in z
      di.SetSecondary(1,13,EDDY::SpatialDerivModulator(mod));
    }
    else if (pev[1]) { // If phase encode in y
      std::vector<unsigned int> mod{1,0,0}; // Linear modulation in x
      di.SetSecondary(0,15,EDDY::SpatialDerivModulator(mod));
      mod = {0,0,1}; // Linear modulation in z
      di.SetSecondary(1,16,EDDY::SpatialDerivModulator(mod));
    }
    return(di);
  }
  else if (indx<4) { // This has the Cubic phase in the PE-direction and is not used for any other derivatives
    unsigned int pedir = (pev[0]) ? 0 : 1;
    float scale = this->GetDerivScale(9+pedir);
    EDDY::DerivativeInstructions di(9+pedir,scale,ModulationType::Spatial,0);
    return(di);
  }
  else throw EddyException("CubicScanECModel::GetCompoundDerivInstructions: indx out of range");
} EddyCatch
