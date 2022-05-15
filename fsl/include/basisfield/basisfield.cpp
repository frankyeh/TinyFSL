// Definitions for class basisfield
//
// basisfield.cpp
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2007 University of Oxford
//
//     CCOPYRIGHT
//
#include <string>
#include <iostream>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimage.h"
#include "miscmaths/bfmatrix.h"
#include "basisfield.h"

using namespace std;
using namespace NEWMAT;
using namespace NEWIMAGE;
using namespace MISCMATHS;

namespace BASISFIELD {

// Constructor, assignement and destructor

basisfield::basisfield(const std::vector<unsigned int>& psz, const std::vector<double>& pvxs)
: ndim(psz.size()), sz(3,1), vxs(3,0.0), coef(), futd(4,false), field(4)
{
  if (psz.size()<1 || psz.size()>3) {throw BasisfieldException("basisfield::basisfield::Invalid dimensionality of field");}
  if (psz.size() != pvxs.size()) {throw BasisfieldException("basisfield::basisfield:: Dimensionality mismatch between psz and pvxs");}
  for (int i=0; i<3; i++) {
    if (i<int(psz.size()) && (psz[i] < 1 || psz[i] > MAX_SIZE)) {throw BasisfieldException("basisfield::basisfield::Invalid size of field");}
    sz[i] = (i<int(psz.size())) ? psz[i] : 1;
    vxs[i] = (i<int(pvxs.size())) ? pvxs[i] : 0.0;
  }
}

basisfield::basisfield(const basisfield& inf)
: ndim(inf.ndim), sz(3,1), vxs(3,0.0), coef(), futd(4,false), field(4)
{
  assign_basisfield(inf);
}

basisfield& basisfield::operator=(const basisfield& inf)
{
  if (&inf == this) {return(*this);} // Detect self
  assign_basisfield(inf);
  return(*this);
}


basisfield::~basisfield() {}

// General utility functions

ReturnMatrix basisfield::mm2vox(unsigned int sz) const
{
  Matrix rmat = IdentityMatrix(sz);
  rmat(1,1) = 1.0/Vxs_x();
  rmat(2,2) = 1.0/Vxs_y();
  rmat(3,3) = 1.0/Vxs_z();
  rmat.Release();
  return(rmat);
}

ReturnMatrix basisfield::vox2mm(unsigned int sz) const
{
  Matrix rmat = IdentityMatrix(sz);
  rmat(1,1) = Vxs_x();
  rmat(2,2) = Vxs_y();
  rmat(3,3) = Vxs_z();
  rmat.Release();
  return(rmat);
}

double basisfield::Peek(unsigned int x, unsigned int y, unsigned int z, FieldIndex fi)
{
  if ( x>=FieldSz_x() || y>=FieldSz_y() || z>=FieldSz_z()) {
    throw BasisfieldException("basisfield::PeekField:: Co-ordinates out of bounds");
  }
  if (!coef) {return(0.0);} // Consider field as zero if no coefficients set
  if (!UpToDate(fi)) {Update(fi);}

  return(UnsafePeek(z*FieldSz_x()*FieldSz_y()+y*FieldSz_x()+x,fi));
}

double basisfield::Peek(unsigned int vi, FieldIndex fi)
{
  if (vi>=FieldSz()) {throw BasisfieldException("basisfield::PeekField:: Voxel index out of bounds");}
  if (!coef) {return(0.0);} // Consider field as zero if no coefficients set
  if (!UpToDate(fi)) {Update(fi);}

  return(UnsafePeek(vi,fi));
}

double basisfield::PeekWide(int i, int j, int k, FieldIndex fi)
{
  if (!(i<0 || j<0 || k<0 || static_cast<unsigned int>(i)>=FieldSz_x() || static_cast<unsigned int>(j)>=FieldSz_y() || static_cast<unsigned int>(k)>=FieldSz_z())) {  // Inside "valid" FOV
    return(Peek(static_cast<unsigned int>(i),static_cast<unsigned int>(j),static_cast<unsigned int>(k),fi));
  }
  else {
    return(peek_outside_fov(i,j,k,fi));
  }
}

void basisfield::SetCoef(const ColumnVector& pcoef)
{
  if (pcoef.Nrows() != int(CoefSz())) {throw BasisfieldException("basisfield::SetCoef::Mismatch between input vector and # of coefficients");}
  if (!coef) {coef = std::shared_ptr<NEWMAT::ColumnVector>(new NEWMAT::ColumnVector(pcoef));}
  else {*coef = pcoef;}
  futd.assign(4,false);
}

/////////////////////////////////////////////////////////////////////
//
// Calulates and sets coefficients such that the field is the best
// possible approximation to the supplied field. The current
// implementation is not as efficient as it could be.
//
/////////////////////////////////////////////////////////////////////

void basisfield::Set(const volume<float>& pfield)
{
  if (int(FieldSz_x()) != pfield.xsize() || int(FieldSz_y()) != pfield.ysize() || int(FieldSz_z()) != pfield.zsize()) {
    throw BasisfieldException("basisfield::Set:: Matrix size mismatch beween basisfield class and supplied field");
  }
  if (Vxs_x() != pfield.xdim() || Vxs_y() != pfield.ydim() || Vxs_z() != pfield.zdim()) {
    throw BasisfieldException("basisfield::Set:: Voxel size mismatch beween basisfield class and supplied field");
  }

  volume<float>   volume_of_ones(pfield.xsize(),pfield.ysize(),pfield.zsize());
  volume_of_ones.copyproperties(pfield);
  volume_of_ones = 1.0;

  double lambda = 0.001;
  ColumnVector y = Jte(pfield,0);
  std::shared_ptr<MISCMATHS::BFMatrix>  XtX = JtJ(volume_of_ones);
  std::shared_ptr<MISCMATHS::BFMatrix>  BeEn = BendEnergyHess();
  XtX->AddToMe(*BeEn,lambda);
  ColumnVector coef_roof = XtX->SolveForx(y,SYM_POSDEF,1e-6,500);
  SetCoef(coef_roof);
}

/////////////////////////////////////////////////////////////////////
//
// Optional gateway to the Set( volume<float>& ) method.
//
/////////////////////////////////////////////////////////////////////

void basisfield::Set(const ColumnVector& pfield)
{
  if (pfield.Nrows() != int(FieldSz())) {throw BasisfieldException("basisfield::Set::Mismatch between input vector and size of field");}

  volume<float>  vol_pfield(FieldSz_x(),FieldSz_y(),FieldSz_z());
  vol_pfield.setdims(Vxs_x(),Vxs_y(),Vxs_z());
  vol_pfield.insert_vec(pfield);

  Set(vol_pfield);
}


void basisfield::AsVolume(volume<float>& vol, FieldIndex fi)
{
  if (int(FieldSz_x()) != vol.xsize() || int(FieldSz_y()) != vol.ysize() || int(FieldSz_z()) != vol.zsize()) {
    throw BasisfieldException("basisfield::AsVolume:: Matrix size mismatch beween field and volume");
  }
  if (Vxs_x() != vol.xdim() || Vxs_y() != vol.ydim() || Vxs_z() != vol.zdim()) {
    throw BasisfieldException("basisfield::AsVolume:: Voxel size mismatch beween field and volume");
  }
  if (!coef) {throw BasisfieldException("basisfield::AsVolume: Coefficients undefined");}

  if (!UpToDate(fi)) {Update(fi);}

  const std::shared_ptr<NEWMAT::ColumnVector> tmptr = Get(fi);
  int vindx=0;
  for (unsigned int k=0; k<FieldSz_z(); k++) {
    for (unsigned int j=0; j<FieldSz_y(); j++) {
      for (unsigned int i=0; i<FieldSz_x(); i++) {
        vol(i,j,k) = tmptr->element(vindx++);
      }
    }
  }
}

// Functions that are declared private or protected

void basisfield::assign_basisfield(const basisfield& inf) // Helper function for copy constructor and assignment
{
  futd = inf.futd;
  ndim = inf.ndim;
  vxs = inf.vxs;
  sz = inf.sz;
  coef = std::shared_ptr<NEWMAT::ColumnVector>(new NEWMAT::ColumnVector(*(inf.coef)));
  for (int i=0; i<int(inf.field.size()); i++) {
    if (inf.field[i]) {field[i] = std::shared_ptr<NEWMAT::ColumnVector>(new NEWMAT::ColumnVector(*(inf.field[i])));}
    else {field[i] = inf.field[i];}
  }
}

std::shared_ptr<NEWMAT::ColumnVector> basisfield::get(FieldIndex fi)
{
  if (!coef) {throw BasisfieldException("basisfield::Get: Coefficients undefined");}

  if (!UpToDate(fi)) {
    Update(fi);
  }
  return(field[fi]);
}

std::shared_ptr<NEWMAT::ColumnVector> basisfield::get_ptr(FieldIndex fi)
{
  if (!field[fi]) {
    field[fi] = std::shared_ptr<NEWMAT::ColumnVector>(new NEWMAT::ColumnVector(FieldSz()));
  }
  return(field[fi]);
}

std::shared_ptr<NEWMAT::ColumnVector> basisfield::get_coef() const
{
  if (!coef) {throw BasisfieldException("basisfield::get_coef: Coefficients undefined");}
  return(coef);
}

double basisfield::get_coef(unsigned int i, unsigned int j, unsigned int k) const
{
  if (i >= CoefSz_x() || j >= CoefSz_y() || k >= CoefSz_z()) throw BasisfieldException("basisfield::get_coef: i, j or k out of range");
  return(coef->element(k*CoefSz_x()*CoefSz_y()+j*CoefSz_x()+i));
}

double basisfield::get_coef(unsigned int i) const
{
  if (i >= CoefSz()) throw BasisfieldException("basisfield::get_coef: i out of range");
  return(coef->element(i));
}


} // End namespace BASISFIELD
