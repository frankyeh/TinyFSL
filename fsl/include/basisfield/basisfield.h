// Declarations of class basisfield.
//
// This class cannot be used as it stands. Its purpose is
// to serve as a base-class for e.g. splinefield and DCTfield.
//
// basisfield.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2007 University of Oxford
//
//     CCOPYRIGHT
//

#ifndef basisfield_h
#define basisfield_h

#include <string>
#include <vector>
#include <memory>
#include "armawrap/newmat.h"
#include "newimage/newimage.h"
#include "miscmaths/bfmatrix.h"

namespace BASISFIELD {

const unsigned int MAX_SIZE = 2048; // Largest field I ever hope to see
  // enum BasisFieldPrecision {FloatPrec, DoublePrec};

class BasisfieldException: public std::exception
{
private:
  std::string m_msg;
public:
  BasisfieldException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("Basisfield:: msg=" + m_msg).c_str();
  }

  ~BasisfieldException() throw() {}
};

typedef enum _FieldIndex {FIELD=0, DFDX=1, DFDY=2, DFDZ=3} FieldIndex;

class basisfield
{
private:

  unsigned int                                           ndim;           // Dimensionality of field (1,2 or 3)
  std::vector<unsigned int>                              sz;             // Size of field in x-, y- and z-direction
  std::vector<double>                                    vxs;            // Voxel-size in x-, y- and z-direction
  std::shared_ptr<NEWMAT::ColumnVector>                coef;           // Basis coefficients
  std::vector<bool>                                      futd;           // If true, field accurately reflects coef
  std::vector<std::shared_ptr<NEWMAT::ColumnVector> >  field;          // Field, dfdx, dfdy, dfdz

protected:

  // Functions for use in this and derived classes

  virtual void set_update_flag(bool state, FieldIndex fi=FIELD) {futd[fi] = state;}
  virtual void set_coef_ptr(std::shared_ptr<NEWMAT::ColumnVector>& cptr) {coef = cptr;}
  // Get smart read/write pointer to updated field
  virtual std::shared_ptr<NEWMAT::ColumnVector> get(FieldIndex fi=FIELD);
  // Get smart read/write pointer to (possibly) not updated field
  virtual std::shared_ptr<NEWMAT::ColumnVector> get_ptr(FieldIndex fi=FIELD);
  virtual std::shared_ptr<NEWMAT::ColumnVector> get_coef() const;
  double get_coef(unsigned int i, unsigned int j, unsigned int k) const;
  double get_coef(unsigned int i) const;
  virtual void assign_basisfield(const basisfield& inf);
  virtual double peek_outside_fov(int i, int j, int k, FieldIndex fi) const {return(0.0);} // Should be replaced by proper method in derived classes

public:

  // Constructors and destructor, including assignment

  basisfield(const std::vector<unsigned int>& psz, const std::vector<double>& pvxz);
  basisfield(const basisfield& inf);
  basisfield& operator=(const basisfield& inf);
  virtual ~basisfield();

  // General utility functions

  virtual unsigned int NDim() const {return(ndim);}
  virtual bool UpToDate(FieldIndex fi=FIELD) const {return(futd[fi]);}
  virtual unsigned int CoefSz() const {return(CoefSz_x()*CoefSz_y()*CoefSz_z());}

  virtual unsigned int FieldSz() const {return(FieldSz_x()*FieldSz_y()*FieldSz_z());}
  virtual unsigned int FieldSz_x() const {return(sz[0]);}
  virtual unsigned int FieldSz_y() const {return(sz[1]);}
  virtual unsigned int FieldSz_z() const {return(sz[2]);}
  virtual double Vxs_x() const {return(vxs[0]);}
  virtual double Vxs_y() const {return(vxs[1]);}
  virtual double Vxs_z() const {return(vxs[2]);}

  virtual NEWMAT::ReturnMatrix mm2vox(unsigned int sz=4) const;
  virtual NEWMAT::ReturnMatrix vox2mm(unsigned int sz=4) const;

  // Looking at individual voxels of the field

  virtual double Peek(unsigned int x, unsigned int y, unsigned int z, FieldIndex fi=FIELD);
  virtual double Peek(unsigned int vi, FieldIndex fi=FIELD);
  virtual double UnsafePeek(unsigned int x, unsigned int y, unsigned int z, FieldIndex fi=FIELD) const {
    return(field[fi]->element(z*FieldSz_x()*FieldSz_y()+y*FieldSz_x()+x));
  }
  virtual double UnsafePeek(unsigned int vi, FieldIndex fi=FIELD) const {return(field[fi]->element(vi));}

  // Get values at individual voxels of the field, where the voxel
  // may be outside the actual defined FOV. I.e. x, y or z may be
  // negative or greater than FieldSz. For DCT the field will just
  // "continue" and for splines it will slowly taper off to zero
  // over the range of a few knot-spacings.

  virtual double PeekWide(int i, int j, int k, FieldIndex fi=FIELD);

  // Getting values of the field at non-integer voxel locations.
  // N.B. that the top one of these is pure virtual.
  virtual double Peek(double x, double y, double z, FieldIndex fi=FIELD) const = 0;
  virtual double Peek(float x, float y, float z, FieldIndex fi=FIELD) const {return(Peek(static_cast<double>(x),static_cast<double>(y),static_cast<double>(z)));}
  virtual double operator()(double x, double y, double z) const {return(Peek(x,y,z));}
  virtual double operator()(float x, float y, float z) const{return(Peek(x,y,z));}

  // Setting or getting coefficients, or getting whole field

  virtual void SetCoef(const NEWMAT::ColumnVector& pcoef);
  virtual void SetToConstant(double fv) = 0;
  virtual const std::shared_ptr<NEWMAT::ColumnVector> GetCoef() const {return(get_coef());}
  virtual double GetCoef(unsigned int i, unsigned int j, unsigned int k) const {return(get_coef(i,j,k));}
  virtual double GetCoef(unsigned int i) const {return(get_coef(i));}
  virtual const std::shared_ptr<NEWMAT::ColumnVector> Get(FieldIndex fi=FIELD) const {
    return(const_cast<BASISFIELD::basisfield *>(this)->get(fi));
  }
  virtual void Set(const NEWMAT::ColumnVector& pfield);
  virtual void Set(const NEWIMAGE::volume<float>& pfield);
  virtual void AsVolume(NEWIMAGE::volume<float>& vol, FieldIndex fi=FIELD);

  virtual void ScaleField(double sf) {
    NEWMAT::ColumnVector   coef(*GetCoef());
    SetCoef(sf*coef);
  }

  // Pure virtual functions that must be defined in derived classes

  virtual unsigned int CoefSz_x() const = 0;
  virtual unsigned int CoefSz_y() const = 0;
  virtual unsigned int CoefSz_z() const = 0;

  virtual bool HasGlobalSupport() const = 0;

  // Get the range of basis-functions with support at point xyz
  virtual void RangeOfBasesWithSupportAtXyz(const NEWMAT::ColumnVector&       xyz,
                                            std::vector<unsigned int>&        first,
                                            std::vector<unsigned int>&        last) const = 0;

  // Get the value of basis lmn at point xyz
  virtual double ValueOfBasisLmnAtXyz(const std::vector<unsigned int>&  lmn,
                                      const NEWMAT::ColumnVector&       xyz) const = 0;

  virtual std::vector<double> SubsampledVoxelSize(unsigned int               ss,
			                          std::vector<double>        vxs = std::vector<double>(),
				                  std::vector<unsigned int>  ms = std::vector<unsigned int>()) const = 0;
  virtual std::vector<double> SubsampledVoxelSize(const std::vector<unsigned int>&  ss,
			                          std::vector<double>               vxs = std::vector<double>(),
				                  std::vector<unsigned int>         ms = std::vector<unsigned int>()) const = 0;

  virtual std::vector<unsigned int> SubsampledMatrixSize(unsigned int               ss,
                                                         std::vector<unsigned int>  ms = std::vector<unsigned int>()) const = 0;
  virtual std::vector<unsigned int> SubsampledMatrixSize(const std::vector<unsigned int>&  ss,
                                                         std::vector<unsigned int>         ms = std::vector<unsigned int>()) const = 0;

  virtual void Update(FieldIndex fi=FIELD) = 0;

  virtual NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&  ima1,
                                   const NEWIMAGE::volume<float>&  ima2,
                                   const NEWIMAGE::volume<char>    *mask) const = 0;

  virtual NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                                   const NEWIMAGE::volume<float>&    ima1,
                                   const NEWIMAGE::volume<float>&    ima2,
                                   const NEWIMAGE::volume<char>      *mask) const = 0;

  virtual NEWMAT::ReturnMatrix Jte(const NEWIMAGE::volume<float>&    ima,
                                   const NEWIMAGE::volume<char>      *mask) const = 0;

  virtual NEWMAT::ReturnMatrix Jte(const std::vector<unsigned int>&  deriv,
                                   const NEWIMAGE::volume<float>&    ima,
                                   const NEWIMAGE::volume<char>      *mask) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>&    ima,
                                                     const NEWIMAGE::volume<char>      *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType  prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>&       ima1,
                                                     const NEWIMAGE::volume<float>&       ima2,
                                                     const NEWIMAGE::volume<char>         *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType     prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>&       deriv,
                                                     const NEWIMAGE::volume<float>&         ima,
                                                     const NEWIMAGE::volume<char>           *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType       prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>&       deriv,
                                                     const NEWIMAGE::volume<float>&         ima1,
                                                     const NEWIMAGE::volume<float>&         ima2,
                                                     const NEWIMAGE::volume<char>           *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType       prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const std::vector<unsigned int>&         deriv1,
                                                     const NEWIMAGE::volume<float>&           ima1,
                                                     const std::vector<unsigned int>&         deriv2,
                                                     const NEWIMAGE::volume<float>&           ima2,
                                                     const NEWIMAGE::volume<char>             *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType         prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> JtJ(const NEWIMAGE::volume<float>&        ima1,
                                                     const basisfield&                     bf2,
                                                     const NEWIMAGE::volume<float>&        ima2,
                                                     const NEWIMAGE::volume<char>          *mask=0,
                                                     MISCMATHS::BFMatrixPrecisionType      prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;


  virtual double MemEnergy() const = 0;
  virtual double BendEnergy() const = 0;

  virtual NEWMAT::ReturnMatrix MemEnergyGrad() const = 0;
  virtual NEWMAT::ReturnMatrix BendEnergyGrad() const = 0;

  virtual std::shared_ptr<MISCMATHS::BFMatrix> MemEnergyHess(MISCMATHS::BFMatrixPrecisionType   prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;
  virtual std::shared_ptr<MISCMATHS::BFMatrix> BendEnergyHess(MISCMATHS::BFMatrixPrecisionType   prec=MISCMATHS::BFMatrixDoublePrecision) const = 0;

  virtual std::shared_ptr<BASISFIELD::basisfield> ZoomField(const std::vector<unsigned int>&  psz,
                                                              const std::vector<double>&        pvxs,
                                                              std::vector<unsigned int>         well=std::vector<unsigned int>()) const = 0;

};

} // End namespace BASISFIELD

#endif // end #ifndef basisfield_h
