//
//  Declarations for displacement vector class DispVec
//
//  displacement_vector.h
//
//  Implements a displacement vector class that can be
//  used to obtain inverses, K-matrices etc.
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2010 University of Oxford
//
/*  CCOPYRIGHT  */
#ifndef displacement_vector_h
#define displacement_vector_h

#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#include "newimage/newimage.h"
#include "miscmaths/SpMat.h"

namespace TOPUP {

class DispVecException: public std::exception
{
private:
  std::string m_msg;
public:
  DispVecException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("DispVec::" + m_msg).c_str();
  }

  ~DispVecException() throw() {}
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class CircularArray:
// Helper class that facilitates implementation of wrap-around.
// Interface includes:
// operator[i]: Allows you to access it as a circular array, so e.g.
//              ca[-1] will return the v[n-1] and ca[n] would return
//              v[0].
// IndexInRange(i): Translates the index i into the index of the
//                  actual element that it would access. So e.g.
//                  ca.IndexInRange(-1) would return n-1 and
//                  ca.IndexInRange(n) returns 0.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class CircularArray
{
public:
  CircularArray() : _n(0), _v(0), _sf(1.0) {}
  CircularArray(unsigned int n) : _n(n), _v(new double[_n]), _sf(1.0) { for (unsigned int i=0; i<_n; i++) _v[i]=0.0; }
  CircularArray(const std::vector<float>& v) : _n(v.size()), _v(new double[_n]), _sf(1.0) { for (unsigned int i=0; i<_n; i++) _v[i]=static_cast<double>(v[i]); }
  CircularArray(const std::vector<double>& v) : _n(v.size()), _v(new double[_n]), _sf(1.0) { for (unsigned int i=0; i<_n; i++) _v[i]=v[i]; }
  CircularArray(const NEWMAT::ColumnVector& v) : _n(v.Nrows()), _v(new double[_n]), _sf(1.0) { for (unsigned int i=0; i<_n; i++) _v[i]=static_cast<double>(v(i+1)); }
  ~CircularArray() { delete [] _v; }
  void Set(const std::vector<float>& v) { if (_n!=v.size()) { if (_v) delete [] _v; _n=v.size(); _v=new double[_n]; } for (unsigned int i=0; i<_n; i++) _v[i]=double(v[i]); }
  void Set(const std::vector<double>& v) { if (_n!=v.size()) { if (_v) delete [] _v; _n=v.size(); _v=new double[_n]; } for (unsigned int i=0; i<_n; i++) _v[i]=v[i]; }
  void Set(const NEWMAT::ColumnVector& v) { if (int(_n)!=v.Nrows()) { if (_v) delete [] _v; _n=v.Nrows(); _v=new double[_n]; } for (unsigned int i=0; i<_n; i++) _v[i]=double(v(i+1)); }
  void SetFromRow(const NEWIMAGE::volume<float>& ima, unsigned int k, unsigned int j) { set_from_row_or_col(ima,k,j,true); }
  void SetFromColumn(const NEWIMAGE::volume<float>& ima, unsigned int k, unsigned int i) { set_from_row_or_col(ima,k,i,false); }
  void Print(const std::string& fname) const;
  unsigned int N() const { return(_n); }
  void SetScaleFactor(double sf) const { _sf=sf; }
  double GetScaleFactor() const { return(_sf); }
  int Find(double x) const;
  double operator[](int i) const { int j=i%_n; if (j<0) return(i+_sf*_v[_n+j]); else return(i+_sf*_v[j]); }
  double Inv(double x) const;
  unsigned int IndexInRange(int i) const { int j=i%static_cast<int>(_n); if (j<0) return(static_cast<unsigned int>(_n+j)); else return(static_cast<unsigned int>(j)); }
private:
  unsigned int    _n;
  double          *_v;
  mutable double  _sf;

  void set_from_row_or_col(const NEWIMAGE::volume<float>& ima, int k, int ij, bool row);
};

//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//
// Class DispVec:
// Holds one row/column of a displacement field. The interface
// includes:
// SetFromRow: Picks values from a row in a volume.
// SetFromColumn: Picks values from a column in a volume.
// GetK_Matrix: Returns a corresponding K-matrix such that
//              y=K*x gives a distorted vector y from a "true"
//              vector x. For this to work the values in it
//              should be scaled to units of voxels. If this is
//              not the case (let's say it is in units of Hz)
//              one can use the form K=v.GetK_Matrix(scale_fac)
//              where in this case scale_fac would be readout
//              time in seconds. It can also be negative to
//              facilitate its use for top-down-bottom-up
//              calculations.
//
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class DispVec
{
public:
  DispVec() {}
  DispVec(unsigned int n) : _ca(n) {}
  DispVec(const std::vector<float>& v) : _ca(v) {}
  DispVec(const std::vector<double>& v) : _ca(v) {}
  DispVec(const NEWMAT::ColumnVector& v) : _ca(v) {}
  ~DispVec() {}
  void Set(const std::vector<float>& v) { _ca.Set(v); }
  void Set(const std::vector<double>& v) { _ca.Set(v); }
  void Set(const NEWMAT::ColumnVector& v) { _ca.Set(v); }
  void SetScaleFactor(double sf) { _ca.SetScaleFactor(sf); }
  double GetScaleFactor() const { return(_ca.GetScaleFactor()); }
  void SetFromRow(const NEWIMAGE::volume<float>& ima, int k, int j) { _ca.SetFromRow(ima,k,j); }
  void SetFromColumn(const NEWIMAGE::volume<float>& ima, int k, int i) { _ca.SetFromColumn(ima,k,i); }
  bool RowIsAlright(const NEWIMAGE::volume<float>& mask, int slice, int row) const;
  bool ColumnIsAlright(const NEWIMAGE::volume<float>& mask, int slice, int col) const;
  void Print(const std::string fname=std::string("")) const { _ca.Print(fname); }
  double operator[](int i) const { return(_ca[i]); }
  double Inv(int i) const { return(_ca.Inv(static_cast<double>(i))); }
  NEWMAT::ReturnMatrix GetDisplacements() const;
  NEWMAT::ReturnMatrix GetInverseDisplacements(double sf=1.0) const;
  NEWMAT::ReturnMatrix GetK_Matrix(double sf=1.0) const;
  NEWMAT::ReturnMatrix GetS_Matrix(bool wrap=true) const;
  MISCMATHS::SpMat<double> GetSparseK_Matrix(double sf=1.0) const { NEWMAT::Matrix K=GetK_Matrix(sf); return(MISCMATHS::SpMat<double>(K)); } // Should be re-written for efficiency

private:
  CircularArray   _ca;

  unsigned int get_non_zero_entries_of_row(unsigned int i, unsigned int *indx, double *val) const;
};


} // End of namespace TOPUP
#endif
