//
//  Definitions for displacement vector class DispVec
//
//  displacement_vector.cpp
//
//  Implements a displacement vector class that can be
//  used to obtain inverses, K-matrices etc.
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2010 University of Oxford
//
/*  CCOPYRIGHT  */

#include <string>
#include <vector>
#include <cmath>
#include "armawrap/newmat.h"
#include "miscmaths/SpMat.h"
#include "displacement_vector.h"

using namespace std;

namespace TOPUP {

bool DispVec::RowIsAlright(const NEWIMAGE::volume<float>& mask, int sl, int row) const
{
  for (int i=0; i<mask.xsize(); i++) if (!mask(i,row,sl)) return(false);
  return(true);
}

bool DispVec::ColumnIsAlright(const NEWIMAGE::volume<float>& mask, int sl, int col) const
{
  for (int i=0; i<mask.ysize(); i++) if (!mask(col,i,sl)) return(false);
  return(true);
}

NEWMAT::ReturnMatrix DispVec::GetDisplacements() const
{
  NEWMAT::ColumnVector  v(_ca.N());
  for (unsigned int i=0; i<_ca.N(); i++) v(i+1) = _ca[i]-i;
  v.Release();

  return(v);
}

NEWMAT::ReturnMatrix DispVec::GetInverseDisplacements(double sf) const
{
  NEWMAT::ColumnVector v(_ca.N());
  double old_sf = _ca.GetScaleFactor();
  _ca.SetScaleFactor(sf);
  for (unsigned int i=0; i<_ca.N(); i++) v(i+1) = _ca.Inv(double(i)) - double(i);
  _ca.SetScaleFactor(old_sf);

  v.Release();
  return(v);
}

NEWMAT::ReturnMatrix DispVec::GetK_Matrix(double sf) const
{
  unsigned int    *indx = new unsigned int[_ca.N()];
  double          *val = new double[_ca.N()];
  NEWMAT::Matrix  K(_ca.N(),_ca.N());
  K = 0.0;
  double old_sf = _ca.GetScaleFactor();
  _ca.SetScaleFactor(sf);
  for (unsigned int i=0; i<_ca.N(); i++) {
    unsigned int nnz = get_non_zero_entries_of_row(i,indx,val);
    for (unsigned int j=0; j<nnz; j++) K(i+1,indx[j]+1) = val[j];
  }
  _ca.SetScaleFactor(old_sf);

  K.Release();
  return(K);
}

NEWMAT::ReturnMatrix DispVec::GetS_Matrix(bool wrap) const
{
  NEWMAT::Matrix S(_ca.N(),_ca.N());
  S = 0.0;
  if (wrap) { S(1,1)=2; S(1,2)=-1; S(1,_ca.N())=-1; }
  else { S(1,1)=1; S(1,2)=-1; }
  for (unsigned int i=2; i<_ca.N(); i++) { S(i,i-1)=-1; S(i,i)=2; S(i,i+1)=-1; }
  if (wrap) { S(_ca.N(),1)=-1; S(_ca.N(),_ca.N()-1)=-1; S(_ca.N(),_ca.N())=2; }
  else { S(_ca.N(),_ca.N()-1)=-1; S(_ca.N(),_ca.N())=1; }

  S.Release();
  return(S);
}

//
// Calculates the non-zero entries of the i'th row of the
// K matrix corresponding to the displacements in ca.
// It returns the # of non-zero elements, the indicies
// of those elements in indx and the values in val.
// It is the responsibility of the caller to make sure
// that indx and val are large enough.
// Both i and indx assumes zero-offset, i.e. C-style.
//
unsigned int DispVec::get_non_zero_entries_of_row(// Input
                                                  unsigned int           i,       // Row index, zero offset
                                                  // Output
                                                  unsigned int           *indx,   // Indicies of non-zero entries, zero offset
                                                  double                 *val)    // Values of non-zero indicies
const
{
  int i1 = _ca.Find(i-0.5);                           // Index before i-0.5
  int i2 = _ca.Find(i+0.5);                           // Index before i+0.5
  double x1 = (i-0.5-_ca[i1]) / (_ca[i1+1]-_ca[i1]);  // Distance of 1st crossing from i1
  double x2 = (i+0.5-_ca[i2]) / (_ca[i2+1]-_ca[i2]);  // Distance of 2nd crossing from i2

  if (i1==i2) {  // If both squeezed between one index and next
    if (x1 < 0.5) {
      if (x2 < 0.5) {
	indx[0] = _ca.IndexInRange(i1);
	val[0] = x2-x1;
        return(1);
      }
      else {
	indx[0] = _ca.IndexInRange(i1);
        val[0] = 0.5-x1;
        indx[1] = _ca.IndexInRange(i1+1);
        val[1] = x2-0.5;
        return(2);
      }
    }
    else {
      indx[0] = _ca.IndexInRange(i1+1);
      val[0] = x2-x1;
      return(1);
    }
  }
  else if (i2-i1==1) { // If consequtive indicies
    if (x1 < 0.5) {
      indx[0] = _ca.IndexInRange(i1);
      val[0] = 0.5-x1;
      if (x2 < 0.5) {
        indx[1] = _ca.IndexInRange(i1+1);
        val[1] = 0.5+x2;
        return(2);
      }
      else {
        indx[1] = _ca.IndexInRange(i1+1);
        val[1] = 1.0;
        indx[2] = _ca.IndexInRange(i1+2);
        val[2] = x2-0.5;
        return(3);
      }
    }
    else { // if (xl > 0.5)
      if (x2 < 0.5) {
        indx[0] = _ca.IndexInRange(i1+1);
        val[0] = 1+x2-x1;
        return(1);
      }
      else {
        indx[0] = _ca.IndexInRange(i1+1);
        val[0] = 1.5-x1;
	indx[1] = _ca.IndexInRange(i1+2);
	val[1] = x2-0.5;
        return(2);
      }
    }
  }
  else if (i2-i1>1) {
    unsigned int n=0;
    if (x1 < 0.5) {
      indx[n] = _ca.IndexInRange(i1);
      val[n++] = 0.5-x1;
      indx[n] = _ca.IndexInRange(i1+1);
      val[n++] = 1.0;
    }
    else {
      indx[n] = _ca.IndexInRange(i1+1);
      val[n++] = 1.5-x1;
    }
    for (int i=i1+2; i<i2; i++) {
      indx[n] = _ca.IndexInRange(i);
      val[n++] = 1.0;
    }
    if (x2 < 0.5) {
      indx[n] = _ca.IndexInRange(i2);
      val[n++] = 0.5+x2;
    }
    else {
      indx[n] = _ca.IndexInRange(i2);
      val[n++] = 1.0;
      indx[n] = _ca.IndexInRange(i2+1);
      val[n++] = x2-0.5;
    }
    return(n);
  }
  else throw DispVecException("get_non_zero_entries_of_row: This doesn't make sense.");

}

//
// Returns the index i of the coordinate where i+d(i)<x and i+1+d(i+1)>x
//
int CircularArray::Find(double x) const
{
  int i=static_cast<int>(ceil(x));
  if (x > (*this)[i]) { i++; while (x > (*this)[i]) i++; return(i-1); }
  else if (x == (*this)[i]) return(i);
  else {
    while (x < (*this)[i-1]) i--;
    return(i-1);
  }
}

//
// The DispVec class is used to implement a mapping x->x' where x'=x_i+d(x_i)
// where x_i is an integer and x' is not. Let us now say we have a value
// x'_i (i.e. an integer location in the x' space) and we want to know the value
// x that would give x'_i=x+d(x). This value (x) is what is returned by
// inv given an input x'_i (denoted x in the code).
//
double CircularArray::Inv(double x) const
{
  int i = Find(static_cast<double>(x));
  double d = (x-(*this)[i]) / ((*this)[i+1]-(*this)[i]);  // Distance of crossing from i
  return(i+d);
}

void CircularArray::set_from_row_or_col(const NEWIMAGE::volume<float>& ima, int k, int ij, bool row)
{
  unsigned int offs = (row) ? k*ima.xsize()*ima.ysize()+ij*ima.xsize(): k*ima.xsize()*ima.ysize()+ij;
  unsigned int ss = (row) ? 1 : ima.xsize();
  unsigned int n = (row) ? ima.xsize() : ima.ysize();

  if (n != _n) {
    if (_n) delete [] _v;
    _n = n;
    _v = new double[_n];
  }

  NEWIMAGE::volume<float>::fast_const_iterator it=ima.fbegin();
  it += offs;
  for (unsigned int i=0; i<_n; i++, it+=ss) {
    _v[i] = static_cast<double>(*it);
  }
}

void CircularArray::Print(const std::string& fname) const
{
  if (fname.size()) { // If a filename is given
    ofstream fs(fname.c_str());
    fs.setf(ios::scientific | ios::showpos);
    fs.precision(6);
    for (unsigned int i=0; i<_n; i++) {
      fs << _sf * _v[i] << endl;
    }
    fs.close();
  }
  else {
    for (unsigned int i=0; i<_n; i++) {
      cout << _sf * _v[i] << endl;
    }
  }
}

} // End of namespace TOPUP
