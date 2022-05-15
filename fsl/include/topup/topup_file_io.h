// Declarations of classes used to write and
// read files written by topup, and potentially
// by other pieces of software as long as they
// are valid magnetic-field files.
//
// topup_file_io.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2009 University of Oxford
/*  CCOPYRIGHT  */

#ifndef topup_file_io_h
#define topup_file_io_h

#include <string>
#include <vector>
#include <memory>
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimage.h"
#include "basisfield/basisfield.h"
#include "basisfield/splinefield.h"
#include "basisfield/dctfield.h"
#include "topup_matrices.h"

namespace TOPUP {

class TopupFileIOException: public std::exception
{
private:
  std::string m_msg;
public:
  TopupFileIOException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("TopupFileIO:: msg=" + m_msg).c_str();
  }

  ~TopupFileIOException() throw() {}
};

//
// Non-standard nifti intent-codes used for FSL-derived
// displacement/coefficient files.
//
#define FSL_TOPUP_CUBIC_SPLINE_COEFFICIENTS      2016
#define FSL_TOPUP_QUADRATIC_SPLINE_COEFFICIENTS  2017
#define FSL_TOPUP_FIELD                          2018

enum FieldFileType {SplineField, FieldField, UnknownField, InvalidField};
enum TopupAbsOrRelWarps {RelativeWarps, AbsoluteWarps, UnknownWarps};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// The TopupFileWrite is a class that writes off-resonance magnetic-field coefficient-files
// and movement parameter files produced by topup.
//
///////////////////////////////////////////////////////////////////////////////////////////////

class TopupFileWriter
{
public:
  // Constructor for coefficient file
  TopupFileWriter(const std::string&                 fname,
                  const BASISFIELD::splinefield&     field);
  // Constructor for field file
  TopupFileWriter(const std::string&                   fname,
                  const NEWIMAGE::volume<float>&       ref,
                  const BASISFIELD::splinefield&       field);
  // Constructor for movement parameter file
  TopupFileWriter(const std::string&       fname,
                  const NEWMAT::Matrix&    mp);
  // Constructor for rigid body matrix files
  TopupFileWriter(const std::string&                    fname,
                  const std::vector<NEWMAT::Matrix>&    mp);
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// The TopupFileReader is a class that reads, checks and decodes an off-resonance
// magnetic-field coefficient file.
// This can be a file created by topup, but it could also have been created for example
// from a fieldmap. For files created by other applications than topup it only reads
// "proper" displacement fields. For topup-files it also reads coefficient-files.
//
// The ultimate purpose of TopupFileReader and TopupFileWriter is to provide an
// interface that makes it easy and transparent to read and write off-resonance
// field files.
//
///////////////////////////////////////////////////////////////////////////////////////////////

class TopupFileReader
{
public:
  // Default constructor
  TopupFileReader() : _mp_valid(false) {}
  // Constructor for coeffcient-field and movement parameters
  TopupFileReader(const std::string& fname);
  // Destructor
  ~TopupFileReader() {}

  void ReadField(const std::string& fname) { common_read(fname); }
  void ReadMovements(const std::string& fname);
  NEWIMAGE::volume<float> FieldAsVolume() const { ensure_volume(); return(*_vol_rep); }
  NEWIMAGE::volume<float> FieldAsVolume(const NEWMAT::Matrix& M) const;
  std::shared_ptr<NEWIMAGE::volume<float> > FieldAsVolumePtr() const { ensure_volume(); return(_vol_rep); }
  BASISFIELD::splinefield Field() const { ensure_field(); return(*_coef_rep); }
  std::shared_ptr<BASISFIELD::splinefield> FieldPtr() const { ensure_field(); return(_coef_rep); }
  NEWMAT::Matrix MoveMatrix(unsigned int i) const { if (i<1 || i>_move.size()) throw TopupFileIOException("TopupFileReader::MoveMatrix: Index i out of bound"); return(_move[i-1]); }
  NEWMAT::ColumnVector MovePar(unsigned int i) const { if (i<1 || i>_mp.size()) { throw TopupFileIOException("TopupFileReader::MovePar: Index i out of bound"); } return(_mp[i-1]); }
  FieldFileType FieldType() const { return(_ft); }
  bool FieldIsValid() const { return(_ft != InvalidField); }
  bool MoveParValid() const { return(_mp_valid); }
  unsigned int NoOfMovPars() const { return(_mp.size()); }

private:
  FieldFileType                                        _ft;
  bool                                                 _mp_valid;
  std::vector<NEWMAT::ColumnVector>                    _mp;
  std::vector<NEWMAT::Matrix>                          _move;
  mutable std::shared_ptr<BASISFIELD::splinefield>   _coef_rep;
  mutable std::shared_ptr<NEWIMAGE::volume<float> >  _vol_rep;

  void common_read(const std::string& fname);
  std::shared_ptr<BASISFIELD::splinefield>  read_coef_file(const NEWIMAGE::volume<float>& vcoef);
  void read_movement(const std::string& fname, const NEWIMAGE::volume<float>& vol);
  void ensure_volume() const;
  void ensure_field() const;

};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// The TopupDatafileReader is a class that reads text files containing phase-encode
// vectors and readout times.
//
///////////////////////////////////////////////////////////////////////////////////////////////

class TopupDatafileReader
{
public:
  // Only one constructor
  TopupDatafileReader(const std::string& fname)
  {
    NEWMAT::Matrix tmp = MISCMATHS::read_ascii_matrix(fname);
    if (tmp.Ncols() != 4) {
      if (tmp.Nrows() == 4) tmp = tmp.t();
      else throw TopupFileIOException(std::string("TopupDatafileReader:: error reading file ")+fname);
    }
    _pv.ReSize(3,tmp.Nrows());
    _rt.ReSize(tmp.Nrows());
    for (int i=0; i<tmp.Nrows(); i++) {
      _pv.Column(i+1) = (tmp.SubMatrix(i+1,i+1,1,3)).t();
      if (fabs(1.0 - (_pv.Column(i+1)).NormFrobenius()) > 0.01) throw TopupFileIOException("TopupDatafileReader:: phase-encode vectors must be unity length");
      _rt(i+1) = tmp(i+1,4);
    }
  }

  // Destructor
  ~TopupDatafileReader() {}

  unsigned int          N() const { return(_pv.Ncols()); }
  NEWMAT::Matrix        PhaseEncodeVectors() const { return(_pv); }
  NEWMAT::ColumnVector  PhaseEncodeVector(unsigned int i) const { if (i<1 || i>N()) throw TopupFileIOException("TopupDatafileReader::PhaseEncodeVector: Invalid index"); else return(_pv.Column(i)); }
  NEWMAT::ColumnVector  ReadOutTimes() const { return(_rt); }
  double                ReadOutTime(unsigned int i) const { if (i<1 || i>N()) throw TopupFileIOException("TopupDatafileReader::ReadOutTime: Invalid index"); else return(static_cast<double>(_rt(i))); }
private:
  NEWMAT::Matrix        _pv;
  NEWMAT::ColumnVector  _rt;
};

} // End namespace TOPUP

#endif // end #ifndef topup_file_io_h
