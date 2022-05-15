// Declarations of two functions used to
// create transformation matrices from movement
// parameters and vice-versa.
//
// topup_matrices.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2016 University of Oxford
/*  CCOPYRIGHT  */

#ifndef topup_matrices_h
#define topup_matrices_h

#include <string>
#include <vector>
#include "armawrap/newmat.h"
#include "newimage/newimage.h"

namespace TOPUP {

class TopupMatrixException: public std::exception
{
private:
  std::string m_msg;
public:
  TopupMatrixException(const std::string& msg) throw(): m_msg(msg) {}

  virtual const char * what() const throw() {
    return std::string("TopupFileIO:: msg=" + m_msg).c_str();
  }

  ~TopupMatrixException() throw() {}
};

NEWMAT::Matrix MovePar2Matrix(const NEWMAT::ColumnVector&     mp,
                              const NEWIMAGE::volume<float>&  vol);

NEWMAT::ColumnVector Matrix2MovePar(const NEWMAT::Matrix&           M,
				    const NEWIMAGE::volume<float>&  vol);
}

#endif
