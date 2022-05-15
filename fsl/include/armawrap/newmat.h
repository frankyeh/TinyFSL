#ifndef __NEWMAT_H__
#define __NEWMAT_H__

/*
 * A newmat-like interface to the armawrap library.
 */

#include "armawrap.hpp"

/*
 * Sigh. Newran (a companion library to newmat, providing random number
 * generatiion) defines all of the same exception types as Newmat.  So I'm
 * using/defining the same include header guard. There will probably be
 * side effects
 */
#ifndef EXCEPTION_LIB
#define EXCEPTION_LIB
namespace RBD_COMMON {
  // Exception types
  typedef std::runtime_error BaseException;
  typedef std::runtime_error Exception;
  typedef std::runtime_error Logic_error;
  typedef std::runtime_error ProgramException;
  typedef std::runtime_error IndexException;
  typedef std::runtime_error VectorException;
  typedef std::runtime_error NotSquareException;
  typedef std::runtime_error SubMatrixDimensionException;
  typedef std::runtime_error IncompatibleDimensionsException;
  typedef std::runtime_error NotDefinedException;
  typedef std::runtime_error CannotBuildException;
  typedef std::runtime_error InternalException;
  typedef std::runtime_error Runtime_error;
  typedef std::runtime_error NPDException;
  typedef std::runtime_error ConvergenceException;
  typedef std::runtime_error SingularException;
  typedef std::runtime_error SolutionException;
  typedef std::runtime_error OverflowException;
  typedef std::runtime_error Bad_alloc;
  typedef std::string        Tracer;
  typedef double             Real;
}
#endif

namespace NEWMAT {

  using namespace RBD_COMMON;

  // Data type
  // Matrix/vector types
  typedef armawrap::AWMatrix<               Real> Matrix;
  typedef armawrap::AWRowVector<            Real> RowVector;
  typedef armawrap::AWColVector<            Real> ColumnVector;
  typedef armawrap::AWIdentityMatrix<       Real> IdentityMatrix;
  typedef armawrap::AWDiagonalMatrix<       Real> DiagonalMatrix;
  typedef armawrap::AWSymmetricMatrix<      Real> SymmetricMatrix;
  typedef armawrap::AWUpperTriangularMatrix<Real> UpperTriangularMatrix;
  typedef armawrap::AWLowerTriangularMatrix<Real> LowerTriangularMatrix;
  typedef armawrap::AWBandMatrix<           Real> BandMatrix;
  typedef armawrap::AWUpperBandMatrix<      Real> UpperBandMatrix;
  typedef armawrap::AWLowerBandMatrix<      Real> LowerBandMatrix;
  typedef armawrap::AWSymmetricBandMatrix<  Real> SymmetricBandMatrix;
  typedef armawrap::AWCroutMatrix<          Real> CroutMatrix;

  typedef armawrap::AWLogAndSign<           Real> LogAndSign;
  typedef armawrap::AWMatrixBandWidth             MatrixBandWidth;

  // Temporary-but-possibly-permanent hacks
  typedef Matrix GeneralMatrix;
  typedef Matrix ReturnMatrix;


  using armawrap::SP;
  using armawrap::KP;
  using armawrap::Cholesky;
  using armawrap::QRZ;
  using armawrap::DotProduct;
  using armawrap::EigenValues;
  using armawrap::Jacobi;
  using armawrap::SortAscending;
  using armawrap::SortDescending;
  using armawrap::FFT;
  using armawrap::FFTI;
  using armawrap::RealFFTI;
  using armawrap::RealFFTI;
  using armawrap::SVD;

  using armawrap::Minimum;
  using armawrap::Maximum;
  using armawrap::MinimumAbsoluteValue;
  using armawrap::MaximumAbsoluteValue;
  using armawrap::SumSquare;
  using armawrap::SumAbsoluteValue;
  using armawrap::Sum;
  using armawrap::Norm1;
  using armawrap::NormInfinity;
  using armawrap::NormFrobenius;
  using armawrap::Trace;
  using armawrap::Determinant;
  using armawrap::LogDeterminant;
  using armawrap::IsZero;
}


namespace RBD_LIBRARIES {
  using namespace NEWMAT;
}

#endif /* __NEWMAT_H__ */
