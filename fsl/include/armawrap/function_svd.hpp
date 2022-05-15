#ifndef __FUNCTION_SVD_HPP__
#define __FUNCTION_SVD_HPP__

/*
 * Singular value decomposition.
 */

namespace armawrap {

  template<typename T1, typename T2, typename T3, typename T4>
  inline  void SVD(const T1  &A,
                   T2        &D,
                   T3        &U,
                   T4        &V,
                   bool withU=true,
                   bool withV=true) {


    arma::Col<typename T1::elem_type> s;
    arma::Mat<typename T1::elem_type> Ut;
    arma::Mat<typename T1::elem_type> At(A);

    /*
     * NEWMAT::SVD does not support decomposition on
     * matrices with more columns than rows, and
     * raises a ProgramException. In armawrap/newmat.h,
     * ProgramException is aliased to runtime_error,
     * so that's what we raise here.
     */
    if (At.n_rows < At.n_cols) {
      throw std::runtime_error("SVD requires that m >= n for a m*n matrix");
    }

    arma::svd_econ(Ut, s, V, At);

    if (!withV) V = 0;
    if ( withU) U = Ut;
    D = s;
  }

  template<typename T1, typename T2>
  inline void SVD(const T1 &A, T2 &D) {

    AWMatrix<typename T1::elem_type> U;
    AWMatrix<typename T1::elem_type> V;

    SVD(A, D, U, V, false, false);
  }

  template<typename T1, typename T2, typename T3>
  inline void SVD(const T1 &A,
                  T2       &D,
                  T3       &U,
                  bool withU=true) {

    AWMatrix<typename T1::elem_type> V(D.Nrows(), D.Ncols());
    SVD(A, D, U, V, withU, false);
  }
}

#endif /* __FUNCTION_SVD_HPP__ */
