/**
 * Elementwise plus operator.
 */
#ifndef __OPERATOR_PLUS_HPP__
#define __OPERATOR_PLUS_HPP__


namespace armawrap {

  // Object + Object (same type)
  template<typename T1, typename T2>
  inline 
  typename enable_if<
    either_are_armawrap_type<T1, T2>::value && 
  arma::is_same_type<
    typename T1::elem_type, 
    typename T2::elem_type>::value,
    AWEGlue<
      typename armawrap_type_map<T1>::type,
      typename armawrap_type_map<T2>::type,
      arma::eglue_plus> >::type
    operator+(const T1 &X, const T2 &Y) {
    return AWEGlue<typename armawrap_type_map<T1>::type,
                   typename armawrap_type_map<T2>::type,
                   arma::eglue_plus>(X, Y);
  }

  // Object + Scalar
  template<typename T>
  inline
  typename enable_if<
    is_armawrap_type<T>::value,
    AWEOp<
      typename armawrap_type_map<T>::type, 
      arma::eop_scalar_plus> >::type
  operator+(const T &X, const typename T::elem_type k) {
    return AWEOp<typename armawrap_type_map<T>::type, 
                 arma::eop_scalar_plus>(X, k);
  }

  // Scalar + Object
  template<typename T>
  inline
  typename enable_if<
    is_armawrap_type<T>::value,
    AWEOp<
      typename armawrap_type_map<T>::type, 
      arma::eop_scalar_plus> >::type
  operator+(const typename T::elem_type k, const T &X) {
    return AWEOp<typename armawrap_type_map<T>::type, 
                 arma::eop_scalar_plus>(X, k);
  }
}

#endif
