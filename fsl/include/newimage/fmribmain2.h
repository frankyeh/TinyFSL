/*  General call feature for templated image classes

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__fmribmain2_h)
#define __fmribmain2_h

#include <iostream>

#include "NewNifti/NewNifti.h"

template <class T, class S>
int fmrib_main(int argc, char* argv[]);

template <class T>
int call_fmrib_main(short datatype, int argc, char* argv[])
{
  datatype=NEWIMAGE::closestTemplatedType(datatype);
  if ( datatype==NiftiIO::DT_UNSIGNED_CHAR ) return fmrib_main<T,char>(argc, argv);
  else if ( datatype==NiftiIO::DT_SIGNED_SHORT ) return fmrib_main<T,short>(argc, argv);
  else if ( datatype==NiftiIO::DT_SIGNED_INT ) return fmrib_main<T,int>(argc, argv);
  else if ( datatype==NiftiIO::DT_FLOAT )  return fmrib_main<T,float>(argc, argv);
  else if ( datatype==NiftiIO::DT_DOUBLE ) return fmrib_main<T,double>(argc, argv);
  return -1;
}

int call_fmrib_main(short datatype1, short datatype2, int argc, char* argv[])
{
  datatype1=NEWIMAGE::closestTemplatedType(datatype1);
  if ( datatype1==NiftiIO::DT_UNSIGNED_CHAR ) return call_fmrib_main<char>(datatype2, argc, argv);
  else if ( datatype1==NiftiIO::DT_SIGNED_SHORT ) return call_fmrib_main<short>(datatype2, argc, argv);
  else if ( datatype1==NiftiIO::DT_SIGNED_INT ) return call_fmrib_main<int>(datatype2, argc, argv);
  else if ( datatype1==NiftiIO::DT_FLOAT )  return call_fmrib_main<float>(datatype2, argc, argv);
  else if ( datatype1==NiftiIO::DT_DOUBLE ) return call_fmrib_main<double>(datatype2, argc, argv);
  return -1;
}

#endif
