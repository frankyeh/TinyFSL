/*  General call feature for templated image classes

    Mark Jenkinson, FMRIB Image Analysis Group

    Copyright (C) 2000 University of Oxford  */

/*  CCOPYRIGHT  */

#if !defined(__fmribmain_h)
#define __fmribmain_h

#include <iostream>
#include "NewNifti/NewNifti.h"
#include "newimage/newimageio.h"

template <class T>
int fmrib_main(int argc, char* argv[]);

int call_fmrib_main(short datatype, int argc, char* argv[])
{
  datatype=NEWIMAGE::closestTemplatedType(datatype);
  if ( datatype==NiftiIO::DT_UNSIGNED_CHAR ) return fmrib_main<char>(argc, argv);
  else if ( datatype==NiftiIO::DT_SIGNED_SHORT ) return fmrib_main<short>(argc, argv);
  else if ( datatype==NiftiIO::DT_SIGNED_INT ) return fmrib_main<int>(argc, argv);
  else if ( datatype==NiftiIO::DT_FLOAT )  return fmrib_main<float>(argc, argv);
  else if ( datatype==NiftiIO::DT_DOUBLE ) return fmrib_main<double>(argc, argv);
  return -1;
}

#endif
