/////////////////////////////////////////////////////////////////////
///
/// \file GpuPredictorChunk.cu
/// \brief Definition of helper class for efficient prediction making on the Gpu.
///
/// \author Jesper Andersson
/// \version 1.0b, March, 2013.
/// \Copyright (C) 2013 University of Oxford 
///
/////////////////////////////////////////////////////////////////////

// Because of a bug in cuda_fp16.hpp, that gets included by cublas_v2.h, it has to
// be included before any include files that set up anything related to the std-lib.
// If not, there will be an ambiguity in cuda_fp16.hpp about wether to use the 
// old-style C isinf or the new (since C++11) std::isinf.
#include "cublas_v2.h"

#include <cstdlib>
#include <string>
#include <vector>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "newimage/newimage.h"
#pragma pop
#include "EddyHelperClasses.h"
#include "EddyCudaHelperFunctions.h"
#include "cuda/GpuPredictorChunk.h"

namespace EDDY {

GpuPredictorChunk::GpuPredictorChunk(unsigned int ntot, const NEWIMAGE::volume<float>& ima) EddyTry : _ntot(ntot)
{
  // Find total global memory on "our" device
  int dev;
  cudaError_t err = cudaGetDevice(&dev);
  if (err != cudaSuccess) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: Unable to get device: cudaGetDevice returned an error: " + EddyCudaHelperFunctions::cudaError2String(err));
  struct cudaDeviceProp prop;
  err = cudaGetDeviceProperties(&prop,dev);
  if (err != cudaSuccess) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: Unable to get device properties: cudaGetDeviceProperties reurned an error: " + EddyCudaHelperFunctions::cudaError2String(err));
  // Check how much of that we can get
  float *skrutt = NULL;
  size_t memsz;
  for (memsz = 0.5 * prop.totalGlobalMem; memsz > my_sizeof(ima); memsz *= 0.9) {
    // printf("Testing memsize: %d kB\n",memsz/1024);
    if (cudaMalloc(&skrutt,memsz) == cudaSuccess) break;
  }
  memsz *= 0.9; // Reduce a little further to accomodate changes in GPU use by other processes.
  if (memsz < my_sizeof(ima)) throw EddyException("GpuPredictorChunk::GpuPredictorChunk: Not enough memory on device");
  cudaFree(skrutt);
  // Calculate chunk-size and make vector of indicies
  _chsz = my_min(_ntot, static_cast<unsigned int>(memsz / my_sizeof(ima)));
  
  // printf("Total mem: %d kB\n",prop.totalGlobalMem/1024);
  // printf("ima size: %d kB\n",my_sizeof(ima)/1024);
  // printf("Chunk size: %d\n",_chsz);
  
  _ind.resize(_chsz);
  for (unsigned int i=0; i<_chsz; i++) _ind[i] = i;
} EddyCatch

GpuPredictorChunk& GpuPredictorChunk::operator++() EddyTry // Prefix ++
{
  if (_ind.back() == (_ntot-1)) { // If we're at the end;
    _ind.resize(1);
    _ind[0] = _ntot;
  } 
  else {
    unsigned int first = _ind.back() + 1;
    unsigned int last = first + _chsz - 1;
    last = (last >= _ntot) ? _ntot-1 : last;
    _ind.resize(last-first+1);
    for (unsigned int i=0; i<_ind.size(); i++) {
      _ind[i] = first + i;
    }
  }
  return(*this);
} EddyCatch

} // End namespace EDDY
