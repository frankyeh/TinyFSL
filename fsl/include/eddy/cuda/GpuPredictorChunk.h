/////////////////////////////////////////////////////////////////////
///
/// \file GpuPredictorChunk.h
/// \brief Declaration of helper class for efficient prediction making on the Gpu.
///
/// \author Jesper Andersson
/// \version 1.0b, March, 2013.
/// \Copyright (C) 2013 University of Oxford
///
/////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <string>
#include <vector>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "newimage/newimage.h"
#pragma pop

#ifndef GpuPredictorChunk_h
#define GpuPredictorChunk_h

namespace EDDY {

class GpuPredictorChunk
{
public:
  GpuPredictorChunk(unsigned int ntot, const NEWIMAGE::volume<float>& ima);
  ~GpuPredictorChunk() {}
  std::vector<unsigned int> Indicies() const EddyTry { return(_ind); } EddyCatch
  GpuPredictorChunk& operator++();       // Prefix
  GpuPredictorChunk operator++(int) EddyTry { GpuPredictorChunk tmp(*this); operator++(); return(tmp); } EddyCatch
  bool operator< (unsigned int rhs) const EddyTry { return(_ind.back() < rhs); } EddyCatch
  friend std::ostream& operator<<(std::ostream& out, const GpuPredictorChunk& pc) EddyTry {
    for (unsigned int i=0; i<pc._ind.size(); i++) out << pc._ind[i] << " ";
    return(out);
  } EddyCatch
private:
  unsigned int              _ntot;
  unsigned int              _chsz;
  std::vector<unsigned int> _ind;

  size_t my_sizeof(const NEWIMAGE::volume<float>& ima) const EddyTry {
    return(sizeof(ima) + ima.xsize()*ima.ysize()*ima.zsize()*sizeof(float));
  } EddyCatch
  template<typename T>
  T my_min(const T& lhs, const T& rhs) const { return((lhs < rhs) ? lhs : rhs); }
};

} // End namespace EDDY

#endif // #ifndef GpuPredictorChunk_h
