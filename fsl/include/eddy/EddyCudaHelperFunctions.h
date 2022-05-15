/////////////////////////////////////////////////////////////////////
///
/// \file EddyCudaHelperFunctions.h
/// \brief Declarations of some low level helper functions for eddy 
/// CUDA code.
///
/// \author Jesper Andersson
/// \version 1.0b, Nov., 2020.
/// \Copyright (C) 2020 University of Oxford 
///
/////////////////////////////////////////////////////////////////////

#ifndef EddyCudaHelperFunctions_h
#define EddyCudaHelperFunctions_h

#include "cublas_v2.h"
#include <string>
#include <sstream>
#include <cuda.h>

namespace EDDY {

/////////////////////////////////////////////////////////////////////
///
/// \brief This class contains a set of static methods that implement
/// various CUDA utility functions for the eddy project.
///
/////////////////////////////////////////////////////////////////////
class EddyCudaHelperFunctions
{
public:
  /// Does a little song and dance to initialize GPU
  static void InitGpu(bool verbose=true);

 /// Returns a formatted string with info about a cudaError_t code
  static std::string cudaError2String(const cudaError_t& ce);

  /// Waits for GPU to finish and checks error status
  static void CudaSync(const std::string& msg);

  /// Returns name of error associated with cs
  static std::string cuBLASGetErrorName(const cublasStatus_t& cs);  

  /// Returns explanatory string for error associated with cs
  static std::string cuBLASGetErrorString(const cublasStatus_t& cs);  

  /// Returns a formatted string with info about a cublasStatus_t code
  static std::string cublasError2String(const cublasStatus_t& ce);  
};

} // End namespace EDDY

#endif // End #ifndef EddyCudaHelperFunctions_h
