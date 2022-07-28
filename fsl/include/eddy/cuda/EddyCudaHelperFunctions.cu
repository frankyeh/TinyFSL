#include "cublas_v2.h"

#include <cstdlib>
#include <string>
#include <cuda.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#include "EddyHelperClasses.h"
#pragma pop
#include "EddyCudaHelperFunctions.h"

namespace EDDY {

std::string EddyCudaHelperFunctions::cudaError2String(const cudaError_t& ce) EddyTry
{
  std::ostringstream oss;
  oss << "cudaError_t = " << ce << ", cudaErrorName = " << cudaGetErrorName(ce) << ", cudaErrorString = " << cudaGetErrorString(ce);
  return(oss.str());
} EddyCatch

void EddyCudaHelperFunctions::CudaSync(const std::string& msg) EddyTry
{
  /*cudaError_t err = cudaDeviceSynchronize();
  if (err!=cudaSuccess) {
    std::ostringstream os;
    os << "EddyKernels::CudaSync: CUDA error after call to " << msg << ", " << EddyCudaHelperFunctions::cudaError2String(err);
    throw EDDY::EddyException(os.str());
  }*/
} EddyCatch

void EddyCudaHelperFunctions::InitGpu(bool verbose) EddyTry
{
  static bool initialized=false;
  if (!initialized) {
    initialized=true;
    int device;
    cudaError_t ce;
    if ((ce = cudaGetDevice(&device)) != cudaSuccess) throw EddyException("EddyCudaHelperFunctions::InitGpu: cudaGetDevice returned an error: " + EddyCudaHelperFunctions::cudaError2String(ce));
    if (verbose) printf("\n...................Allocated GPU # %d...................\n", device); 
    int *q;
    if ((ce = cudaMalloc((void **)&q, sizeof(int))) != cudaSuccess) {
      throw EddyException("EddyCudaHelperFunctions::InitGpu: cudaMalloc returned an error: " + EddyCudaHelperFunctions::cudaError2String(ce));
    }
    cudaFree(q);
    EddyCudaHelperFunctions::CudaSync("EddyGpuUtils::InitGpu");
  }
} EddyCatch

std::string EddyCudaHelperFunctions::cuBLASGetErrorName(const cublasStatus_t& cs) EddyTry
{
  std::string rval;

  switch (cs) {
    case CUBLAS_STATUS_SUCCESS:
      rval = "CUBLAS_STATUS_SUCCESS";
      break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
      rval = "CUBLAS_STATUS_NOT_INITIALIZED";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED:
      rval = "CUBLAS_STATUS_ALLOC_FAILED";
      break;
    case CUBLAS_STATUS_INVALID_VALUE:
      rval = "CUBLAS_STATUS_INVALID_VALUE";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      rval = "CUBLAS_STATUS_ARCH_MISMATCH";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      rval = "CUBLAS_STATUS_MAPPING_ERROR";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      rval = "CUBLAS_STATUS_EXECUTION_FAILED";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      rval = "CUBLAS_STATUS_INTERNAL_ERROR";
      break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      rval = "CUBLAS_STATUS_NOT_SUPPORTED";
      break;
    case CUBLAS_STATUS_LICENSE_ERROR:
      rval = "CUBLAS_STATUS_LICENSE_ERROR";
      break;
    default:
      rval = "Unkown CUBLAS status code";
  }
  return(rval);
} EddyCatch

std::string EddyCudaHelperFunctions::cuBLASGetErrorString(const cublasStatus_t& cs) EddyTry
{
  std::string rval;

  switch (cs) {
    case CUBLAS_STATUS_SUCCESS:
      rval = "The operation completed successfully";
      break;
    case CUBLAS_STATUS_NOT_INITIALIZED:
      rval = "The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call, an error in the CUDA Runtime API called by the cuBLAS routine, or an error in the hardware setup.\n\nTo correct: call cublasCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED:
      rval = "Resource allocation failed inside the cuBLAS library. This is usually caused by a cudaMalloc() failure.\n\nTo correct: prior to the function call, deallocate previously allocated memory as much as possible.";
      break;
    case CUBLAS_STATUS_INVALID_VALUE:
      rval = "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n\nTo correct: ensure that all the parameters being passed have valid values.";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      rval = "The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.\n\nTo correct: compile and run the application on a device with appropriate compute capability, which is 1.3 for double precision.";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      rval = "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n\nTo correct: prior to the function call, unbind any previously bound textures.";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      rval = "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n\nTo correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed.";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      rval = "An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n\nTo correct: check that the hardware, an appropriate version of the driver, and the cuBLAS library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
      break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      rval = "The functionality requested is not supported";
      break;
    case CUBLAS_STATUS_LICENSE_ERROR:
      rval = "The functionality requested requires some license and an error was detected when trying to check the current licensing. This error can happen if the license is not present or is expired or if the environment variable NVIDIA_LICENSE_FILE is not set properly.";
      break;
    default:
      rval = "An unknown cublasStatus_t values was encountered";
  }
  return(rval);
} EddyCatch

std::string EddyCudaHelperFunctions::cublasError2String(const cublasStatus_t& ce) EddyTry
{
  std::ostringstream oss;
  oss << "cublasStatus_t = " << ce << ", cublasErrorName = " << cuBLASGetErrorName(ce) << "," << std::endl << "cublasErrorString = " << cuBLASGetErrorString(ce);
  return(oss.str());
} EddyCatch

} // End namespace EDDY
