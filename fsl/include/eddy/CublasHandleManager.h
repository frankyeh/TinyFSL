/*! \file CublasHandleManager.h
    \brief Contains declaration and definition of a single class used to manage a cuBlas handle.

    \author Jesper Andersson
    \version 1.0b, Sep., 2012.
*/
// Declarations of classes that implements useful
// concepts for the eddy current project.
//
// CublasHandleManager.h
//
// Jesper Andersson, FMRIB Image Analysis Group
//
// Copyright (C) 2020 University of Oxford
//

// Because of a bug in cuda_fp16.hpp, that gets included by cublas_v2.h, it has to
// be included before any include files that set up anything related to the std-lib.
// If not, there will be an ambiguity in cuda_fp16.hpp about wether to use the
// old-style C isinf or the new (since C++11) std::isinf.

#if !defined(CUBLAS_V2_H_)
#error cublas_v2.h must be included at the very top of any file including CublasHandleManager.h
#endif

#include <stdio.h>
#pragma push
#pragma diag_suppress = code_is_unreachable // Supress warnings from armawrap
#pragma diag_suppress = expr_has_no_effect  // Supress warnings from boost
#include "armawrap/newmat.h"
#include "miscmaths/miscmaths.h"
#include "newimage/newimageall.h"
#pragma pop
#include "EddyHelperClasses.h"
#include <cuda.h>
#include <cuda_runtime.h>


class CublasHandleManager
{
public:
  CublasHandleManager(const CublasHandleManager&) = delete;
  CublasHandleManager& operator=(const CublasHandleManager&) = delete;
  static cublasHandle_t& GetHandle();
private:
  CublasHandleManager() {
    cublasStatus_t status;
    status = cublasCreate(&_handle);
    if (status != CUBLAS_STATUS_SUCCESS) throw EDDY::EddyException("EDDY::CublasHandleManager::CublasHandleManager: cuBLAS initialization failed");
  }
  ~CublasHandleManager() { cublasDestroy(_handle); }

  cublasHandle_t _handle;
};

cublasHandle_t& CublasHandleManager::GetHandle()
{
  static CublasHandleManager instance;
  return(instance._handle);
}
