cmake_minimum_required(VERSION 3.19)
if(CUDA_ARCH)
    project(eddy VERSION "2022.01.17" LANGUAGES CUDA CXX;C)
else()
    project(eddy VERSION "2022.01.17" LANGUAGES CXX;C)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE) 
    set(BLA_VENDOR Apple)
else()
    set(BLA_VENDOR OpenBLAS)
endif()
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)
if(CUDA_ARCH)
    find_package(CUDA REQUIRED)
    set(OTHER_SOURCES
        ../fsl/include/eddy/cuda/CudaVolume.cu
        ../fsl/include/eddy/cuda/GpuPredictorChunk.cu
        ../fsl/include/eddy/cuda/DiffusionGP.cu
        ../fsl/include/eddy/cuda/EddyCudaHelperFunctions.cu
        ../fsl/include/eddy/cuda/PostEddyCF.cu
        ../fsl/include/eddy/cuda/EddyGpuUtils.cu
        ../fsl/include/eddy/cuda/LSResampler.cu
        ../fsl/include/eddy/cuda/EddyMatrixKernels.cu
        ../fsl/include/eddy/cuda/StackResampler.cu
        ../fsl/include/eddy/cuda/DerivativeCalculator.cu
        ../fsl/include/eddy/cuda/EddyKernels.cu
        ../fsl/include/eddy/cuda/EddyInternalGpuUtils.cu

        ../fsl/include/cudabasisfield/CBFKernelDefinitions.cu
        ../fsl/include/cudabasisfield/CBFSparseDiagonalMatrix.cu
        ../fsl/include/cudabasisfield/CBFSplineField.cu)
else()
    set(OTHER_SOURCES
        ../fsl/include/eddy/LSResampler.cpp
        ../fsl/include/eddy/PostEddyCF.cpp
        )
endif()

set(SOURCES
    ../fsl/include/eddy/b0Predictor.cpp
    ../fsl/include/eddy/BiasFieldEstimatorImpl.cpp
    ../fsl/include/eddy/CPUStackResampler.cpp
    ../fsl/include/eddy/DiffusionGP.cpp
    ../fsl/include/eddy/ECModels.cpp
    ../fsl/include/eddy/ECScanClasses.cpp
    ../fsl/include/eddy/eddy.cpp
    ../fsl/include/eddy/EddyCommandLineOptions.cpp
    ../fsl/include/eddy/EddyHelperClasses.cpp
    ../fsl/include/eddy/EddyUtils.cpp
    ../fsl/include/eddy/HyParEstimator.cpp
    ../fsl/include/eddy/KMatrix.cpp
    ../fsl/include/eddy/MoveBySuscCF.cpp
    ../fsl/include/eddy/PostEddyAlignShellsFunctions.cpp
    ../fsl/include/basisfield/basisfield.cpp
    ../fsl/include/basisfield/dctfield.cpp
    ../fsl/include/basisfield/splinefield.cpp
    ../fsl/include/cprob/bdtr.cc
    ../fsl/include/cprob/btdtr.cc
    ../fsl/include/cprob/chdtr.cc
    ../fsl/include/cprob/const.cc
    ../fsl/include/cprob/drand.cc
    ../fsl/include/cprob/expx2.cc
    ../fsl/include/cprob/fdtr.cc
    ../fsl/include/cprob/gamma.cc
    ../fsl/include/cprob/gdtr.cc
    ../fsl/include/cprob/igam.cc
    ../fsl/include/cprob/igami.cc
    ../fsl/include/cprob/incbet.cc
    ../fsl/include/cprob/incbi.cc
    ../fsl/include/cprob/kolmogorov.cc
    ../fsl/include/cprob/mtherr.cc
    ../fsl/include/cprob/nbdtr.cc
    ../fsl/include/cprob/ndtr.cc
    ../fsl/include/cprob/ndtri.cc
    ../fsl/include/cprob/pdtr.cc
    ../fsl/include/cprob/polevl.cc
    ../fsl/include/cprob/stdtr.cc
    ../fsl/include/cprob/unity.cc
    ../fsl/include/cprob/xmath.cc
    ../fsl/include/meshclass/mesh.cpp
    ../fsl/include/meshclass/mpoint.cpp
    ../fsl/include/meshclass/point.cpp
    ../fsl/include/meshclass/profile.cpp
    ../fsl/include/meshclass/pt_special.cpp
    ../fsl/include/meshclass/triangle.cpp
    ../fsl/include/miscmaths/base2z.cc
    ../fsl/include/miscmaths/bfmatrix.cpp
    ../fsl/include/miscmaths/cspline.cc
    ../fsl/include/miscmaths/f2z.cc
    ../fsl/include/miscmaths/histogram.cc
    ../fsl/include/miscmaths/kernel.cc
    ../fsl/include/miscmaths/minimize.cc
    ../fsl/include/miscmaths/miscmaths.cc
    ../fsl/include/miscmaths/miscprob.cc
    ../fsl/include/miscmaths/nonlin.cpp
    ../fsl/include/miscmaths/optimise.cc
    ../fsl/include/miscmaths/rungekutta.cc
    ../fsl/include/miscmaths/Simplex.cpp
    ../fsl/include/miscmaths/sparse_matrix.cc
    ../fsl/include/miscmaths/sparsefn.cc
    ../fsl/include/miscmaths/SpMatMatrices.cpp
    ../fsl/include/miscmaths/t2z.cc
    ../fsl/include/newimage/complexvolume.cc
    ../fsl/include/newimage/imfft.cc
    ../fsl/include/newimage/lazy.cc
    ../fsl/include/newimage/newimage.cc
    ../fsl/include/newimage/newimagefns.cc
    ../fsl/include/NewNifti/legacyFunctions.cc
    ../fsl/include/NewNifti/NewNifti.cc
    ../fsl/include/topup/displacement_vector.cpp
    ../fsl/include/topup/topup_costfunctions.cpp
    ../fsl/include/topup/topup_file_io.cpp
    ../fsl/include/topup/topup_matrices.cpp
    ../fsl/include/topup/topupfns.cpp
    ../fsl/include/utils/check.cc
    ../fsl/include/utils/functions.cc
    ../fsl/include/utils/log.cc
    ../fsl/include/utils/matches.cc
    ../fsl/include/utils/parse.cc
    ../fsl/include/utils/time_tracer.cc
    ../fsl/include/utils/usage.cc
    ../fsl/include/warpfns/fnirt_file_reader.cpp
    ../fsl/include/warpfns/fnirt_file_writer.cpp
    ../fsl/include/warpfns/point_list.cpp
    ../fsl/include/warpfns/warpfns.cc
    ../fsl/include/znzlib/znzlib.c
    ../fsl/include/utils/FSLProfiler.cpp
    ../fsl/include/newimage/costfns.cc
    ../fsl/include/newimage/generalio.cc)

add_executable(eddy ${SOURCES} ${OTHER_SOURCES} ${OTHER_EXEC})

set_target_properties(eddy PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

target_compile_definitions(eddy PUBLIC ARMA_NO_DEBUG)
if(CUDA_ARCH)
    target_compile_definitions(eddy PUBLIC BUILDSTRING=TinyFSL_CUDA_SM${CUDA_ARCH})
else()
    target_compile_definitions(eddy PUBLIC BUILDSTRING=TinyFSL)
endif()

if(MSVC)
    target_compile_options(eddy PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>")
endif()

if(APPLE) 
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -framework Accelerate")
endif()

target_include_directories(eddy PUBLIC
  ${CMAKE_SOURCE_DIR}/../fsl/include/cudabasisfield
  ${CMAKE_SOURCE_DIR}/../fsl/include/eddy
  ${CMAKE_SOURCE_DIR}/../fsl/include
  ${CMAKE_SOURCE_DIR}/../fsl
  ${CMAKE_BINARY_DIR}
  ${CUDA_INCLUDE_DIRS})

if(EXISTS "${TIPL_DIR}")
  include_directories(eddy ${TIPL_DIR})
else()
  find_package(TIPL REQUIRED)
  target_link_libraries(eddy TIPL::tipl)
endif()
target_link_directories(eddy PUBLIC ${BLAS_LIB_DIR})
target_link_libraries(eddy ${BLAS_LIBRARIES} ZLIB::ZLIB)


if(CUDA_ARCH)
  target_compile_definitions(eddy PUBLIC COMPILE_GPU)
  target_link_libraries(eddy ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} )
  include_directories(eddy ${CUDA_PATH}/include)
  set_property(TARGET eddy PROPERTY CUDA_ARCHITECTURES ${CUDA_ARCH})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -Xcompiler /Zc:__cplusplus")
  set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
endif()


