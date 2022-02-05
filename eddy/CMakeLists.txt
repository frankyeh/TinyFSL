cmake_minimum_required(VERSION 3.19)
if(CUDA_ARCH)
    project(eddy VERSION "2022.01.17" LANGUAGES CUDA CXX;C)
else()
    project(eddy VERSION "2022.01.17" LANGUAGES CXX;C)
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenBLAS)
find_package(ZLIB REQUIRED)
if(CUDA_ARCH)
    find_package(CUDA REQUIRED)
endif()

set(SOURCES
    include/eddy/b0Predictor.cpp
    include/eddy/BiasFieldEstimatorImpl.cpp
    include/eddy/CPUStackResampler.cpp
    include/eddy/DiffusionGP.cpp
    include/eddy/ECModels.cpp
    include/eddy/ECScanClasses.cpp
    include/eddy/eddy.cpp
    include/eddy/EddyCommandLineOptions.cpp
    include/eddy/EddyHelperClasses.cpp
    include/eddy/EddyUtils.cpp
    include/eddy/HyParEstimator.cpp
    include/eddy/KMatrix.cpp
    include/eddy/LSResampler.cpp
    include/eddy/MoveBySuscCF.cpp
    include/eddy/PostEddyAlignShellsFunctions.cpp
    include/eddy/PostEddyCF.cpp
    include/basisfield/basisfield.cpp
    include/basisfield/dctfield.cpp
    include/basisfield/splinefield.cpp
    include/cprob/bdtr.cc
    include/cprob/btdtr.cc
    include/cprob/chdtr.cc
    include/cprob/const.cc
    include/cprob/drand.cc
    include/cprob/expx2.cc
    include/cprob/fdtr.cc
    include/cprob/gamma.cc
    include/cprob/gdtr.cc
    include/cprob/igam.cc
    include/cprob/igami.cc
    include/cprob/incbet.cc
    include/cprob/incbi.cc
    include/cprob/kolmogorov.cc
    include/cprob/mtherr.cc
    include/cprob/nbdtr.cc
    include/cprob/ndtr.cc
    include/cprob/ndtri.cc
    include/cprob/pdtr.cc
    include/cprob/polevl.cc
    include/cprob/stdtr.cc
    include/cprob/unity.cc
    include/cprob/xmath.cc
    include/meshclass/mesh.cpp
    include/meshclass/mpoint.cpp
    include/meshclass/point.cpp
    include/meshclass/profile.cpp
    include/meshclass/pt_special.cpp
    include/meshclass/triangle.cpp
    include/miscmaths/base2z.cc
    include/miscmaths/bfmatrix.cpp
    include/miscmaths/cspline.cc
    include/miscmaths/f2z.cc
    include/miscmaths/histogram.cc
    include/miscmaths/kernel.cc
    include/miscmaths/minimize.cc
    include/miscmaths/miscmaths.cc
    include/miscmaths/miscprob.cc
    include/miscmaths/nonlin.cpp
    include/miscmaths/optimise.cc
    include/miscmaths/rungekutta.cc
    include/miscmaths/Simplex.cpp
    include/miscmaths/sparse_matrix.cc
    include/miscmaths/sparsefn.cc
    include/miscmaths/SpMatMatrices.cpp
    include/miscmaths/t2z.cc
    include/newimage/complexvolume.cc
    include/newimage/imfft.cc
    include/newimage/lazy.cc
    include/newimage/newimage.cc
    include/newimage/newimagefns.cc
    include/newnifti/legacyFunctions.cc
    include/newnifti/NewNifti.cc
    include/topup/displacement_vector.cpp
    include/topup/topup_costfunctions.cpp
    include/topup/topup_file_io.cpp
    include/topup/topup_matrices.cpp
    include/topup/topupfns.cpp
    include/utils/check.cc
    include/utils/functions.cc
    include/utils/log.cc
    include/utils/matches.cc
    include/utils/parse.cc
    include/utils/time_tracer.cc
    include/utils/usage.cc
    include/warpfns/fnirt_file_reader.cpp
    include/warpfns/fnirt_file_writer.cpp
    include/warpfns/point_list.cpp
    include/warpfns/warpfns.cc
    include/znzlib/znzlib.c
    include/utils/FSLProfiler.cpp
    include/newimage/costfns.cc
    include/newimage/generalio.cc)

add_executable(eddy ${SOURCES})

set_target_properties(eddy PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

target_compile_definitions(dsi_studio PUBLIC ARMA_NO_DEBUG)


set_property(TARGET eddy PROPERTY CUDA_ARCHITECTURES 52 60 70 75)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")

if(EXISTS "${TIPL_DIR}")
  include_directories(eddy ${TIPL_DIR})
else()
  find_package(TIPL REQUIRED)
  target_link_libraries(eddy TIPL::tipl)
endif()

target_link_libraries(eddy ${CUDA_LIBRARIES} OpenBLAS::OpenBLAS)

