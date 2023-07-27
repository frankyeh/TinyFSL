cmake_minimum_required(VERSION 3.16)
project(bet2 VERSION "2022.01.17" LANGUAGES CXX;C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF)

if(APPLE) 
    set(BLA_VENDOR Apple)
    set(BLAS_LIBRARIES "-framework Accelerate")
else()
    set(CMAKE_EXE_LINKER_FLAGS "-static")
    set(BLA_VENDOR OpenBLAS)
endif()
find_package(BLAS REQUIRED)
find_package(ZLIB REQUIRED)

set(SOURCES
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
    ../fsl/include/meshclass/mesh.cpp
    ../fsl/include/meshclass/mpoint.cpp
    ../fsl/include/meshclass/point.cpp
    ../fsl/include/meshclass/profile.cpp
    ../fsl/include/meshclass/pt_special.cpp
    ../fsl/include/meshclass/triangle.cpp
    ../fsl/include/newimage/complexvolume.cc
    ../fsl/include/newimage/imfft.cc
    ../fsl/include/newimage/lazy.cc
    ../fsl/include/newimage/newimage.cc
    ../fsl/include/newimage/newimagefns.cc
    ../fsl/include/newimage/costfns.cc
    ../fsl/include/newimage/generalio.cc
    ../fsl/include/NewNifti/legacyFunctions.cc
    ../fsl/include/NewNifti/NewNifti.cc
    ../fsl/include/utils/check.cc
    ../fsl/include/utils/functions.cc
    ../fsl/include/utils/log.cc
    ../fsl/include/utils/matches.cc
    ../fsl/include/utils/parse.cc
    ../fsl/include/utils/time_tracer.cc
    ../fsl/include/utils/usage.cc
    ../fsl/include/utils/FSLProfiler.cpp
    ../fsl/include/znzlib/znzlib.c
    ../fsl/include/bet2/bet2.cpp)

add_executable(bet2 ${SOURCES} ${OTHER_EXEC})

set_target_properties(bet2 PROPERTIES CXX_STANDARD 17 CXX_EXTENSIONS NONE)

target_compile_definitions(bet2 PUBLIC ARMA_NO_DEBUG)
target_compile_definitions(bet2 PUBLIC BUILDSTRING=TinyFSL)

if(MSVC)
    target_compile_options(bet2 PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/Zc:__cplusplus>")
endif()

if(UNIX AND NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
endif()


target_include_directories(bet2 PUBLIC
  ${CMAKE_SOURCE_DIR}/../fsl/../fsl/include/bet2
  ${CMAKE_SOURCE_DIR}/../fsl/include
  ${CMAKE_SOURCE_DIR}/../fsl
  ${CMAKE_BINARY_DIR})

if(EXISTS "${TIPL_DIR}")
  include_directories(bet2 ${TIPL_DIR})
else()
  find_package(TIPL REQUIRED)
  target_link_libraries(bet2 TIPL::tipl)
endif()
target_link_directories(bet2 PUBLIC ${BLAS_LIB_DIR})
target_link_libraries(bet2 ${BLAS_LIBRARIES} ZLIB::ZLIB)
