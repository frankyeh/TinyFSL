#
# This is the makefile for the eddy project. It can be used
# to compile the following variants of eddy:
#
#   - make cuda=1 (compiles a CUDA-capable version)
#   - make cpu=1  (compiles a multi-threaded CPU version)
#   - make        (equivalent to make cpu=1)
#
# The resulting eddy binary executables will be named like so:
#   - cpu:   eddy_cpu
#   - cuda:  eddy_cudaX.Y (where X.Y is the CUDA version that
#            the binary was compiled against)
#
# Both variants can be compiled in one invocation, e.g.:
#
#     make cpu=1 cuda=1
#
# Notes regarding the cuda variant:
#
#  - By default a fat binary with code for all the compute
#    capabilities allowed for by a given CUDA version will
#    be compiled. If one only need 3.0--3.5 (for testing)
#    one can build with the fastbuild=1 option which saves
#    a lot of compilation time, e.g.:
#
#        make cuda=1 fastbuild=1
#
#  - Alternately, to compile for a specific compute
#    capabillity (or range thereof), you can specify the
#    GENCODEFLAGS variable, e.g.:
#
#        make cuda=1 GENCODEFLAGS="-gencode arch=compute_30,code=sm_30"
#
#  - Or you can specify a specific compute capability with
#    the COMPUTE_CAPABILITY vraiable, e.g.:
#
#        make cuda=1 COMPUTE_CAPABILITY="30"
#
#  - By default, all CUDA libraries (libcuda, libcudart, and those
#    specified in the CUDALIBS variable below) will be dynamically
#    linked in the resulting executable. The CUDA_STATIC variable
#    can be specified to statically link these libraries instead,
#    e.g.:
#
#        make cuda=1 CUDA_STATIC=1
#
# The CUDA_HOME, GENCODEFLAGS, and CUDA_STATIC variables are all
# handled in $FSLCONFDIR/buildSettings.mk.

include ${FSLCONFDIR}/default.mk

PROJNAME = eddy
SCRIPTS  = eddy
XFILES   =

ifdef cpu
  XFILES += eddy_cpu
endif
ifdef cuda
  XFILES += eddy_cuda${CUDA_VER}
endif

# Default to cpu variant
# if no flags were provided
ifeq (${XFILES},)
  XFILES = eddy_cpu
endif

# -rdynamic allows meaningful backtraces to
# be emitted on segmentation faults and
# other crashes (see fsl/utils/stack_dump.h).
# Understood by both clang++ and g++
USRLDFLAGS   = -rdynamic
USRNVCCFLAGS = -DCOMPILE_GPU -Icuda
CUDALIBS     = -lfsl-cudabasisfield_cuda${CUDA_VER} -lcublas
LIBS         = -lfsl-topup -lfsl-warpfns -lfsl-meshclass -lfsl-basisfield \
               -lfsl-newimage -lfsl-miscmaths -lfsl-cprob -lfsl-NewNifti \
               -lfsl-znz -lfsl-utils

# Skip CUDA fat binary creation if fastbuild
# is set (overriding GENCODEFLAGS defined
# in FSLDIR/config/buildSettings.mk)
ifeq ($(fastbuild),1)
  GENCODEFLAGS := -gencode arch=compute_30,code=sm_30
endif

# Build a specific compute capability if
# requested (overridding fastbuild and
# GENCODEFLAGS)
ifdef COMPUTE_CAPABILITY
  GENCODEFLAGS := -gencode arch=compute_$(COMPUTE_CAPABILITY),code=sm_$(COMPUTE_CAPABILITY)
endif

# Compiled for all variants
OBJS     := eddy.o b0Predictor.o BiasFieldEstimatorImpl.o \
            CPUStackResampler.o DiffusionGP.o ECModels.o \
            ECScanClasses.o EddyCommandLineOptions.o \
            EddyHelperClasses.o EddyUtils.o HyParEstimator.o \
            KMatrix.o MoveBySuscCF.o PostEddyAlignShellsFunctions.o

# Compiled for CPU variant
CPUOBJS  := LSResampler.o PostEddyCF.o

# Compiled for cuda variant
CUDAOBJS := CudaVolume.o DerivativeCalculator.o DiffusionGP.o \
            EddyCudaHelperFunctions.o EddyGpuUtils.o EddyInternalGpuUtils.o \
            EddyKernels.o EddyMatrixKernels.o GpuPredictorChunk.o \
            LSResampler.o PostEddyCF.o StackResampler.o

# use separate build dirs for each variant
CPUOBJS  := $(OBJS:%.o=cpubuild/%.o)  $(CPUOBJS:%.o=cpubuild/%.o)
CUDAOBJS := $(OBJS:%.o=cudabuild/%.o) $(CUDAOBJS:%.o=cudabuild/cuda${CUDA_VER}/%.o)

all: ${XFILES}

clean:
	@rm -f depend.mk eddy_cpu eddy_cuda*
	@rm -rf cudabuild cpubuild

#################################
# CPU executable and object files
#################################

eddy_cpu: ${CPUOBJS}
	@mkdir -p cpubuild
	${CXX} ${CXXFLAGS} -fopenmp -o $@ $^ ${LDFLAGS}

cpubuild/%.o: %.cpp
	@mkdir -p cpubuild
	$(CXX) $(CXXFLAGS) -fopenmp -c -o $@ $<

#################################
# GPU executable and object files
#################################

eddy_cuda${CUDA_VER}: ${CUDAOBJS}
	${NVCC} ${NVCCFLAGS} -o $@ $^ ${NVCCLDFLAGS}

cudabuild/cuda${CUDA_VER}/%.o: cuda/%.cu
	@mkdir -p cudabuild/cuda${CUDA_VER}
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

cudabuild/%.o: %.cpp
	@mkdir -p cudabuild
	$(CXX) $(CUDACXXFLAGS) -c -o $@ $<

# For the CUDA variant, we explicitly list dependencies
# of each object file to minimise [re-]compilation time.
cudabuild/cuda${CUDA_VER}/EddyCudaHelperFunctions.o: cuda/EddyCudaHelperFunctions.h \
                                                     cuda/EddyCudaHelperFunctions.cu
cudabuild/cuda${CUDA_VER}/GpuPredictorChunk.o:       cuda/GpuPredictorChunk.h \
                                                     cuda/GpuPredictorChunk.cu \
                                                     EddyHelperClasses.h
cudabuild/cuda${CUDA_VER}/StackResampler.o:          cuda/StackResampler.h \
                                                     cuda/StackResampler.cu \
                                                     cuda/EddyCudaHelperFunctions.h \
                                                     EddyHelperClasses.h
cudabuild/cuda${CUDA_VER}/EddyKernels.o:             cuda/EddyKernels.h \
                                                     cuda/EddyKernels.cu
cudabuild/cuda${CUDA_VER}/EddyMatrixKernels.o:       cuda/EddyMatrixKernels.h \
                                                     cuda/eddy_matrix_kernels_internal.h \
                                                     cuda/EddyMatrixKernels.cu
cudabuild/cuda${CUDA_VER}/CudaVolume.o:              cuda/CudaVolume.h \
                                                     cuda/CudaVolume.cu \
                                                     cuda/EddyKernels.h \
                                                     cuda/EddyFunctors.h \
                                                     cuda/EddyCudaHelperFunctions.h \
                                                     EddyHelperClasses.h
cudabuild/cuda${CUDA_VER}/EddyInternalGpuUtils.o:    cuda/EddyInternalGpuUtils.h \
                                                     cuda/EddyInternalGpuUtils.cu \
                                                     cuda/CudaVolume.h \
                                                     cuda/EddyKernels.h \
                                                     cuda/EddyFunctors.h \
                                                     cuda/StackResampler.h \
                                                     cuda/DerivativeCalculator.h \
                                                     cuda/EddyCudaHelperFunctions.h \
                                                     EddyHelperClasses.h \
                                                     DiffusionGP.h \
                                                     b0Predictor.h \
                                                     ECScanClasses.h
cudabuild/cuda${CUDA_VER}/EddyGpuUtils.o:            cuda/EddyGpuUtils.h \
                                                     cuda/EddyGpuUtils.cu \
                                                     cuda/EddyInternalGpuUtils.h \
                                                     cuda/CudaVolume.h \
                                                     cuda/EddyKernels.h \
                                                     EddyHelperClasses.h \
                                                     DiffusionGP.h \
                                                     b0Predictor.h \
                                                     ECScanClasses.h \
                                                     EddyUtils.h
cudabuild/cuda${CUDA_VER}/DerivativeCalculator.o:    cuda/DerivativeCalculator.h \
                                                     cuda/DerivativeCalculator.cu \
                                                     cuda/EddyInternalGpuUtils.h \
                                                     cuda/EddyKernels.h \
                                                     cuda/CudaVolume.h \
                                                     cuda/EddyCudaHelperFunctions.h \
                                                     EddyHelperClasses.h \
                                                     ECScanClasses.h
cudabuild/cuda${CUDA_VER}/DiffusionGP.o:             cuda/DiffusionGP.cu \
                                                     DiffusionGP.h \
                                                     EddyHelperClasses.h \
                                                     DWIPredictionMaker.h \
                                                     EddyUtils.h \
                                                     cuda/CudaVolume.h
cudabuild/cuda${CUDA_VER}/LSResampler.o:             cuda/LSResampler.cu \
                                                     cuda/EddyCudaHelperFunctions.h \
                                                     LSResampler.h \
                                                     EddyHelperClasses.h \
                                                     EddyUtils.h \
                                                     ECScanClasses.h
cudabuild/cuda${CUDA_VER}/PostEddyCF.o:              cuda/PostEddyCF.cu \
                                                     PostEddyCF.h \
                                                     cuda/CudaVolume.h \
                                                     EddyUtils.h \
                                                     cuda/EddyInternalGpuUtils.h
