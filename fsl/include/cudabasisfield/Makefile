include ${FSLCONFDIR}/default.mk

PROJNAME = cudabasisfield
SOFILES  = libfsl-cudabasisfield_cuda${CUDA_VER}.so
HFILES   = *.h *.cuh
OBJS     = CBFKernelDefinitions.o CBFSparseDiagonalMatrix.o CBFSplineField.o

all: ${SOFILES}

libfsl-cudabasisfield_cuda${CUDA_VER}.so: ${OBJS}
	${NVCC} ${NVCCFLAGS} -shared -o $@ $^ ${NVCCLDFLAGS}

%.o: %.cu
	${NVCC} ${NVCCFLAGS} -c -o $@ $<

# Define explicit prerequisites to minimise
# re-compilation time during development.
CBFKernelDefinitions.o:    CBFKernelDefinitions.cu \
                           CBFKernels.cuh \
                           CBFKernelHelpers.cuh
CBFSparseDiagonalMatrix.o: CBFSparseDiagonalMatrix.cu \
                           CBFSparseDiagonalMatrix.cuh
CBFSplineField.o:          CBFSplineField.cu \
                           CBFSplineField.cuh \
                           CBFMemory.h \
                           CBFSparseDiagonalMatrix.cuh \
                           CBFKernels.cuh \
                           CBFKernelHelpers.cuh
