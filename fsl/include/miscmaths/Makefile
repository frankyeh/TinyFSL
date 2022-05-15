# A Makefile for the miscmaths class

include ${FSLCONFDIR}/default.mk

PROJNAME = miscmaths
SOFILES  = libfsl-miscmaths.so
LIBS     = -lfsl-utils -lfsl-cprob -lfsl-NewNifti
OBJS     = miscmaths.o optimise.o miscprob.o kernel.o histogram.o \
           base2z.o t2z.o f2z.o minimize.o cspline.o sparse_matrix.o \
           sparsefn.o rungekutta.o nonlin.o bfmatrix.o Simplex.o \
           SpMatMatrices.o

all: libfsl-miscmaths.so

libfsl-miscmaths.so: ${OBJS}
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ ${LDFLAGS}
