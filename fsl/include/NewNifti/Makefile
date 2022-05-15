include ${FSLCONFDIR}/default.mk

PROJNAME = NewNifti
LIBS     = -lfsl-znz
SOFILES  = libfsl-NewNifti.so
OBJS     = legacyFunctions.o NewNifti.o

all: libfsl-NewNifti.so

libfsl-NewNifti.so: ${OBJS}
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ ${LDFLAGS}
