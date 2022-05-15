# A Makefile for the newimage library
include ${FSLCONFDIR}/default.mk

# Generate a string which describes
# the newimage version we are compiling
GIT_SHA1    = $(shell git describe --tags --dirty)
GIT_DATE    = $(shell git log -1 --format=%ad --date=iso-strict)
BUILDSTRING = "${GIT_SHA1} ${GIT_DATE}"

PROJNAME     = newimage
SOFILES      = libfsl-newimage.so
USRINCFLAGS  = -DEXPOSE_TREACHEROUS -DBUILDSTRING=\"${BUILDSTRING}\"
LIBS         = -lfsl-miscmaths -lfsl-cprob -lfsl-NewNifti -lfsl-utils \
               -lfsl-znz

OBJS  = lazy.o newimage.o generalio.o newimagefns.o \
        complexvolume.o imfft.o costfns.o

all: libfsl-newimage.so

libfsl-newimage.so: ${OBJS}
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ ${LDFLAGS}
