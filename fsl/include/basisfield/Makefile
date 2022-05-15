#
# Makfile for objects in the BASISFIELD namespace
#

include ${FSLCONFDIR}/default.mk

PROJNAME = basisfield
LIBS     = -lfsl-miscmaths -lfsl-newimage

all: libfsl-basisfield.so

libfsl-basisfield.so: dctfield.o splinefield.o basisfield.o
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ ${LDFLAGS}
