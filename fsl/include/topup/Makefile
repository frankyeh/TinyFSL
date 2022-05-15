include ${FSLCONFDIR}/default.mk

PROJNAME = topup
XFILES   = topup applytopup
FXFILES  = test_displacement_vector
SOFILES  = libfsl-topup.so
LIBS     = -lfsl-warpfns -lfsl-meshclass -lfsl-basisfield \
           -lfsl-newimage -lfsl-miscmaths -lfsl-cprob -lfsl-NewNifti \
           -lfsl-znz -lfsl-utils

all: ${XFILES} ${SOFILES} schedule

schedule:
	@if [ ! -d ${DESTDIR}/etc ] ; then ${MKDIR} ${DESTDIR}/etc ; ${CHMOD} g+w ${DESTDIR}/etc ; fi
	@if [ ! -d ${DESTDIR}/etc/flirtsch ] ; then ${MKDIR} ${DESTDIR}/etc/flirtsch ; ${CHMOD} g+w ${DESTDIR}/etc/flirtsch ; fi
	${CP} -rf flirtsch/* ${DESTDIR}/etc/flirtsch/.

libfsl-topup.so: topup_matrices.o topup_file_io.o topup_costfunctions.o topupfns.o displacement_vector.o
	${CXX} ${CXXFLAGS} -shared -o $@ $^ ${LDFLAGS}

topup: topup.o libfsl-topup.so
	${CXX} ${CXXFLAGS} -o $@ $< libfsl-topup.so ${LDFLAGS}

applytopup: applytopup.o  libfsl-topup.so
	${CXX} ${CXXFLAGS} -o $@ $< libfsl-topup.so ${LDFLAGS}

test_displacement_vector: test_displacement_vector.o libfsl-topup.so
	${CXX} ${CXXFLAGS} -o $@ $< libfsl-topup.so ${LDFLAGS}
