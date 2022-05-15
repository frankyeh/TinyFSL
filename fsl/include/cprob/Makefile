include ${FSLCONFDIR}/default.mk

PROJNAME   = cprob
SOFILES    = libfsl-cprob.so
TESTXFILES = dftest

OBJS = bdtr.o btdtr.o chdtr.o drand.o expx2.o fdtr.o gamma.o gdtr.o \
igam.o igami.o incbet.o incbi.o mtherr.o nbdtr.o ndtr.o ndtri.o pdtr.o \
stdtr.o unity.o polevl.o const.o xmath.o

all: libfsl-cprob.so

dftest: dftest.o libfsl-cprob.so
	$(CXX) $(CXXFLAGS) -o dftest dftest.o -lfsl-cprob $(LDFLAGS)

libfsl-cprob.so: $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ ${LDFLAGS}
