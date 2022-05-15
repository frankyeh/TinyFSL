include ${FSLCONFDIR}/default.mk

PROJNAME = bet2
XFILES   = bet2 betsurf robustfov
SCRIPTS  = old_betall bet
RUNTCLS  = Bet
LIBS     = -lfsl-meshclass -lfsl-newimage -lfsl-miscmaths \
           -lfsl-NewNifti -lfsl-cprob -lfsl-utils -lfsl-znz

all: ${XFILES}

%: %.o
	${CXX} ${CXXFLAGS} -o $@ $^ ${LDFLAGS}
