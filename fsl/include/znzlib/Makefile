include ${FSLCONFDIR}/default.mk

PROJNAME    = znzlib
TESTXFILES  = testprog
SOFILES     = libfsl-znz.so

all: libfsl-znz.so

test: ${TESTXFILES}

libfsl-znz.so: znzlib.o
	${CC} ${CFLAGS} -shared -o $@ $^ ${LDFLAGS}

testprog: libfsl-znz.so testprog.c
	${CC} ${CFLAGS} -o testprog testprog.c -lfsl-znz ${LDFLAGS}
