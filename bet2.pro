QT -= gui core

DEFINES += ARMA_NO_DEBUG
CONFIG += c++11 console

win32* {
LIBS += -llibopenblas -lzlibstatic
}

linux* {
LIBS += -lz -lopenblas
}

mac{
CONFIG -= app_bundle
LIBS += -lz /usr/local/opt/openblas/lib/libopenblas64_.0.3.0.dev.a \
            /usr/local/opt/gcc/lib/gcc/11/libgfortran.a \
            /usr/local/opt/gcc/lib/gcc/11/libquadmath.a \
            /usr/local/opt/gcc/lib/gcc/11/libgomp.a \
            /usr/local/opt/gcc/lib/gcc/11/libgcc_s.1.dylib
}


# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

INCLUDEPATH += include $$[QT_INSTALL_HEADERS]/QtZlib
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    include/cprob/bdtr.cc \
    include/cprob/btdtr.cc \
    include/cprob/chdtr.cc \
    include/cprob/const.cc \
    include/cprob/drand.cc \
    include/cprob/expx2.cc \
    include/cprob/fdtr.cc \
    include/cprob/gamma.cc \
    include/cprob/gdtr.cc \
    include/cprob/igam.cc \
    include/cprob/igami.cc \
    include/cprob/incbet.cc \
    include/cprob/incbi.cc \
    include/cprob/kolmogorov.cc \
    include/cprob/mtherr.cc \
    include/cprob/nbdtr.cc \
    include/cprob/ndtr.cc \
    include/cprob/ndtri.cc \
    include/cprob/pdtr.cc \
    include/cprob/polevl.cc \
    include/cprob/stdtr.cc \
    include/cprob/unity.cc \
    include/cprob/xmath.cc \
    include/miscmaths/base2z.cc \
    include/miscmaths/bfmatrix.cpp \
    include/miscmaths/cspline.cc \
    include/miscmaths/f2z.cc \
    include/miscmaths/histogram.cc \
    include/miscmaths/kernel.cc \
    include/miscmaths/minimize.cc \
    include/miscmaths/miscmaths.cc \
    include/miscmaths/miscprob.cc \
    include/miscmaths/nonlin.cpp \
    include/miscmaths/optimise.cc \
    include/miscmaths/rungekutta.cc \
    include/miscmaths/Simplex.cpp \
    include/miscmaths/sparse_matrix.cc \
    include/miscmaths/sparsefn.cc \
    include/miscmaths/SpMatMatrices.cpp \
    include/miscmaths/t2z.cc \
    include/meshclass/mesh.cpp \
    include/meshclass/mpoint.cpp \
    include/meshclass/point.cpp \
    include/meshclass/profile.cpp \
    include/meshclass/pt_special.cpp \
    include/meshclass/triangle.cpp \
    include/newimage/complexvolume.cc \
    include/newimage/imfft.cc \
    include/newimage/lazy.cc \
    include/newimage/newimage.cc \
    include/newimage/newimagefns.cc \
    include/newimage/costfns.cc \
    include/newimage/generalio.cc \
    include/newnifti/legacyFunctions.cc \
    include/newnifti/NewNifti.cc \
    include/utils/check.cc \
    include/utils/functions.cc \
    include/utils/log.cc \
    include/utils/matches.cc \
    include/utils/parse.cc \
    include/utils/time_tracer.cc \
    include/utils/usage.cc \
    include/utils/FSLProfiler.cpp \
    include/znzlib/znzlib.c \
    include/bet2/bet2.cpp
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
