QT -= gui core

DEFINES += ARMA_NO_DEBUG
CONFIG += c++17 console

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

INCLUDEPATH += fsl/include $$[QT_INSTALL_HEADERS]/QtZlib
# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    fsl/include/cprob/bdtr.cc \
    fsl/include/cprob/btdtr.cc \
    fsl/include/cprob/chdtr.cc \
    fsl/include/cprob/const.cc \
    fsl/include/cprob/drand.cc \
    fsl/include/cprob/expx2.cc \
    fsl/include/cprob/fdtr.cc \
    fsl/include/cprob/gamma.cc \
    fsl/include/cprob/gdtr.cc \
    fsl/include/cprob/igam.cc \
    fsl/include/cprob/igami.cc \
    fsl/include/cprob/incbet.cc \
    fsl/include/cprob/incbi.cc \
    fsl/include/cprob/kolmogorov.cc \
    fsl/include/cprob/mtherr.cc \
    fsl/include/cprob/nbdtr.cc \
    fsl/include/cprob/ndtr.cc \
    fsl/include/cprob/ndtri.cc \
    fsl/include/cprob/pdtr.cc \
    fsl/include/cprob/polevl.cc \
    fsl/include/cprob/stdtr.cc \
    fsl/include/cprob/unity.cc \
    fsl/include/cprob/xmath.cc \
    fsl/include/miscmaths/base2z.cc \
    fsl/include/miscmaths/bfmatrix.cpp \
    fsl/include/miscmaths/cspline.cc \
    fsl/include/miscmaths/f2z.cc \
    fsl/include/miscmaths/histogram.cc \
    fsl/include/miscmaths/kernel.cc \
    fsl/include/miscmaths/minimize.cc \
    fsl/include/miscmaths/miscmaths.cc \
    fsl/include/miscmaths/miscprob.cc \
    fsl/include/miscmaths/nonlin.cpp \
    fsl/include/miscmaths/optimise.cc \
    fsl/include/miscmaths/rungekutta.cc \
    fsl/include/miscmaths/Simplex.cpp \
    fsl/include/miscmaths/sparse_matrix.cc \
    fsl/include/miscmaths/sparsefn.cc \
    fsl/include/miscmaths/SpMatMatrices.cpp \
    fsl/include/miscmaths/t2z.cc \
    fsl/include/meshclass/mesh.cpp \
    fsl/include/meshclass/mpoint.cpp \
    fsl/include/meshclass/point.cpp \
    fsl/include/meshclass/profile.cpp \
    fsl/include/meshclass/pt_special.cpp \
    fsl/include/meshclass/triangle.cpp \
    fsl/include/newimage/complexvolume.cc \
    fsl/include/newimage/imfft.cc \
    fsl/include/newimage/lazy.cc \
    fsl/include/newimage/newimage.cc \
    fsl/include/newimage/newimagefns.cc \
    fsl/include/newimage/costfns.cc \
    fsl/include/newimage/generalio.cc \
    fsl/include/newnifti/legacyFunctions.cc \
    fsl/include/newnifti/NewNifti.cc \
    fsl/include/utils/check.cc \
    fsl/include/utils/functions.cc \
    fsl/include/utils/log.cc \
    fsl/include/utils/matches.cc \
    fsl/include/utils/parse.cc \
    fsl/include/utils/time_tracer.cc \
    fsl/include/utils/usage.cc \
    fsl/include/utils/FSLProfiler.cpp \
    fsl/include/znzlib/znzlib.c \
    fsl/include/bet2/bet2.cpp
# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
