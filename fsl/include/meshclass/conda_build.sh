if [ -z ${CONDA_BUILD+x} ]; then
    source /opt/conda/conda-bld/fsl-meshclass_1637922058555/work/build_env_setup.sh
fi
#!/usr/bin/env bash

export FSLDIR=$PREFIX
export FSLDEVDIR=$PREFIX

mkdir -p $PREFIX/src/
cp -r $(pwd) $PREFIX/src/$PKG_NAME

. $FSLDIR/etc/fslconf/fsl-devel.sh

make
make install
