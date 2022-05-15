if [ -z ${CONDA_BUILD+x} ]; then
    source /opt/conda/conda-bld/fsl-bet2_1637942042847/work/build_env_setup.sh
fi
#!/usr/bin/env bash

export FSLDIR=$PREFIX
export FSLDEVDIR=$PREFIX

mkdir -p $PREFIX/src/
cp -r $(pwd) $PREFIX/src/$PKG_NAME

. $FSLDIR/etc/fslconf/fsl-devel.sh

make
make install
