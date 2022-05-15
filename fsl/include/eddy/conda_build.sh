if [ -z ${CONDA_BUILD+x} ]; then
    source /opt/conda/conda-bld/fsl-eddy-cuda-11.3_1638204222068/work/build_env_setup.sh
fi
#!/usr/bin/env bash

export FSLDIR=$PREFIX
export FSLDEVDIR=$PREFIX

mkdir -p $PREFIX/src/
cp -r $(pwd) $PREFIX/src/$PKG_NAME

. $FSLDIR/etc/fslconf/fsl-devel.sh

# Only build/install GPU components,
# static linking to CUDA runtime
make CUDA_STATIC=1 cuda=1
make CUDA_STATIC=1 cuda=1 install
