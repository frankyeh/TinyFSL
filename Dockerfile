FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  unzip \
  curl \
  make \
  git \
  libboost-all-dev \
  zlib1g-dev \
  ca-certificates \
  qt5-qmake \
  qt5-default \
  gcc \
  g++ && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Fix issues: Singularity container cannot load libQt5Core.so.5 on CentOS 7
RUN strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5
RUN ldconfig

ADD "https://api.github.com/repos/frankyeh/TinyFSL/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit


RUN git clone https://github.com/frankyeh/TIPL.git \
  && git clone https://github.com/frankyeh/TinyFSL.git \
  && mv TinyFSL tiny_fsl \
  && mv tiny_fsl /opt \
  && mv Tiny tipl \
  && mv tipl /opt/tiny_fsl/include

RUN cd /opt/tiny_fsl \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2 --output fsl-armawrap-0.6.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-avwutils-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-bet2-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-basisfield-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-cprob-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-eddy-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-meshclass-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-miscmaths-2111.1-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-newimage-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 --output fsl-newnifti-4.0.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-topup-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-topup-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-utils-2111.1-h2bc3f7f_0.tar.bz2 --output fsl-utils-2111.1-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-warpfns-2111.0-h2bc3f7f_0.tar.bz2 \
  && curl https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/linux-64/fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2 --output fsl-znzlib-2111.0-h2bc3f7f_0.tar.bz2 \
  && bunzip2 *.bz2 \
  && for i in *.tar;do tar -xf "$i";done \
  && cd src \
  && for i in "fsl-"*;do mv -v -f "$i" "${i#"fsl-"}";done \
  && for i in *;do cp -v -r -f "$i" ../include/;done \
  && cd .. \
  && cd patches \
  && for i in *;do cp -v -r -f "$i" ../include/;done \
  && cd .. 


RUN mkdir -p /opt/tiny_fsl/build \
  && cd /opt/tiny_fsl/build \
  && qmake ../eddy.pro \
  && make \
  && qmake ../topup.pro \
  && make \
  && qmake ../applytopup.pro \
  && make \
  && qmake ../bet2.pro \
  && make \
  && chmod 755 eddy \
  && chmod 755 topup \
  && chmod 755 applytopup \
  && chmod 755 bet2 \
  && mkdir -p /opt/tiny_fsl_bin \
  && mv eddy /opt/tiny_fsl_bin \
  && mv topup /opt/tiny_fsl_bin \
  && mv applytopup /opt/tiny_fsl_bin \
  && mv bet2 /opt/tiny_fsl_bin


RUN cd /opt
  && rm -fr tiny_fsl \
  && mkdir tiny_fsl \
  && mv tiny_fsl_bin tiny_fsl \
  && cd /opt/tiny_fsl \
  && mv tiny_fsl_bin bin

ENV OS="Linux" \
    FSLDIR="/opt/tiny_fsl" \
    FSL_DIR="$FSLDIR" \
    FSLOUTPUTTYPE="NIFTI_GZ" \
    FSLMULTIFILEQUIT="TRUE" \
    LD_LIBRARY_PATH="$FSLDIR/lib:$LD_LIBRARY_PATH" \
    PATH="$FSLDIR/bin:$PATH"
  





