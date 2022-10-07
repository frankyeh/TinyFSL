FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  libopenblas-dev \
  unzip \
  curl \
  cmake \
  ninja-build \
  make \
  git \
  libboost-all-dev \
  zlib1g-dev \
  ca-certificates \
  gcc \
  g++ && \
  apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ADD "https://api.github.com/repos/frankyeh/TinyFSL/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

ENV OS="Linux"
ENV FSLDIR=/opt/tiny_fsl
ENV FSL_DIR=$FSLDIR
ENV FSLOUTPUTTYPE="NIFTI_GZ"
ENV FSLMULTIFILEQUIT="TRUE"
ENV LD_LIBRARY_PATH=$FSLDIR/lib:$LD_LIBRARY_PATH
ENV PATH=$FSLDIR/bin:$PATH



RUN git clone https://github.com/frankyeh/TIPL.git \
  && git clone https://github.com/frankyeh/TinyFSL.git \
  && mv TinyFSL tiny_fsl \
  && mv tiny_fsl /opt \
  && mv TIPL /opt/tiny_fsl/fsl \
  && mkdir -p /opt/tiny_fsl_bin 


RUN mkdir -p /opt/tiny_fsl/build_topup \
  && cd /opt/tiny_fsl/topup \
  && cmake -S . -B ../build_topup -GNinja "-DCMAKE_BUILD_TYPE:STRING=Release" -DTIPL_DIR=../fsl \
  && cmake --build ../build_topup --parallel --config Release \
  && mv ../build_topup/topup /opt/tiny_fsl_bin
  
RUN mkdir -p /opt/tiny_fsl/build_bet2 \
  && cd /opt/tiny_fsl/bet2 \
  && cmake -S . -B ../build_bet2 -GNinja "-DCMAKE_BUILD_TYPE:STRING=Release" -DTIPL_DIR=../fsl \
  && cmake --build ../build_bet2 --parallel --config Release \
  && mv ../build_bet2/bet2 /opt/tiny_fsl_bin


RUN mkdir -p /opt/tiny_fsl/build_applytopup \
  && cd /opt/tiny_fsl/applytopup \
  && cmake -S . -B ../build_applytopup -GNinja "-DCMAKE_BUILD_TYPE:STRING=Release" -DTIPL_DIR=../fsl \
  && cmake --build ../build_applytopup --parallel --config Release \
  && mv ../build_applytopup/applytopup /opt/tiny_fsl_bin

RUN cd /opt \
  && rm -fr tiny_fsl \
  && mv /opt/tiny_fsl_bin /opt/bin

  





