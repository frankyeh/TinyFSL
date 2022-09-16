FROM ubuntu:20.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt update && apt full-upgrade -y && \
  apt install -y --no-install-recommends \
  libopenblas-dev \
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
  && cp -fr /opt/tiny_fsl/patches/* /opt/tiny_fsl/fsl/include/


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


RUN cd /opt \
  && rm -fr tiny_fsl \
  && mkdir tiny_fsl \
  && mv tiny_fsl_bin tiny_fsl \
  && cd /opt/tiny_fsl \
  && mv tiny_fsl_bin bin

  
FROM scratch
COPY --from=builder-stage /opt/tiny_fsl /




