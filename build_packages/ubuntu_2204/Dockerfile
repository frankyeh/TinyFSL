FROM ubuntu:22.04 as builder-stage

ENV DEBIAN_FRONTEND noninteractive

# Prepare environment
RUN apt update && apt upgrade -y && \
  apt install -y \
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
  apt-get -y autoremove --purge && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


ADD "https://api.github.com/repos/frankyeh/TinyFSL/commits?per_page=1" latest_commit
ADD "https://api.github.com/repos/frankyeh/TIPL/commits?per_page=1" latest_commit

RUN git clone https://github.com/frankyeh/TIPL.git \
  && git clone https://github.com/frankyeh/TinyFSL.git \
  && mv TinyFSL tiny_fsl \
  && mv tiny_fsl /opt \
  && mv TIPL /opt/tiny_fsl/fsl \
  && cp -fr /opt/tiny_fsl/patches/* /opt/tiny_fsl/fsl/include/ \
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


FROM scratch
COPY --from=builder-stage /opt/tiny_fsl_bin /




