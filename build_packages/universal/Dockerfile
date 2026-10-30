# Portable TinyFSL CPU/CUDA binaries; base images match DSI Studio's universal Linux builds.
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS builder

ARG CUDA_BUILD=0
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /opt/tiny_fsl

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev libboost-all-dev zlib1g-dev gfortran git file binutils \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN rm -rf fsl/include/TIPL \
    && git clone --depth 1 https://github.com/frankyeh/TIPL.git fsl/include/TIPL

RUN set -eu; \
    blas="$(find /usr/lib -name libopenblas.a -print -quit)"; \
    zlib="$(find /usr/lib -name libz.a -print -quit)"; \
    test -s "$blas" && test -s "$zlib"; \
    mkdir -p /opt/tiny_fsl_bin; \
    for app in topup applytopup eddy; do \
      cmake -S "$app" -B "/opt/build_$app" -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTIPL_DIR=/opt/tiny_fsl/fsl/include \
        -DBLAS_LIBRARIES="$blas" -DZLIB_LIBRARY="$zlib"; \
      cmake --build "/opt/build_$app" --parallel 4; \
      if [ "$app" = eddy ]; then \
        cp "/opt/build_$app/eddy" /opt/tiny_fsl_bin/eddy_cpu; \
        cp "/opt/build_$app/eddy" /opt/tiny_fsl_bin/eddy; \
      else \
        cp "/opt/build_$app/$app" /opt/tiny_fsl_bin/; \
      fi; \
    done; \
    if [ "$CUDA_BUILD" = 1 ]; then \
      command -v nvcc; \
      cmake -S eddy -B /opt/build_eddy_cuda -GNinja \
        -DCMAKE_BUILD_TYPE=Release \
        -DTIPL_DIR=/opt/tiny_fsl/fsl/include \
        -DBLAS_LIBRARIES="$blas" -DZLIB_LIBRARY="$zlib" \
        -DCUDA_ARCH=all; \
      cmake --build /opt/build_eddy_cuda --parallel 4; \
      cp /opt/build_eddy_cuda/eddy /opt/tiny_fsl_bin/eddy_cuda; \
      test -x /opt/tiny_fsl_bin/eddy_cuda; \
      ! ldd /opt/tiny_fsl_bin/eddy_cuda 2>&1 | grep -Eq 'lib(cudart|cublas|cufft|curand|cusolver|cusparse|nvrtc)[^ ]*\\.so'; \
    fi; \
    cp FSL_LICENSE /opt/tiny_fsl_bin/; \
    chmod 755 /opt/tiny_fsl_bin/eddy /opt/tiny_fsl_bin/eddy_cpu /opt/tiny_fsl_bin/topup /opt/tiny_fsl_bin/applytopup; \
    test -x /opt/tiny_fsl_bin/eddy_cpu

FROM scratch
COPY --from=builder /opt/tiny_fsl_bin/ /tiny_fsl/
