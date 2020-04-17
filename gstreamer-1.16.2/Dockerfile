# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================
ARG dldt=dldt-binaries 
ARG gst=gst-internal
ARG OpenVINO_VERSION

FROM ubuntu:18.04 AS base
WORKDIR /home

# COMMON BUILD TOOLS
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends cmake build-essential automake autoconf libtool make git wget pciutils cpio libtool lsb-release ca-certificates pkg-config bison flex libcurl4-gnutls-dev zlib1g-dev

# python
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libxv1 libxcb-shm0 libxcb-shape0 libxcb-xfixes0 libsdl2-2.0-0
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libnuma1 libass9 libssl1.0.0 libglib2.0 libpango-1.0-0 libpangocairo-1.0-0 gobject-introspection libudev1 libx11-xcb1 libgl1-mesa-glx libxrandr2 libegl1-mesa
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libpng16-16 libvisual-0.4-0 libtheora0 libcdparanoia0 libsoup2.4-1 libjpeg8 libjpeg-turbo8 python3 python3-pip python-yaml python3-cairo-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libgtk-3.0 clinfo libcairo2-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends libusb-1.0-0-dev libboost-all-dev libjson-c-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y -q --no-install-recommends gcovr vim git gdb ca-certificates libssl1.0-dev uuid-dev libgirepository1.0-dev

# Build x264
ARG X264_VER=stable
ARG X264_REPO=https://github.com/mirror/x264

RUN apt-get update && apt-get install -y -q --no-install-recommends nasm yasm

RUN  git clone ${X264_REPO} && \
     cd x264 && \
     git checkout ${X264_VER} && \
     ./configure --prefix="/usr" --libdir=/usr/lib/x86_64-linux-gnu --enable-shared && \
     make -j8 && \
     make install DESTDIR="/home/build" && \
     make install

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libx11-dev xorg-dev libgl1-mesa-dev openbox

# Build Intel(R) Media SDK
ARG MSDK_REPO=https://github.com/Intel-Media-SDK/MediaSDK/releases/download/intel-mediasdk-20.1.0/MediaStack.tar.gz

RUN wget -O - ${MSDK_REPO} | tar xz && \
    cd MediaStack && \
    \
    cp -r opt/ /home/build && \
    cp -r etc/ /home/build && \
    \
    cp -a opt/. /opt/ && \
    cp -a etc/. /opt/ && \
    ldconfig

ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/opt/intel/mediasdk/lib64/pkgconfig
ENV LIBRARY_PATH=/opt/intel/mediasdk/lib64:/usr/lib:${LIBRARY_PATH}
ENV LIBVA_DRIVERS_PATH=/opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1

#clinfo needs to be installed after build directory is copied over
RUN mkdir neo && cd neo && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-gmmlib_20.1.1_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-igc-core_1.0.3698_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-igc-opencl_1.0.3698_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-opencl_20.14.16441_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-ocloc_20.14.16441_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/20.14.16441/intel-level-zero-gpu_0.8.16441_amd64.deb && \
    dpkg -i *.deb && \
    dpkg-deb -x intel-gmmlib_20.1.1_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-core_1.0.3698_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-opencl_1.0.3698_amd64.deb /home/build/ && \
    dpkg-deb -x intel-opencl_20.14.16441_amd64.deb /home/build/ && \
    dpkg-deb -x intel-ocloc_20.14.16441_amd64.deb /home/build/ && \
    dpkg-deb -x intel-level-zero-gpu_0.8.16441_amd64.deb /home/build/ && \
    cp -a /home/build/. /

FROM base AS gst-internal
WORKDIR /home
# Build the gstreamer core

# TODO: If you step up this version to version 'x.y.z', please review gst-plugins-good installation step and remove rtpjitterbuffer patch applying, if patch is contained in gst-plugins-good-'x.y.z'
ARG GST_VER=1.16.2
ARG GST_REPO=https://gstreamer.freedesktop.org/src/gstreamer/gstreamer-${GST_VER}.tar.xz

RUN  DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libglib2.0-dev gobject-introspection libgirepository1.0-dev libpango-1.0-0 libpangocairo-1.0-0 autopoint
RUN  wget -O - ${GST_REPO} | tar xJ && \
     cd gstreamer-${GST_VER} && \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared \
        --enable-introspection \
        --disable-examples  \
        --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install;

# Build the gstreamer plugin bad set
ARG GST_ORC_VER=0.4.29
ARG GST_ORC_REPO=https://gstreamer.freedesktop.org/src/orc/orc-${GST_ORC_VER}.tar.xz

RUN  wget -O - ${GST_ORC_REPO} | tar xJ && \
     cd orc-${GST_ORC_VER} && \
     ./autogen.sh --prefix=/usr --libdir=/usr/lib/x86_64-linux-gnu \
                --libexecdir=/usr/lib/x86_64-linux-gnu \
                --enable-shared \
                --disable-examples  \
                --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

RUN  apt-get update && apt-get install -y -q --no-install-recommends libxrandr-dev libegl1-mesa-dev autopoint bison flex libudev-dev

# Build the gstreamer plugin base
ARG GST_PLUGIN_BASE_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-${GST_VER}.tar.xz

RUN  DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libxv-dev libvisual-0.4-dev libtheora-dev libglib2.0-dev libasound2-dev libcdparanoia-dev libpango1.0-dev

RUN  wget -O - ${GST_PLUGIN_BASE_REPO} | tar xJ && \
     cd gst-plugins-base-${GST_VER} && \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-introspection \
        --enable-shared \
        --disable-examples  \
        --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install


# Build the gstreamer plugin good set

# Lines below extract patch needed for Smart City Sample (OVS use case). Patch is applied before building gst-plugins-good
RUN  mkdir gst-plugins-good-${GST_VER} && \
    git clone https://github.com/GStreamer/gst-plugins-good.git && \
    cd gst-plugins-good && \
    git diff 080eba64de68161026f2b451033d6b455cb92a05 37d22186ffb29a830e8aad2e4d6456484e716fe7 > ../gst-plugins-good-${GST_VER}/rtpjitterbuffer-fix.patch

ARG GST_PLUGIN_GOOD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-${GST_VER}.tar.xz

RUN  apt-get update && apt-get install -y -q --no-install-recommends libsoup2.4-dev libjpeg-dev libssl1.0-dev

RUN  wget -O - ${GST_PLUGIN_GOOD_REPO} | tar xJ && \
     cd gst-plugins-good-${GST_VER} && \
     patch -p1 < rtpjitterbuffer-fix.patch && \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared \
        --disable-examples  \
        --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

# Build librtmp
RUN git clone http://git.ffmpeg.org/rtmpdump && \
    cd rtmpdump && \
    sed -i 's#prefix=/usr/local#prefix=/usr#g' Makefile && \
    make SYS=posix && \
    make install DESTDIR=/home/build && \
    make install

# Build the gstreamer plugin bad set
ARG GST_PLUGIN_BAD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-${GST_VER}.tar.xz

RUN  wget -O - ${GST_PLUGIN_BAD_REPO} | tar xJ && \
     cd gst-plugins-bad-${GST_VER} && \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-rtmp \
        --enable-shared \
        --disable-examples  \
        --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

# Build the gstreamer plugin ugly set
ARG GST_PLUGIN_UGLY_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-ugly/gst-plugins-ugly-${GST_VER}.tar.xz

RUN  wget -O - ${GST_PLUGIN_UGLY_REPO} | tar xJ; \
     cd gst-plugins-ugly-${GST_VER}; \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared \
        --disable-examples  \
        --disable-gtk-doc && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

# Build gst-libav
ARG GST_PLUGIN_LIBAV_REPO=https://gstreamer.freedesktop.org/src/gst-libav/gst-libav-${GST_VER}.tar.xz

RUN wget -O - ${GST_PLUGIN_LIBAV_REPO} | tar xJ && \
    cd gst-libav-${GST_VER} && \
    ./autogen.sh \
        --prefix="/usr" \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared \
        --enable-gpl \
        --disable-gtk-doc && \
    make -j $(nproc) && \
    make install DESTDIR=/home/build && \
    make install

# Build gstreamer plugin vaapi
ARG GST_PLUGIN_VAAPI_REPO=https://gstreamer.freedesktop.org/src/gstreamer-vaapi/gstreamer-vaapi-${GST_VER}.tar.xz

COPY ./patches/gstreamer-vaapi /patches/gstreamer-vaapi

RUN  wget -O - ${GST_PLUGIN_VAAPI_REPO} | tar xJ && \
    cd gstreamer-vaapi-${GST_VER} && git apply /patches/gstreamer-vaapi/vasurface_qdata.patch && \
     ./autogen.sh \
        --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared \
        --disable-examples \
        --disable-gtk-doc  && \
     sed -i 's#define USE_GST_GL_HELPERS 1#undef USE_GST_GL_HELPERS#g' config.h && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

# Build gstreamer-media-SDK (gstreamer plugin mfx)
COPY ./patches/gstreamer-media-SDK /patches/gstreamer-media-SDK
RUN git clone https://github.com/intel/gstreamer-media-SDK/ && \
    cd gstreamer-media-SDK && git apply /patches/gstreamer-media-SDK/mfx.patch && \
    mkdir build && cd build && cmake .. && make && \
    make install DESTDIR=/home/build && \
    make install

RUN apt-get install -y -q --no-install-recommends gtk-doc-tools

RUN pip3 install setuptools wheel
RUN pip3 install PyGObject numpy

ENV LIBVA_DRIVERS_PATH=/opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1

ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/opt/intel/mediasdk/lib64/pkgconfig:${PKG_CONFIG_PATH}
ENV OpenCV_DIR=/opt/intel/dldt/inference-engine/external/opencv/cmake
ENV LIBRARY_PATH=/opt/intel/mediasdk/lib64:/usr/lib:${LIBRARY_PATH}
ENV PATH=/usr/bin:/opt/intel/mediasdk/bin:${PATH}

