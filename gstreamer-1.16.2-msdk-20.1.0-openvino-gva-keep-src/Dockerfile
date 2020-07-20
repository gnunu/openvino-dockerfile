# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

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

# Build libva
ARG LIBVA_REPO=https://github.com/intel/libva/releases/download/2.7.0/libva-2.7.0.tar.bz2

RUN wget -O - ${LIBVA_REPO} | tar xj && \
    cd libva-2.7.0 && \
    ./configure --prefix=/usr \
        --libdir=/usr/lib/x86_64-linux-gnu \
        --libexecdir=/usr/lib/x86_64-linux-gnu \
        --enable-shared && \
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

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

# # Build Intel(R) Media Driver
ARG MEDIA_DRIVER_REPO=https://github.com/intel/media-driver/archive/intel-media-20.1.1.tar.gz
RUN wget -O - ${MEDIA_DRIVER_REPO} | tar xz

ARG GMMLIB_REPO=https://github.com/intel/gmmlib/archive/intel-gmmlib-20.1.1.tar.gz
RUN wget -O - ${GMMLIB_REPO} | tar xz && ln -s gmmlib-intel-gmmlib-20.1.1 gmmlib

RUN mkdir media-driver-build && cd media-driver-build && \
    cmake ../media-driver-intel-media-20.1.1 && \
    make -j $(nproc) && \
    make install DESTDIR="/home/build" && \
    make install

ENV LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
ENV LIBVA_DRIVER_NAME=iHD
ENV PKG_CONFIG_PATH=/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib/pkgconfig:${PKG_CONFIG_PATH}

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
     make -j $(nproc) && \
     make install DESTDIR=/home/build && \
     make install

# Build gstreamer-media-SDK (gstreamer plugin mfx)
#COPY ./patches/gstreamer-media-SDK /patches/gstreamer-media-SDK
#RUN git clone https://github.com/intel/gstreamer-media-SDK/ && \
#    cd gstreamer-media-SDK && git apply /patches/gstreamer-media-SDK/mfx.patch && \
#    mkdir build && cd build && cmake .. && make && \
#    make install DESTDIR=/home/build && \
#    make install

RUN apt-get install -y -q --no-install-recommends gtk-doc-tools python-gi-dev

RUN pip3 install setuptools wheel
RUN pip3 install PyGObject numpy

ENV GST_VAAPI_ALL_DRIVERS=1

FROM gst-internal AS openvino-internal

WORKDIR /home
ARG OpenVINO_VERSION=2020.2.120
COPY l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz .

RUN tar -xvzf l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz && \
    cd l_openvino_toolkit_p_${OpenVINO_VERSION} && \
    sed -i 's#decline#accept#g' silent.cfg && \
    sed -i 's#COMPONENTS=DEFAULTS#COMPONENTS=intel-openvino-ie-sdk-ubuntu-bionic__x86_64;intel-openvino-ie-rt-cpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-gpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-vpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-gna-ubuntu-bionic__x86_64;intel-openvino-ie-rt-hddl-ubuntu-bionic__x86_64;intel-openvino-opencv-lib-ubuntu-bionic__x86_64#g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && rm -rf l_openvino_toolkit_p_${OpenVINO_VERSION}

ARG IE_DIR=/home/build/opt/intel/dldt/inference-engine

RUN mkdir -p ${IE_DIR}/include && \
    cp -r /opt/intel/openvino/inference_engine/include/* ${IE_DIR}/include && \
    mkdir -p ${IE_DIR}/lib/intel64 && \
    cp -r /opt/intel/openvino/inference_engine/lib/intel64/* ${IE_DIR}/lib/intel64 && \
    mkdir -p ${IE_DIR}/share && \
    cp -r  /opt/intel/openvino/inference_engine/share/* ${IE_DIR}/share/ && \
    mkdir -p ${IE_DIR}/external/ && \
    cp -r /opt/intel/openvino/inference_engine/external/* ${IE_DIR}/external && \
    mkdir -p ${IE_DIR}/external/opencv && \
    cp -r /opt/intel/openvino/opencv/* ${IE_DIR}/external/opencv/ && \
    mkdir -p ${IE_DIR}/external/ngraph && \
    cp -r /opt/intel/openvino/deployment_tools/ngraph/* ${IE_DIR}/external/ngraph/

RUN echo "\
/usr/local/lib\n\
/usr/lib/x86_64-linux-gnu/gstreamer-1.0\n\
/opt/intel/openvino/inference_engine/lib/intel64/\n\
/opt/intel/openvino/inference_engine/external/tbb/lib\n\
/opt/intel/openvino/inference_engine/external/hddl/lib\n\
/opt/intel/openvino/deployment_tools/ngraph/lib\n\
/opt/intel/openvino/opencv/lib" > /etc/ld.so.conf.d/opencv-dldt-gst.conf && ldconfig

ENV PYTHONPATH /opt/intel/openvino/python/python3.6/openvino:/opt/intel/openvino/python/python3

FROM openvino-internal

WORKDIR /root
ARG GIT_INFO
ARG SOURCE_REV
ARG ENABLE_PAHO_INSTALLATION=false
ARG ENABLE_RDKAFKA_INSTALLATION=false
ARG BUILD_TYPE=Release
ARG EXTERNAL_GVA_BUILD_FLAGS

ENV InferenceEngine_DIR=/opt/intel/openvino/inference_engine/share
ENV OpenCV_DIR=/opt/intel/openvino/opencv/cmake

RUN git clone https://github.com/opencv/gst-video-analytics.git && \
    mkdir -p gst-video-analytics/build \
    && cd gst-video-analytics/build \
    && cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DVERSION_PATCH=${SOURCE_REV} \
    -DGIT_INFO=${GIT_INFO} \
    -DBUILD_SHARED_LIBS=ON \
    -DENABLE_PAHO_INSTALLATION=${ENABLE_PAHO_INSTALLATION} \
    -DENABLE_RDKAFKA_INSTALLATION=${ENABLE_RDKAFKA_INSTALLATION} \
    -DHAVE_VAAPI=ON \
    -DENABLE_VAS_TRACKER=OFF \
    ${EXTERNAL_GVA_BUILD_FLAGS} \
    .. \
    && make -j $(nproc) \
    && make install \
    && echo "/usr/lib/gst-video-analytics" >> /etc/ld.so.conf.d/opencv-dldt-gst.conf && ldconfig

ENV GST_PLUGIN_PATH=/usr/lib/gst-video-analytics/:/root/gst-video-analytics/
ENV PYTHONPATH=/root/gst-video-analytics/python:$PYTHONPATH
