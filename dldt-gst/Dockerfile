FROM ubuntu:16.04 AS build
WORKDIR /home

# COMMON BUILD TOOLS
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends build-essential autoconf make git wget curl pciutils cpio libtool lsb-release ca-certificates pkg-config bison flex libcurl4-gnutls-dev zlib1g-dev automake

# Install cmake
ARG CMAKE_VER=3.13.1
ARG CMAKE_REPO=https://cmake.org/files
RUN wget -O - ${CMAKE_REPO}/v${CMAKE_VER%.*}/cmake-${CMAKE_VER}.tar.gz | tar xz && \
    cd cmake-${CMAKE_VER} && \
    ./bootstrap --prefix="/usr" && \
    make -j8 && \
    make install

# Build NASM
ARG NASM_VER=2.13.03
ARG NASM_REPO=https://www.nasm.us/pub/nasm/releasebuilds/${NASM_VER}/nasm-${NASM_VER}.tar.bz2
RUN  wget ${NASM_REPO} && \
     tar -xaf nasm* && \
     cd nasm-${NASM_VER} && \
     ./autogen.sh && \
     ./configure --prefix="/usr" --libdir=/usr/lib/x86_64-linux-gnu && \
     make -j8 && \
     make install

# Build YASM
ARG YASM_VER=1.3.0
ARG YASM_REPO=https://www.tortall.net/projects/yasm/releases/yasm-${YASM_VER}.tar.gz
RUN  wget -O - ${YASM_REPO} | tar xz && \
     cd yasm-${YASM_VER} && \
     sed -i "s/) ytasm.*/)/" Makefile.in && \
     ./configure --prefix="/usr" --libdir=/usr/lib/x86_64-linux-gnu && \
     make -j8 && \
     make install

# install python3.6
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install python3.6 -y && \
    apt-get install libpython3.6-dev -y
RUN apt-get install wget curl && \
    wget https://bootstrap.pypa.io/ez_setup.py -O - | python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.6
# defaults to python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 && \
    pip3 uninstall setuptools -y && \
    pip3 install --ignore-installed setuptools==39.1.0 -i  https://mirrors.aliyun.com/pypi/simple/

# Build the gstremaer core
ARG GST_VER=1.16.0
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

# Build Oil Runtime Compiler (ORC)
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

# Build the gstremaer plugin base
ARG GST_PLUGIN_BASE_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-base/gst-plugins-base-${GST_VER}.tar.xz

RUN  DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libxv-dev libvisual-0.4-dev libtheora-dev libglib2.0-dev libasound2-dev libcdparanoia-dev libgl1-mesa-dev libpango1.0-dev

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

# Build the gstremaer plugin good set
ARG GST_PLUGIN_GOOD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-good/gst-plugins-good-${GST_VER}.tar.xz

RUN  apt-get update && apt-get install -y -q --no-install-recommends libsoup2.4-dev libjpeg-dev

RUN  wget -O - ${GST_PLUGIN_GOOD_REPO} | tar xJ && \
     cd gst-plugins-good-${GST_VER} && \
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

# Build the gstremaer plugin bad set
ARG GST_PLUGIN_BAD_REPO=https://gstreamer.freedesktop.org/src/gst-plugins-bad/gst-plugins-bad-${GST_VER}.tar.xz

RUN  apt-get update && apt-get install -y -q --no-install-recommends libssl-dev

RUN  wget -O - ${GST_PLUGIN_BAD_REPO} | tar xJ && \
     cd gst-plugins-bad-${GST_VER} && \
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

# Build the gstremaer plugin ugly set
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

# Fetch gmmlib
ARG GMMLIB_VER=intel-gmmlib-19.2.1
ARG GMMLIB_REPO=https://github.com/intel/gmmlib/archive/${GMMLIB_VER}.tar.gz

RUN wget -O - ${GMMLIB_REPO} | tar xz && mv gmmlib-${GMMLIB_VER} gmmlib;

# Build libva
ARG LIBVA_VER=2.5.0
ARG LIBVA_REPO=https://github.com/intel/libva/archive/${LIBVA_VER}.tar.gz

RUN apt-get -y remove libva*

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libdrm-dev libx11-dev xorg-dev libgl1-mesa-dev openbox

RUN wget -O - ${LIBVA_REPO} | tar xz && \
    cd libva-${LIBVA_VER} && \
    ./autogen.sh --prefix=/usr --libdir=/usr/lib/x86_64-linux-gnu && \
    make -j8 && \
    make install DESTDIR=/home/build && \
    make install;

# Build libva-utils
ARG LIBVA_UTILS_VER=2.5.0
ARG LIBVA_UTILS_REPO=https://github.com/intel/libva-utils/archive/${LIBVA_UTILS_VER}.tar.gz

RUN wget -O - ${LIBVA_UTILS_REPO} | tar xz; \
    cd libva-utils-${LIBVA_UTILS_VER}; \
    ./autogen.sh --prefix=/usr --libdir=/usr/lib/x86_64-linux-gnu; \
    make -j8; \
    make install DESTDIR=/home/build; \
    make install;

# Build media driver
ARG MEDIA_DRIVER_VER=intel-media-19.2.0
ARG MEDIA_DRIVER_REPO=https://github.com/intel/media-driver/archive/${MEDIA_DRIVER_VER}.tar.gz

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libdrm-dev libpciaccess-dev libx11-dev xorg-dev libgl1-mesa-dev

RUN wget -O - ${MEDIA_DRIVER_REPO} | tar xz && mv media-driver-${MEDIA_DRIVER_VER} media-driver && \
    mkdir -p media-driver/build && \
    cd media-driver/build && \
    cmake -DBUILD_TYPE=release -DBUILD_ALONG_WITH_CMRTLIB=1 -DMEDIA_VERSION="2.0.0" -DBS_DIR_GMMLIB=/home/gmmlib/Source/GmmLib -DBS_DIR_COMMON=/home/gmmlib/Source/Common -DBS_DIR_INC=/home/gmmlib/Source/inc -DBS_DIR_MEDIA=/home/media-driver -Wno-dev -DCMAKE_INSTALL_PREFIX=/usr .. && \
    make -j8 && \
    make install DESTDIR=/home/build && \
    make install

# Build Media SDK
ARG MSDK_VER=intel-mediasdk-19.2.0
ARG MSDK_REPO=https://github.com/Intel-Media-SDK/MediaSDK/archive/${MSDK_VER}.tar.gz

RUN wget -O - ${MSDK_REPO} | tar xz && mv MediaSDK-${MSDK_VER} MediaSDK && \
    mkdir -p MediaSDK/build && \
    cd MediaSDK/build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_INSTALL_INCLUDEDIR=include/mfx -DBUILD_SAMPLES=OFF -DENABLE_OPENCL=OFF -Wno-dev .. && \
    make -j8 && \
    make install DESTDIR=/home/build && \
    make install;

# Build gst-libav
ARG GST_PLUGIN_LIBAV_REPO=https://gstreamer.freedesktop.org/src/gst-libav/gst-libav-${GST_VER}.tar.xz

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends zlib1g-dev libssl-dev libbz2-dev liblzma-dev

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

# Build gstremaer plugin vaapi
ARG GST_PLUGIN_VAAPI_REPO=https://gstreamer.freedesktop.org/src/gstreamer-vaapi/gstreamer-vaapi-${GST_VER}.tar.xz

RUN  wget -O - ${GST_PLUGIN_VAAPI_REPO} | tar xJ && \
    cd gstreamer-vaapi-${GST_VER} && \
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

#opencl drivers
#clinfo needs to be installed after build directory is copied over
RUN mkdir neo && cd neo && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-gmmlib_18.4.1_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-core_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-igc-opencl_18.50.1270_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-opencl_19.04.12237_amd64.deb && \
    wget https://github.com/intel/compute-runtime/releases/download/19.04.12237/intel-ocloc_19.04.12237_amd64.deb && \
    dpkg -i *.deb && \
    dpkg-deb -x intel-gmmlib_18.4.1_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-core_18.50.1270_amd64.deb /home/build/ && \
    dpkg-deb -x intel-igc-opencl_18.50.1270_amd64.deb /home/build/ && \
    dpkg-deb -x intel-opencl_19.04.12237_amd64.deb /home/build/ && \
    dpkg-deb -x intel-ocloc_19.04.12237_amd64.deb /home/build/

# dldt
ARG DLDT_VER=2019_R1.1
ARG DLDT_REPO=https://github.com/opencv/dldt.git

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install --no-install-recommends -y -q --no-install-recommends libusb-1.0-0-dev libboost-all-dev libgtk2.0-dev python-yaml

RUN git clone -b ${DLDT_VER} ${DLDT_REPO} && \
    cd dldt/inference-engine && \
    git submodule init && \
    git submodule update --recursive && \
    pip3 install -r ie_bridges/python/requirements.txt && \
    mkdir build && \
    cd build && \
    cmake -DENABLE_VALIDATION_SET=OFF -DCMAKE_INSTALL_PREFIX=/opt/intel/dldt -DLIB_INSTALL_PATH=/opt/intel/dldt -DENABLE_SAMPLE_CORE=OFF -DENABLE_PYTHON=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so -DPYTHON_INCLUDE_DIR=/usr/include/python3.6 -DCMAKE_BUILD_TYPE=Release .. && \
    make -j $(nproc) && \
    rm -rf ../bin/intel64/Release/lib/libgtest* && \
    rm -rf ../bin/intel64/Release/lib/libgmock* && \
    rm -rf ../bin/intel64/Release/lib/libmock* && \
    rm -rf ../bin/intel64/Release/lib/libtest*

ARG libdir=/opt/intel/dldt/inference-engine/lib/intel64

RUN mkdir -p build${libdir} && \
    cp -r dldt/inference-engine/bin/intel64/Release/lib/* build${libdir} && \
    mkdir -p build/opt/intel/dldt/inference-engine/external/ && \
    cp -r dldt/inference-engine/temp/* build/opt/intel/dldt/inference-engine/external/


# target image
FROM ubuntu:16.04

# install python3.6
RUN apt-get update && \
    apt-get install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && \
    apt-get install python3.6 -y && \
    apt-get install libpython3.6-dev -y
RUN apt-get install wget curl && \
    wget https://bootstrap.pypa.io/ez_setup.py -O - | python3 && \
    curl https://bootstrap.pypa.io/get-pip.py | python3.6
# defaults to python3.6
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.5 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 2 && \
    pip3 uninstall setuptools -y && \
    pip3 install --ignore-installed setuptools==39.1.0 -i  https://mirrors.aliyun.com/pypi/simple/

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends \
libxv1 libxcb-shm0 libxcb-shape0 libxcb-xfixes0 libsdl2-2.0-0 libasound2 libvdpau1 libass5 libssl1.0.0 \
libdrm-intel1 libx11-xcb1 libgl1-mesa-glx libxrandr2 libegl1-mesa \
libvisual-0.4-0 libtheora0 libcdparanoia0 libsoup2.4-1 \
libnuma1 libglib2.0 libpango-1.0-0 libpangocairo-1.0-0 gobject-introspection libgirepository1.0-dev libudev1 \
libpng12-0 libjpeg8 libjpeg-turbo8 \
libgtk2.0 clinfo libdrm2 \
libusb-1.0-0 libboost-all-dev libjson-c2 python3-yaml python3-gst-1.0 \
ocl-icd-libopencl1 wget vim \
    && rm -rf /var/lib/apt/lists/*

# Install
COPY --from=build /home/build /

RUN echo "\
/usr/local/lib\n\
/usr/lib/x86_64-linux-gnu/gstreamer-1.0\n\
/opt/intel/dldt/inference-engine/lib/intel64/\n\
/opt/intel/dldt/inference-engine/external/opencv_4.1.0_ubuntu16/lib\n\
/opt/intel/dldt/inference-engine/external/tbb/lib\n\
/opt/intel/dldt/inference-engine/external/mkltiny_lnx_20190131/lib\n\
/opt/intel/dldt/inference-engine/external/vpu/hddl/lib" > /etc/ld.so.conf.d/opencv-dldt-gst.conf && ldconfig

RUN pip3 install opencv-python numpy cython

RUN ln -s /usr/lib/python3/dist-packages/gi/_gi.cpython-35m-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/gi/_gi.cpython-36m-x86_64-linux-gnu.so \
  && ln -s /usr/lib/python3/dist-packages/gi/overrides/_gi_gst.cpython-35m-x86_64-linux-gnu.so /usr/lib/python3/dist-packages/gi/overrides/_gi_gst.cpython-36m-x86_64-linux-gnu.so

ENV LIBVA_DRIVERS_PATH=/usr/lib/x86_64-linux-gnu/dri
ENV LIBVA_DRIVER_NAME=iHD
ENV GST_VAAPI_ALL_DRIVERS=1
ENV PYTHONPATH=/opt/intel/dldt/inference-engine/lib/intel64/python_api/python3.6
