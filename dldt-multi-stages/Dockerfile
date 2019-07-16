FROM ubuntu:16.04 AS build
WORKDIR /home

# COMMON BUILD TOOLS
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends build-essential autoconf make git wget curl pciutils cpio libtool lsb-release ca-certificates pkg-config bison flex libcurl4-gnutls-dev zlib1g-dev cmake

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
    sed -i 's/sudo -E //g' ./install_dependencies.sh && \
    ./install_dependencies.sh && \
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
libnuma1 libglib2.0 libpango-1.0-0 libpangocairo-1.0-0 gobject-introspection libudev1 \
libpng12-0 libjpeg8 libjpeg-turbo8 \
libgtk2.0 clinfo libdrm-dev \
libusb-1.0-0-dev libboost-all-dev libjson-c-dev \
ocl-icd-opencl-dev wget vim \
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

RUN pip3 install opencv-python
