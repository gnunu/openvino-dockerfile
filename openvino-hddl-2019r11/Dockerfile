FROM ubuntu:16.04
ENV http_proxy http://10.239.4.160:913
ENV https_proxy https://10.239.4.160:913
#ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13522/l_openvino_toolkit_fpga_p_2018.3.343.tgz
ARG DOWNLOAD_LINK=http://10.238.135.69:8000/l_openvino_toolkit_fpga_p_2019.1.144.tgz
ARG INSTALL_DIR=/opt/intel/openvino
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    rpm2cpio \
    wget \
    cpio \
    sudo \
    libboost-filesystem1.58.0 libboost-thread1.58.0 \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    ./install_openvino_dependencies.sh && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rpm2cpio rpm/intel-openvino-gfx-driver-2019.1.144-2019.1-144.x86_64.rpm | cpio -id && \
    ./opt/intel/openvino_2019.1.144/install_dependencies/install_NEO_OCL_driver.sh && \
    ldconfig && \
    rm -rf $TEMP_DIR
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"
ENV LIBVA_DRIVERS_PATH /opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME iHD
