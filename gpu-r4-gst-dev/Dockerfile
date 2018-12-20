FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
#ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13522/l_openvino_toolkit_fpga_p_2018.3.343.tgz
ARG DOWNLOAD_LINK=http://10.238.135.69:8000/l_openvino_toolkit_fpga_p_2018.4.420.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    mplayer \
    git automake autoconf libtool libgirepository1.0-dev nasm yasm \
    rpm2cpio \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rpm2cpio rpm/intel-gfx_driver-2018.4.420-2018.4-420.noarch.rpm | cpio -id ./opt/intel/computer_vision_sdk_2018.4.420/install_dependencies/intel-opencl*.deb && \
    dpkg -i ./opt/intel/computer_vision_sdk_2018.4.420/install_dependencies/intel-opencl*.deb && \
    ldconfig && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_cv_sdk_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"
COPY libOpenCL.so.1.0.0 /usr/lib/x86_64-linux-gnu/
RUN ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0 && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/x86_64-linux-gnu/libOpenCL.so.1 && \
    ln -s /usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0 /usr/lib/x86_64-linux-gnu/libOpenCL.so
ENV LIBVA_DRIVERS_PATH /opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME iHD
