FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
#ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/13522/l_openvino_toolkit_fpga_p_2018.3.343.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    cpio \
    sudo \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR && cd $TEMP_DIR
COPY l_openvino_toolkit* .
RUN tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    rm -rf $TEMP_DIR
RUN $INSTALL_DIR/install_dependencies/install_cv_sdk_dependencies.sh
# build Inference Engine samples
RUN mkdir $INSTALL_DIR/deployment_tools/inference_engine/samples/build && cd $INSTALL_DIR/deployment_tools/inference_engine/samples/build && \
    /bin/bash -c "source $INSTALL_DIR/bin/setupvars.sh && cmake .. && make -j1"
COPY intel-opencl*.deb /opt/gfx/
RUN cd /opt/gfx && \
   dpkg -i intel-opencl*.deb && \
   ldconfig && \
   rm -rf /opt/gfx
RUN useradd -G video -ms /bin/bash ali
RUN echo ali:123456 | chpasswd
RUN adduser ali sudo
USER ali
ENV LIBVA_DRIVERS_PATH /opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME iHD
