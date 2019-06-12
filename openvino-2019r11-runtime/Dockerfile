FROM ubuntu:16.04
ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTPS_PROXY
#ARG DOWNLOAD_LINK=http://registrationcenter-download.intel.com/akdlm/irc_nas/14919/l_openvino_toolkit_fpga_p_2018.4.420.tgz
ARG DOWNLOAD_LINK=http://10.238.135.69:8000/l_openvino_toolkit_fpga_p_2018.5.445.tgz
ARG INSTALL_DIR=/opt/intel/computer_vision_sdk
ARG TEMP_DIR=/tmp/openvino_installer
RUN apt-get update && apt-get install -y --no-install-recommends \
    rpm2cpio \
    wget \
    cpio \
    sudo \
    lsb-release \
    libnuma1 ocl-icd-libopencl1 \
    libpng12-0 libcairo2 libpango1.0-0 libglib2.0-0 libgtk2.0-0 \
    libswscale-ffmpeg3 libavutil-ffmpeg54 libavcodec-ffmpeg56 libavformat-ffmpeg56 \
    libgstreamer1.0-0 gstreamer1.0-plugins-base libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*
RUN mkdir -p $TEMP_DIR
COPY gen-runtime-openvino-docker.sh $TEMP_DIR/
RUN cd $TEMP_DIR && \
    wget -c $DOWNLOAD_LINK && \
    tar xf l_openvino_toolkit*.tgz && \
    cd l_openvino_toolkit* && \
    sed -i 's/decline/accept/g' silent.cfg && \
    sed -i 's/COMPONENTS=DEFAULTS/COMPONENTS=intel-inference_engine_cpu__noarch;intel-inference_engine_gpu__noarch;intel-inference_engine_vpu__noarch;intel-inference_engine_gna__noarch;intel-inference_engine_dlia__noarch;intel-inference_engine_hddl__noarch;intel-opencv_ubuntu_16_rel__noarch;intel-openvx_ubuntu__noarch/g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    ../gen-runtime-openvino-docker.sh && \
    rm -rf $TEMP_DIR
RUN ldconfig
ENV LIBVA_DRIVERS_PATH /opt/intel/mediasdk/lib64
ENV LIBVA_DRIVER_NAME iHD
