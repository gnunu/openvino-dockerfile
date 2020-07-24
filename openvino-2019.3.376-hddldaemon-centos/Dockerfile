# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================
FROM centos:7.6.1810 AS base

WORKDIR /home
ARG OpenVINO_VERSION=2019.3.376

# dependencies
RUN yum install -y cpio

COPY l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz .

RUN tar -xvzf l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz && \
    cd l_openvino_toolkit_p_${OpenVINO_VERSION} && \
    sed -i 's#decline#accept#g' silent.cfg && \
    sed -i 's#COMPONENTS=DEFAULTS#COMPONENTS=intel-openvino-ie-rt-vpu__x86_64;intel-openvino-ie-rt-hddl__x86_64#g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && rm l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz


FROM centos:7.6.1810

WORKDIR /home

# openvino rt
RUN yum install -y boost-devel libusb-devel perl
COPY --from=base /opt/intel /opt/intel

# for build
# RUN yum -y distro-sync
COPY kernel-devel-3.10.0-957.el7.x86_64.rpm .
RUN rpm -i ./kernel-devel-3.10.0-957.el7.x86_64.rpm
RUN cd /lib/modules && mkdir 3.10.0-957.el7.x86_64 && cd 3.10.0-957.el7.x86_64 && ln -s /usr/src/kernels/3.10.0-957.el7.x86_64 build

RUN yum install -y gcc gcc-c++ make
# needed for buiding myd drivers
RUN yum install -y sudo
RUN source /opt/intel/openvino/bin/setupvars.sh && \
    cd /opt/intel/openvino/inference_engine/external/hddl/drivers/drv_ion && \
    sed -i 's#@modprobe#@echo#g' Makefile && \
    sed -i 's#@modinfo#@echo#g' Makefile && \
    cd /opt/intel/openvino/inference_engine/external/hddl/drivers/drv_vsc && \
    sed -i 's#@modprobe#@echo#g' Makefile && \
    sed -i 's#@modinfo#@echo#g' Makefile && \
    cd /opt/intel/openvino/inference_engine/external/hddl && \
    sed -i 's#lsmod#echo myd_ion myd_vsc#g' install_IVAD_VPU_dependencies.sh && \
    sed -i 's#Please reboot#Have fun!#g' install_IVAD_VPU_dependencies.sh && \
    sed -i 's#setup.sh install$#setup.sh install USE_SYSTEM_CONFIG_HEAP#g' install_IVAD_VPU_dependencies.sh && \
    ./install_IVAD_VPU_dependencies.sh

COPY start_hddl.sh /root/start_hddl.sh
