# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================
FROM ubuntu:18.04 AS base

WORKDIR /home
ARG OpenVINO_VERSION=2019.3.376

# dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends cpio

COPY l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz .

RUN tar -xvzf l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz && \
    cd l_openvino_toolkit_p_${OpenVINO_VERSION} && \
    sed -i 's#decline#accept#g' silent.cfg && \
    sed -i 's#COMPONENTS=DEFAULTS#COMPONENTS=intel-openvino-ie-rt-vpu-ubuntu-bionic__x86_64;intel-openvino-ie-rt-hddl-ubuntu-bionic__x86_64#g' silent.cfg && \
    ./install.sh -s silent.cfg && \
    cd .. && rm l_openvino_toolkit_p_${OpenVINO_VERSION}.tgz

FROM ubuntu:18.04
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y -q --no-install-recommends libboost-all-dev libusb-1.0-0-dev libjson-c-dev
COPY --from=base /opt/intel /opt/intel
