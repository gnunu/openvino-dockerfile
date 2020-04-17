#!/bin/bash
# for OpenVINO 2019r3.1 (3.376) fully installed with gstreamner pipeline and Link Visual
# docker build --build-arg http_proxy=${HTTP_PROXY} --build-arg https_proxy=${HTTPS_PROXY} -t gst-openvino-2019r376-lv-rtmp-msdk19.4.0 .
# docker build --build-arg http_proxy=${HTTP_PROXY} --build-arg https_proxy=${HTTPS_PROXY} --no-cache -t gst-openvino-2019r376-lv-rtmp-msdk19.4.0-nocache . 2>&1 | tee ~/nfs-share/docker-build.log
docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -t gst-openvino-2019r376-lv-rtmp-msdk19.4.0-gva .

