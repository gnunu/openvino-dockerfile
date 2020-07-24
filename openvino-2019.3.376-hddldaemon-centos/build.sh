#!/bin/bash
#docker build --build-arg http_proxy=${http_proxy} --build-arg https_proxy=${http_proxy} -t openvino-3.376-hddldaemon .
docker build -t openvino-3.376-hddldaemon-centos .

