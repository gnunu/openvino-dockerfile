#!/bin/sh

# this script will install openvino runtime from openvino toolkit

YEAR=2018
VER=5
REV=445
YVR=${YEAR}.${VER}.${REV}
YV_R=${YEAR}.${VER}-${REV}
PKG_VER=${YVR}-${YV_R}
PKG_PREFIX=intel
TARGET=ubuntu

rm -rf /opt/intel/computer_vision_sdk_${YVR}/documentation
rm -rf /opt/intel/computer_vision_sdk_${YVR}/install_dependencies
rm -rf /opt/intel/computer_vision_sdk_${YVR}/openvino_toolkit_fpga_uninstaller

rm -rf /opt/intel/computer_vision_sdk_${YVR}/deployment_tools/inference_engine/include
rm -rf /opt/intel/computer_vision_sdk_${YVR}/deployment_tools/inference_engine/share
rm -rf /opt/intel/computer_vision_sdk_${YVR}/deployment_tools/inference_engine/lib/centos*
rm -rf /opt/intel/computer_vision_sdk_${YVR}/deployment_tools/inference_engine/tools/centos*

# opencv
rm -rf /opt/intel/computer_vision_sdk_${YVR}/opencv/cmake
rm -rf /opt/intel/computer_vision_sdk_${YVR}/opencv/etc
rm -rf /opt/intel/computer_vision_sdk_${YVR}/opencv/include
rm -rf /opt/intel/computer_vision_sdk_${YVR}/opencv/samples

# openvx
rm -rf /opt/intel/computer_vision_sdk_${YVR}/openvx/include
rm -rf /opt/intel/computer_vision_sdk_${YVR}/openvx/samples

# mediasdk
echo -n install mediasdk ...
rpm2cpio rpm/${PKG_PREFIX}-media_stack-${PKG_VER}.noarch.rpm | cpio -id
rm -rf ./opt/intel/mediasdk/doc
rm -rf ./opt/intel/mediasdk/include
rm -rf ./opt/intel/mediasdk/opensource
mv ./opt/intel/mediasdk /opt/intel/
mv ./opt/intel/common /opt/intel/
echo "/opt/intel/mediasdk/lib64" > /etc/ld.so.conf.d/intel-mediasdk.conf
echo "/opt/intel/common/mdf/lib64" > /etc/ld.so.conf.d/intel-mdf.conf
echo OK

# opencl driver
echo -n install Intel opencl driver ...
rpm2cpio rpm/${PKG_PREFIX}-gfx_driver-${PKG_VER}.noarch.rpm | cpio -id ./opt/intel/computer_vision_sdk_${YVR}/install_dependencies/intel-opencl*.deb
dpkg -i ./opt/intel/computer_vision_sdk_${YVR}/install_dependencies/intel-opencl*.deb
echo OK

