modprobe i2c_dev
modprobe i2c_i801
modprobe myd_vsc
modprobe myd_ion
udevadm control --reload-rules
udevadm trigger
source /opt/intel/openvino/bin/setupvars.sh
/opt/intel/openvino/inference_engine/external/hddl/bin/hddldaemon
