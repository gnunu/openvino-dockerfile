### Depend

`sudo apt-get install libevent-dev`

### Build

```
mkdir server/build
cd server/build
source /opt/intel/computer_vision_sdk/bin/setupvars.sh
cmake ..
make
```

### Run

```
/server [options] [[fd_model] lm_model reid_model]
        -c gpu_kernel
        -d dev_fd[,dev_lm[,dev_reid]] (default: CPU,CPU,CPU)
        -e expand_ratio (default: 1.15)
        -l cpu_library
        -p port (default: 1080)
        -s widthXheight (default: 600X600)
        -t threshold (default: 0.60)
```

### Test

`wget -nv -O - --post-file=image.jpg http://127.0.0.1:1080/infer`

or:

`curl -s --data-binary @image.jpg http://127.0.0.1:1080/infer`
