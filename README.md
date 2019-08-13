# Tensorflow GStreamer Plugin

Allows for adding ML inferencing to any gstreamer pipeline. As part of the Simbotic family of tools, it's meant to play nice with UnrealEngine gstreamer sources and other feeds, to allow complex real and synthetic pipelines.


## Install Tensorflow

Using one of the following 3 methods.

### Bazel

Tensorflow from bazel

PIP
```
./configure
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow-version-tags.whl
```

C-API
```
bazel build --config opt //tensorflow/tools/lib_package:libtensorflow
```
Find at:
```
bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz
```
Install:
```
tar -C /usr/local -xzf libtensorflow.tar.gz
```

### Script

Install precompilled from script

[scripts/install_tensorflow.sh](scripts/install_tensorflow.sh)

### Rust

Use tensorflow provided by [tensorflow-sys](https://github.com/tensorflow/rust/tree/master/libtensorflow-sys)



