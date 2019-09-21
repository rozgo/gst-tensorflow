#!/usr/bin/env bash

export GST_PLUGIN_PATH=`pwd`/target/debug

cargo build && \
GST_DEBUG_DUMP_DOT_DIR=dots \
GST_TRACERS="debugserver(port=8080)" \
gst-launch-1.0 \
    filesrc location=assets/sample.webm ! \
    decodebin ! videoconvert ! videoscale ! \
    tf_segmentation ! video/x-raw,format=RGB,width=512,height=288 ! \
    videoconvert ! ximagesink

# gst-debugger-1.0

# queue ! morpheus_rgb camera=fpv ! video/x-raw,format=RGB,width=512,height=288 ! tf_segmentation
# queue ! morpheus_depth camera=bottom ! 
