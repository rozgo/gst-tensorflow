#!/usr/bin/env bash

export GST_PLUGIN_PATH=`pwd`/target/debug

cargo build && gst-launch-1.0 filesrc location=assets/sample.webm ! decodebin ! videoconvert ! videoscale ! tf_segmentation ! video/x-raw,format=RGB,width=512,height=288 ! videoconvert ! ximagesink
