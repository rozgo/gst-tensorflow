#!/usr/bin/env bash

export GST_PLUGIN_PATH=`pwd`/target/debug

# cargo build && gst-launch-1.0 filesrc location=OYZT7794.MP4 ! decodebin ! videoscale ! video/x-raw,width=513 ! tf_segmentation ! ximagesink

cargo build && gst-launch-1.0 filesrc location=flylow.webm ! decodebin ! videoconvert ! videoscale ! tf_segmentation ! video/x-raw,format=RGB,width=512,height=288 ! videoconvert ! ximagesink

# ffmpeg -ss 00:03:00.0 -i flylowlong.mp4 -c copy -t 00:03:00.0 flylow.mp4