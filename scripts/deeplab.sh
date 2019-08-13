#!/usr/bin/env bash

g++ deeplab.cpp \
    -std=c++11 \
    -I/usr/local/include/tensorflow/c -I/usr/include/ImageMagick-6 -I/usr/include/x86_64-linux-gnu/ImageMagick-6 \
    -ltensorflow -lMagickCore-6.Q16 -lMagickWand-6.Q16 \
    -o deeplab
