#!/usr/bin/env bash


wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
sudo ldconfig
