#!/bin/bash

# Script for setting up environnement for face recon api to launch on

# TODO : Script for installation for mac and windows
# TODO : Make the library work with both python 2 and python 3

apt-get update
apt-get install -y build-essential cmake pkg-config
apt-get install -y libx11-dev libatlas-base-dev
apt-get install -y libgtk-3-dev libboost-python-dev
# TODO : Installation of CUDA Support
apt-get install -y python3-dev python3-pip
# TODO : dlib installation for gpu support

python3 -m pip install -r requirements.txt
