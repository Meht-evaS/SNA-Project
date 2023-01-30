#!/bin/bash

sudo apt-get update
sudo apt-get -y install python3-pip
python3 -m pip install --upgrade pip
sudo apt-get -y install graphviz graphviz-dev
sudo apt-get -y install python3-tk

pip3 install -r requirements.txt