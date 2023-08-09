#!/bin/bash

apt update
apt install cuda-nvcc-11-7
apt install cuda-toolkit-11-7
pip install -q -U fastapi==0.93.0
pip install -q -U gdown

gdown --fuzzy -O ~/rwkv-2.pth https://drive.google.com/file/d/1dQ_QLPgIb7crfdwwkI9yrfijQVeW3RD1/view?usp=sharing
gdown --fuzzy -O ~/all-es.binidx.tgz https://drive.google.com/file/d/108wQdxHM6CJNvliWlVqOLCPahDYq0E9p/view?usp=sharing
tar xzvf all-es.binidx.tgz

git clone https://github.com/webpolis/musai.git
cd musai
# remove all miditok references
pip install -q -U -r requirements.txt

cd /usr/local/cuda/
@rm bin
@rm include
@rm lib64
ln -s /usr/local/cuda-11.7/* .
