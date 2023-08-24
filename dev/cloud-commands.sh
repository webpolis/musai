#!/bin/bash

apt update && apt upgrade -y
add-apt-repository ppa:deadsnakes/ppa
apt install python3.10
apt install -y cuda-nvcc-11-7
apt install -y cuda-toolkit-11-7
pip install -q -U fastapi==0.93.0
pip install -q -U gdown

gdown --fuzzy -O ~/embvae_466.pth https://drive.google.com/file/d/1kzVEwAJMm6bBiHbSkgjjzpFzRydfxTf4/view?usp=sharing
gdown --fuzzy -O ~/all2.tgz https://drive.google.com/file/d/1wl8z-r0TvszlvFI32CSaJ_QKE4q9v9tI/view?usp=sharing
tar xzvf all2.binidx.tgz

git clone https://github.com/webpolis/musai.git
cd musai
pip install -q -U -r requirements.txt

cd /usr/local/cuda/
@rm bin
@rm include
@rm lib64
ln -s /usr/local/cuda-11.7/* .

python src/tools/trainer.py -v ~/embvae_466.pth -t ~/home/nico/data/ai/models/midi/all2 -o ~/ -e 1024 -n 12 -b 6 -c 1024 -p 1000 -s 1000 -i 1e-5 -q