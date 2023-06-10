#!/bin/bash
sudo add-apt-repository ppa:jonathonf/ffmpeg-4
sudo apt-get update
sudo apt-get install ffmpeg
# Download and install Miniconda (a lightweight version of Anaconda)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda
rm ~/miniconda.sh

# Add conda to the system PATH
echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Activate the conda base environment
conda init
source ~/.bashrc

conda create --name stg-nv python=3.6 -y
conda activate stg-nv
pip install --user nvidia-pyindex
export PATH=$PATH:$HOME/.local/bin
conda install -c conda-forge openmpi -y
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/anaconda3/envs/stg-nv/lib/
pip install --user nvidia-tensorflow[horovod]
