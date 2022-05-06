# This Dockerfile creates a base dev environment for running grl package code.

# use nvidia/cuda image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# apt install conda dependencies, openspiel dependencies, and general utils
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 \
    git virtualenv clang cmake curl python3 python3-dev python3-pip python3-setuptools python3-wheel python3-tk \
    tmux nano htop

# add optional tmux config not required for running code
# https://github.com/gpakosz/.tmux
RUN cd /root && \
   git clone https://github.com/gpakosz/.tmux.git && \
   ln -s -f .tmux/.tmux.conf && \
   cp .tmux/.tmux.conf.local . && \
   echo "tmux_conf_theme_root_attr='bold'" >> /root/.tmux.conf.local && \
   echo "set -g mouse on" >> /root/.tmux.conf.local

# copy all repo files (except those listed in .dockerignore)
COPY . /grl
WORKDIR /grl

# install grl python package
RUN pip install -r requirements.txt