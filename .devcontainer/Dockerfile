FROM ubuntu:jammy

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y -qq --no-install-recommends \
  wget \
  git \
  python3-pip \
  libxext-dev \
  libglvnd0 \
  libgl1 \
  libglx0 \
  libegl1 \
  libxext6 \
  libx11-6 \
  libglfw3 \
  libglfw3-dev \
  libglu1-mesa-dev \
  mesa-common-dev \
  freeglut3-dev \
  libglib2.0-0 \
  libsm6 \
  libxrender1 \
  libxext6

RUN pip3 install numpy pygame opencv-python

ENV QT_X11_NO_MITSHM 1
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# Add a non-root user
ARG USERNAME=project3-dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME
