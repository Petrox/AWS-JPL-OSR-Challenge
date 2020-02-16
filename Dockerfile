FROM ros:melodic
LABEL author="Jordan Gleeson"

# Install nvidia drivers
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN apt install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev \
    libxi-dev libglu1-mesa libglu1-mesa-dev
RUN apt install -y gcc-6 g++-6

RUN add-apt-repository ppa:graphics-drivers/ppa
RUN DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends nvidia-384 nvidia-384-dev

# CUDA installation
RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
RUN mv cuda_9.0.176_384.81_linux-run cuda_9.0.176_384.81_linux.run
RUN chmod +x cuda_9.0.176_384.81_linux.run
RUN ./cuda_9.0.176_384.81_linux.run --silent --toolkit --override

RUN apt-get update && sudo apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install tensorflow-gpu==1.12.2

RUN touch ~/.bashrc
RUN echo 'export PATH=/usr/local/cuda-9.0/bin:$PATH' >> ~/.bashrc \
  && echo 'export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc \
  && echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc \
  && . ~/.bashrc

# Find CudNN download link by making an account, go to https://developer.nvidia.com/rdp/cudnn-download, start downloading CuDNN Library for Linux in Firefox, right click already started download, Copy Download Link then replacing the link below
RUN wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/9.0_20191031/cudnn-9.0-linux-x64-v7.6.5.32.tgz?eRPoxjc2OUNT3Bp3Tg-cgMhu1rrJ1Oub3NVG3gkSjFR9cUhlji01tVCI98abzPsEHVnqc6szkec2P5-Y8tt1cPJ5BvMrjTSbXDwOlAfs3q8yCyF1Bom6m1Yg3eLT0hXOru1ApF0lUYxVPkhLpzFjzafGLbp859Sx0JlK_Fjymf5OedTmYFzthXlYOO3ysaCXKGDgTAcsC_ISF7AtDyF8EroTowZsjj8 \
  && mv cudnn-9.0-linux-x64-v7.6.5.32* cudnn-9.0-linux-x64-v7.6.5.32.tgz \
  && tar -xzvf /cudnn-9.0-linux-x64-v7.6.5.32.tgz

RUN cp -P cuda/include/cudnn.h /usr/local/cuda-9.0/include
RUN cp -P cuda/lib64/libcudnn* /usr/local/cuda-9.0/lib64/
RUN chmod a+r /usr/local/cuda-9.0/lib64/libcudnn*

# Install bootstrap tools
RUN apt-get update && apt-get install -y dirmngr wget curl mlocate tmux htop

# Install ROS Python tools
RUN apt-get update && apt-get install -y ros-melodic-desktop-full \
        wget git nano python-rosinstall python3-colcon-common-extensions python3-pip

# match robomaker dir structure
RUN mkdir -p /home/ubuntu/catkin_ws/src
# create central dir for mapping data back to the host
RUN mkdir /data
# install minio
RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio
RUN sudo chmod +x minio
RUN sudo mv minio /usr/local/bin
RUN sudo useradd -r minio-user -s /sbin/nologin
RUN sudo chown minio-user:minio-user /usr/local/bin/minio
# install ELK
RUN wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
RUN echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
RUN sudo apt-get update
RUN sudo apt-get install -y default-jdk
RUN sudo apt-get install -y elasticsearch logstash kibana
# install additional useful python packages
RUN pip3 install tornado jupyter
RUN pip3 install elasticsearch python-logstash seaborn jupyter-tensorboard