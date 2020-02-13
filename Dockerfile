FROM nvidia/cuda:9.0-devel
LABEL author="Jordan Gleeson"
# # setup timezone
# RUN echo 'Etc/UTC' > /etc/timezone && \
#     ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
#     apt-get update && apt-get install -q -y tzdata && rm -rf /var/lib/apt/lists/*
# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    && rm -rf /var/lib/apt/lists/*
# setup environment
# ENV LANG C.UTF-8
# ENV LC_ALL C.UTF-8
# ENV ROS_DISTRO melodic
# # bootstrap rosdep
# RUN rosdep init && \
#   rosdep update --rosdistro $ROS_DISTRO
# # install ros packages
# RUN apt-get update && apt-get install -y \
#     ros-melodic-ros-core=1.4.1-0* \
#     && rm -rf /var/lib/apt/lists/*
# # installing apt packages
# RUN apt-get update && apt-get install -y dirmngr wget curl mlocate tmux htop
# # adding keys for ROS
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -
# # installing ROS
# RUN apt-get update && apt-get install -y ros-melodic-desktop-full \
#         wget git nano python-rosinstall python3-colcon-common-extensions python3-pip
# #RUN rosdep init && rosdep update
# # installing colcon bundle tools
# RUN pip3 install -U setuptools && pip3 install colcon-ros-bundle
# match robomaker dir structure
RUN mkdir -p /home/ubuntu/catkin_ws/src
# create central dir for mapping data back to the host
RUN mkdir /data
CMD tail -f /dev/null
# # install minio
# RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio
# RUN sudo chmod +x minio
# RUN sudo mv minio /usr/local/bin
# RUN sudo useradd -r minio-user -s /sbin/nologin
# RUN sudo chown minio-user:minio-user /usr/local/bin/minio
# # install ELK
# RUN wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
# RUN echo "deb https://artifacts.elastic.co/packages/6.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-6.x.list
# RUN sudo apt-get update
# RUN sudo apt-get install -y default-jdk
# RUN sudo apt-get install -y elasticsearch logstash kibana
# # install additional useful python packages
# RUN pip3 install elasticsearch python-logstash jupyter seaborn jupyter-tensorboard
# here are some examples of how to install cuda
# the one you should use is dependent on the Nvidia driver version you have installed
# hopefully one will work for you, if not just CPU train like me
# i'm using nvidia driver v440.44, which is too recent I think
# Option 1
# RUN sudo apt install -y nvidia-cuda-toolkit
# Option 2 - you must manually run the installer in the container
#   cd /home/ubuntu
#   sudo sh cuda_10.2.89_440.33.01_linux.run
# RUN wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
# RUN sudo chmod +x cuda_10.2.89_440.33.01_linux.run
# RUN sudo mv cuda_10.2.89_440.33.01_linux.run /home/ubuntu/
# Option 3
# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# RUN sudo apt-get update
