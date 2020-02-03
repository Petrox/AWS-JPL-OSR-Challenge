FROM dorowu/ubuntu-desktop-lxde-vnc
LABEL author="Joe Barbere"

# installing apt packages
RUN apt-get update && apt-get install -y dirmngr wget curl mlocate tmux htop

# adding keys for ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
RUN wget http://packages.osrfoundation.org/gazebo.key -O - | apt-key add -

# installing ROS
RUN apt-get update && apt-get install -y ros-melodic-desktop-full \
		wget git nano python-rosinstall python3-colcon-common-extensions python3-pip
RUN rosdep init && rosdep update

# installing colcon bundle tools
RUN pip3 install -U setuptools && pip3 install colcon-ros-bundle

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
RUN pip3 install elasticsearch python-logstash jupyter seaborn jupyter-tensorboard