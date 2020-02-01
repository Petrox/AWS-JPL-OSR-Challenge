source /opt/ros/melodic/setup.bash
cd /home/ubuntu/catkin_ws/
catkin_make
source /home/ubuntu/catkin_ws/devel/setup.bash

sudo rosdep fix-permissions
rosdep update
rosdep install --from-paths src --ignore-src --rosdistro=melodic -y

cd src/simulation_ws/src
colcon build
cd rl-agent
pip3 install .
cd ..
source install/local_setup.sh

source /home/ubuntu/catkin_ws/src/simulation_ws/src/install/local_setup.bash