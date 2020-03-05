from __future__ import print_function

import time
import boto3
import gym
import numpy as np
from gym import spaces
import PIL
from PIL import Image
import os
import random
import math
import sys
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Pose, Quaternion
from gazebo_msgs.srv import SetModelState, SetModelConfiguration
from gazebo_msgs.msg import ModelState, ContactsState
from sensor_msgs.msg import Image as sensor_image
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Point
from std_msgs.msg import Float64
from std_msgs.msg import String
from PIL import Image
import queue

# logging to ELK
import logging
import logstash
elk_logger = logging.getLogger('python-logstash-logger')
elk_logger.addHandler(logstash.LogstashHandler('localhost', 5959, version=1))

from std_srvs.srv import Empty


VERSION = "0.0.1"
TRAINING_IMAGE_WIDTH = 160
TRAINING_IMAGE_HEIGHT = 120
TRAINING_IMAGE_SIZE = (TRAINING_IMAGE_WIDTH, TRAINING_IMAGE_HEIGHT)

LIDAR_SCAN_MAX_DISTANCE = 4.5  # Max distance Lidar scanner can measure
CRASH_DISTANCE = 0.9  # Min distance to obstacle (The LIDAR is in the center of the 1M Rover)

# Size of the image queue buffer, we want this to be one so that we consume 1 image
# at a time, but may want to change this as we add more algorithms
IMG_QUEUE_BUF_SIZE = 1

# Prevent unknown "stuck" scenarios with a kill switch (MAX_STEPS)
MAX_STEPS = 2000

# Destination Point
CHECKPOINT_X = -44.255
CHECKPOINT_Y = -4.05
CHECKPOINT_PADDING = .5

# Initial position of the robot
INITIAL_POS_X = -0.170505086911
INITIAL_POS_Y = 0.114341186761
INITIAL_POS_Z = -0.0418765865136

INITIAL_ORIENT_X = 0.0135099011407
INITIAL_ORIENT_Y = 0.040927747122
INITIAL_ORIENT_Z = 0.0365547169101
INITIAL_ORIENT_W = 0.998401800258


# Initial distance to checkpoint
INITIAL_DISTANCE_TO_CHECKPOINT = abs(math.sqrt(((CHECKPOINT_X - INITIAL_POS_X) ** 2) +
                                               ((CHECKPOINT_Y - INITIAL_POS_Y) ** 2)))


# SLEEP INTERVALS - a buffer to give Gazebo, RoS and the rl_agent to sync.
SLEEP_AFTER_RESET_TIME_IN_SECOND = 0.3
SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND = 0.3 # LIDAR Scan is 5 FPS (0.2sec).
SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND = 0.01


class MarsEnv(gym.Env):
    def __init__(self):
        self.x = INITIAL_POS_X                                                  # Current position of Rover 
        self.y = INITIAL_POS_Y                                                  # Current position of Rover
        self.last_position_x = INITIAL_POS_X                                    # Previous position of Rover
        self.last_position_y = INITIAL_POS_Y                                    # Previous position of Rover
        #self.orientation = None
        self.aws_region = os.environ.get("AWS_REGION", "us-east-1")             # Region for CloudWatch Metrics
        self.reward_in_episode = 0                                              # Global episodic reward variable
        self.steps = 0                                                          # Global episodic step counter
        self.collision_threshold = sys.maxsize                                  # current collision distance
        self.last_collision_threshold = sys.maxsize                             # previous collision distance
        self.collision = False                                                  # Episodic collision detector
        self.distance_travelled = 0                                             # Global episodic distance counter
        self.current_distance_to_checkpoint = INITIAL_DISTANCE_TO_CHECKPOINT    # current distance to checkpoint
        self.closer_to_checkpoint = False                                       # Was last step closer to checkpoint?
        self.state = None                                                       # Observation space
        self.steering = 0
        self.throttle = 0
        self.power_supply_range = MAX_STEPS                                     # Kill switch (power supply)
        self.episode_count = 0
        self.last_reward = 0
        
        self.distance_travelled_list = [45]

        # Imu Sensor readings
        self.max_lin_accel_x = 0
        self.max_lin_accel_y = 0
        self.max_lin_accel_z = 0
        
        self.reached_waypoint_1 = False
        self.reached_waypoint_2 = False
        self.reached_waypoint_3 = False


        # action space -> steering angle, throttle
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([+1, +3]), dtype=np.float32)


        # Create the observation space
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(TRAINING_IMAGE_SIZE[1], TRAINING_IMAGE_SIZE[0], 3),
                                            dtype=np.uint8)

        self.image_queue = queue.Queue(IMG_QUEUE_BUF_SIZE)

        # ROS initialization
        self.ack_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size=100)

        # ROS Subscriptions
        self.current_position_pub = rospy.Publisher('/current_position', Point, queue_size=3)
        self.distance_travelled_pub = rospy.Publisher('/distance_travelled', String, queue_size=3)
        # ################################################################################

        # Gazebo model state
        self.gazebo_model_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.gazebo_model_configuration_service = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        rospy.init_node('rl_coach', anonymous=True)

        # Subscribe to ROS topics and register callbacks
        rospy.Subscriber('/odom', Odometry, self.callback_pose)
        rospy.Subscriber('/scan', LaserScan, self.callback_scan)
        rospy.Subscriber('/robot_bumper', ContactsState, self.callback_collision)
        rospy.Subscriber('/camera/image_raw', sensor_image, self.callback_image)
        # IMU Sensors
        rospy.Subscriber('/imu/wheel_lb', Imu, self.callback_wheel_lb)



    '''
    DO NOT EDIT - Function called by rl_coach to instruct the agent to take an action
    '''
    def step(self, action):
        # initialize rewards, next_state, done
        self.reward = None
        self.done = False
        self.next_state = None

        steering = float(action[0])
        throttle = float(action[1])
        self.steps += 1
        self.send_action(steering, throttle)
        time.sleep(SLEEP_BETWEEN_ACTION_AND_REWARD_CALCULATION_TIME_IN_SECOND)

        self.call_reward_function(action)

        info = {}  # additional data, not to be used for training

        return self.next_state, self.reward, self.done, info


    '''
    DO NOT EDIT - Function called at the conclusion of each episode to reset episodic values
    '''
    def reset(self):
        print('Total Episodic Reward=%.2f' % self.reward_in_episode,
              'Total Episodic Steps=%.2f' % self.steps)
        # self.send_reward_to_cloudwatch(self.reward_in_episode)

        # Reset global episodic values
        self.reward = None
        self.done = False
        self.next_state = None
        self.ranges= None
        self.send_action(0, 0) # set the throttle to 0
        self.rover_reset()
        self.call_reward_function([0, 0])

        return self.next_state


    '''
    DO NOT EDIT - Function called to send the agent's chosen action to the simulator (Gazebo)
    '''
    def send_action(self, steering, throttle):
        speed = Twist()
        speed.linear.x = throttle
        speed.angular.z = steering
        self.ack_publisher.publish(speed)


    '''
    DO NOT EDIT - Function to reset the rover to the starting point in the world
    '''
    def rover_reset(self):
        
        # Reset Rover-related Episodic variables
        rospy.wait_for_service('gazebo/set_model_state')

        self.x = INITIAL_POS_X
        self.y = INITIAL_POS_Y

        # Put the Rover at the initial position
        model_state = ModelState()
        model_state.pose.position.x = INITIAL_POS_X
        model_state.pose.position.y = INITIAL_POS_Y
        model_state.pose.position.z = INITIAL_POS_Z
        model_state.pose.orientation.x = INITIAL_ORIENT_X
        model_state.pose.orientation.y = INITIAL_ORIENT_Y
        model_state.pose.orientation.z = INITIAL_ORIENT_Z
        model_state.pose.orientation.w = INITIAL_ORIENT_W
        model_state.twist.linear.x = 0
        model_state.twist.linear.y = 0
        model_state.twist.linear.z = 0
        model_state.twist.angular.x = 0
        model_state.twist.angular.y = 0
        model_state.twist.angular.z = 0
        model_state.model_name = 'rover'

        # List of joints to reset (this is all of them)
        joint_names_list = ["rocker_left_corner_lb",
                            "rocker_right_corner_rb",
                            "body_rocker_left",
                            "body_rocker_right",
                            "rocker_right_bogie_right",
                            "rocker_left_bogie_left",
                            "bogie_left_corner_lf",
                            "bogie_right_corner_rf",
                            "corner_lf_wheel_lf",
                            "imu_wheel_lf_joint",
                            "bogie_left_wheel_lm",
                            "imu_wheel_lm_joint",
                            "corner_lb_wheel_lb",
                            "imu_wheel_lb_joint",
                            "corner_rf_wheel_rf",
                            "imu_wheel_rf_joint",
                            "bogie_right_wheel_rm",
                            "imu_wheel_rm_joint",
                            "corner_rb_wheel_rb",
                            "imu_wheel_rb_joint"]
        # Angle to reset joints to
        joint_positions_list = [0 for _ in range(len(joint_names_list))]

        self.gazebo_model_state_service(model_state)
        self.gazebo_model_configuration_service(model_name='rover', urdf_param_name='rover_description', joint_names=joint_names_list, joint_positions=joint_positions_list)

        self.last_collision_threshold = sys.maxsize
        self.last_position_x = self.x
        self.last_position_y = self.y

        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()


        time.sleep(SLEEP_AFTER_RESET_TIME_IN_SECOND)

        self.distance_travelled = 0
        self.current_distance_to_checkpoint = INITIAL_DISTANCE_TO_CHECKPOINT
        self.steps = 0
        self.reward_in_episode = 0
        self.collision = False
        self.closer_to_checkpoint = False
        self.power_supply_range = MAX_STEPS
        self.reached_waypoint_1 = False
        self.reached_waypoint_2 = False
        self.reached_waypoint_3 = False
        self.max_lin_accel_x = 0
        self.max_lin_accel_y = 0
        self.max_lin_accel_z = 0
        
        # First clear the queue so that we set the state to the start image
        _ = self.image_queue.get(block=True, timeout=None)
        self.set_next_state()

    '''
    DO NOT EDIT - Function to find the distance between the rover and nearest object within 4.5M via LIDAR
    '''
    def get_distance_to_object(self):

        while not self.ranges:
            time.sleep(SLEEP_WAITING_FOR_IMAGE_TIME_IN_SECOND)

        size = len(self.ranges)
        x = np.linspace(0, size - 1, 360)
        xp = np.arange(size)
        val = np.clip(np.interp(x, xp, self.ranges), 0, LIDAR_SCAN_MAX_DISTANCE)
        val[np.isnan(val)] = LIDAR_SCAN_MAX_DISTANCE

        # Find min distance
        self.collision_threshold = np.amin(val)


    '''
    DO NOT EDIT - Function to resize the image from the camera and set observation_space
    '''
    def set_next_state(self):
        try:
            # Make sure the first image is the starting image
            image_data = self.image_queue.get(block=True, timeout=None)
        
            # Read the image and resize to get the state
            image = Image.frombytes('RGB', (image_data.width, image_data.height), image_data.data, 'raw', 'RGB', 0, 1)
            image = image.resize((TRAINING_IMAGE_WIDTH,TRAINING_IMAGE_HEIGHT), PIL.Image.ANTIALIAS)
            
            # TODO - can we crop this image to get additional savings?
           
            self.next_state = np.array(image)
        except Exception as err:
            print("Error!::set_next_state:: {}".format(err))


    '''
    DO NOT EDIT - Reward Function buffer
    '''
    def call_reward_function(self, action):
        self.get_distance_to_object() #<-- Also evaluate for sideswipe and collistion damage
        
        # Get the observation
        self.set_next_state()
        
        # reduce power supply range
        self.power_supply_range = MAX_STEPS - self.steps
        
        # calculate reward
        reward, done = self.reward_function()

        # Accumulate reward for the episode
        self.reward_in_episode += reward

        # Get average Imu reading
        if self.max_lin_accel_x > 0 or self.max_lin_accel_y > 0 or self.max_lin_accel_z > 0:
            avg_imu = (self.max_lin_accel_x + self.max_lin_accel_y + self.max_lin_accel_z) / 3
        else:
            avg_imu = 0
    
        print('Step:%.2f' % self.steps,
              'Steering:%.1f' % action[0],
              'Episode:%.1f' % self.episode_count,                # Current episode
              'Tot_R:%.2f' % self.reward_in_episode,              #Reward in episode
              'R:%.2f' % reward,                                  # Reward
              'DTCP:%.3f' % self.current_distance_to_checkpoint,  # Distance to Check Point
              'DT:%.2f' % self.distance_travelled,                # Distance Travelled
              'CT:%.2f' % self.collision_threshold,               # Collision Threshold
              'CTCP:%.1f' % self.closer_to_checkpoint,            # Is closer to checkpoint
              'PSR: %.1f' % self.power_supply_range,              # Steps remaining in Episode
              'IMU: %.3f' % avg_imu,
              'x: %.2f' % self.x,
              'y: %.2f' % self.y)

        try:
            extra = {
                'Step': self.steps,
                'Steering': int(action[0]),
                'Episode': self.episode_count,
                'R': float(reward),
                'Tot_r': self.reward_in_episode,
                'DTCP': self.current_distance_to_checkpoint,
                'DT': self.distance_travelled,
                'CT': self.collision_threshold,
                'CTCP': self.closer_to_checkpoint,
                'PSR': self.power_supply_range,
                'IMU': avg_imu,
                'x': self.x,
                'y': self.y
            }
            elk_logger.info('reward_function', extra=extra)
        except Exception as err:
            print("logging error: {}".format(err))

        self.reward = reward
        self.done = done

        self.last_position_x = self.x
        self.last_position_y = self.y

    def at_destination(self):
        reached_dest_x = (CHECKPOINT_X - CHECKPOINT_PADDING) <= self.x <= (CHECKPOINT_X + CHECKPOINT_PADDING)
        reached_dest_y = (CHECKPOINT_Y - CHECKPOINT_PADDING) <= self.y <= (CHECKPOINT_Y + CHECKPOINT_PADDING)
        return reached_dest_x and reached_dest_y

    '''
    EDIT - but do not change the function signature. 
    Must return a reward value as a float 
    Must return a boolean value indicating if episode is complete
    Must be returned in order of reward, done
    '''
    def reward_function(self):

        '''
        :return: reward as float
                 done as boolean
        '''

        GUIDE_POINTS = [[0, 0], [-9.2, -3.3], [-15.4, -3.5], [-26.1, -4.3], [-36.2, -2.9], [-44.254, -4.05], [-46, -4.05]]

        GUIDERAILS_X_MIN = -50
        GUIDERAILS_X_MAX = 3
        GUIDERAILS_Y_MIN = -7
        GUIDERAILS_Y_MAX = 7

        return_reward = 0

        if self.steps > 0:
            # Has the Rover reached the destination
            if self.x > CHECKPOINT_X - 0.01 and self.x <= CHECKPOINT_X:
                if self.y <= CHECKPOINT_Y and self.y > CHECKPOINT_Y - 0.1:
                    print("Congratulations! The rover has reached the checkpoint!")
                    avg_imu = 0
                    if self.max_lin_accel_x > 0 or self.max_lin_accel_y > 0 or self.max_lin_accel_z > 0:
                        avg_imu = (self.max_lin_accel_x + self.max_lin_accel_y + self.max_lin_accel_z) / 3
                    steps_bias = 0.5
                    dist_bias = 0.5
                    imu_bias = 1
                    # Give a flat termination reward, and make all previous steps irrelevant
                    reward = 1000 - (self.steps / steps_bias) - (self.distance_travelled / dist_bias) - (avg_imu / imu_bias) - self.reward_in_episode
                    print("Final termination reward:", reward, ", score: ", 10000 - self.steps - self.distance_travelled - avg_imu)
                    return reward, True

            line = 0
            distance_from_path = 0
            for i in range(len(GUIDE_POINTS)):
                if self.x <= GUIDE_POINTS[i][0]:
                    line = i
            if not line == len(GUIDE_POINTS) - 1:
                num_p1 = (GUIDE_POINTS[line + 1][1] - GUIDE_POINTS[line][1]) * self.x
                num_p2 = (GUIDE_POINTS[line + 1][0] - GUIDE_POINTS[line][0]) * self.y
                num_p3 = (GUIDE_POINTS[line + 1][0] * GUIDE_POINTS[line][1]) - (GUIDE_POINTS[line + 1][1] * GUIDE_POINTS[line][0])
                den = np.sqrt(np.square(GUIDE_POINTS[line + 1][1] - GUIDE_POINTS[line][1]) + np.square(GUIDE_POINTS[line + 1][0] - GUIDE_POINTS[line][0]))
                distance_from_path = abs(num_p1 - num_p2 + num_p3)/ den
            else:
                distance_from_path = 5
            # If the rover has left the desired path
            off_path_penalty = -0.001
            if distance_from_path > 1:
                print("Rover has left the desired path")
                return_reward = self.reward_in_episode * off_path_penalty

            # If it has not reached the check point is it still on the map?
            out_of_bounds_penalty = -0.001
            if self.x < (GUIDERAILS_X_MIN - .45) or self.x > (GUIDERAILS_X_MAX + .45):
                print("Rover has left the mission map!")
                return_reward = self.reward_in_episode * out_of_bounds_penalty
            if self.y < (GUIDERAILS_Y_MIN - .45) or self.y > (GUIDERAILS_Y_MAX + .45):
                print("Rover has left the mission map!")
                return_reward = self.reward_in_episode * out_of_bounds_penalty

            # Has LIDAR registered a hit
            lidar_crash_penalty = -0.001
            if self.collision_threshold <= CRASH_DISTANCE:
                print("Rover has sustained sideswipe damage")
                return_reward = self.reward_in_episode * lidar_crash_penalty

            # Have the gravity sensors registered too much G-force
            imu_crash_penalty = -0.001
            if self.collision:
                print("Rover has collided with an object")
                return_reward = self.reward_in_episode * imu_crash_penalty
            
            # Has the rover reached the max steps
            power_penalty = -0.001
            if self.power_supply_range < 1:
                print("Rover's power supply has been drained (MAX Steps reached")
                return_reward = self.reward_in_episode * power_penalty

            # Has the rover stopped moving?
            stopped_penalty = -0.001
            self.distance_travelled_list.append(self.distance_travelled)
            if len(self.distance_travelled_list) >= 20:
                if max(self.distance_travelled_list) - min(self.distance_travelled_list) < 0.5:
                    self.distance_travelled_list.pop(0)
                    print("The rover hasn't moved for too long.")
                    return_reward = self.reward_in_episode * stopped_penalty
                else:
                    self.distance_travelled_list.pop(0)
 
            # Return due to episode ending event
            if return_reward != 0:
                self.distance_travelled_list = [45]
                return return_reward, True

            # No Episode ending events - continue to calculate reward
            reward = 0
 
            # dist 50 = 0, dist 25 = 0.25, dist 0 = 1
            if self.current_distance_to_checkpoint < 50:
                distance_reward = ((50 - self.current_distance_to_checkpoint) ** 2) / 2500
            else:
                distance_reward = 0
                
            if self.current_distance_to_checkpoint < 1:
                distance_reward_2 = ((1 - self.current_distance_to_checkpoint) ** 5) * 2
            else:
                distance_reward_2 = 0

            reward += distance_reward
            reward += distance_reward_2
 
            return reward, False
        else:
            self.episode_count += 1
        return 0, False

    '''
    DO NOT EDIT - Function to receive LIDAR data from a ROSTopic
    '''
    def callback_scan(self, data):
        self.ranges = data.ranges


    '''
    DO NOT EDIT - Function to receive image data from the camera RoSTopic
    '''
    def callback_image(self, data):
        try:
             self.image_queue.put_nowait(data)
        except queue.Full:
            pass
        except Exception as ex:
           print("Error! {}".format(ex))


    '''
    DO NOT EDIT - Function to receive IMU data from the Rover wheels
    '''
    def callback_wheel_lb(self, data):
        lin_accel_x = data.linear_acceleration.x
        lin_accel_y = data.linear_acceleration.x
        lin_accel_z = data.linear_acceleration.x

        if lin_accel_x > self.max_lin_accel_x:
            self.max_lin_accel_x = lin_accel_x

        if lin_accel_y > self.max_lin_accel_y:
            self.max_lin_accel_y = lin_accel_y

        if lin_accel_z > self.max_lin_accel_z:
            self.max_lin_accel_z = lin_accel_z



    '''
    DO NOT EDIT - Function to receive Position/Orientation data from a ROSTopic
    '''
    def callback_pose(self, data):
        #self.orientation = data.pose.pose.orientation
        self.linear_trajectory = data.twist.twist.linear
        self.angular_trajectory = data.twist.twist.angular

        new_position = data.pose.pose.position

        p = Point(new_position.x, new_position.y, new_position.z)

        # Publish current position
        self.current_position_pub.publish(p)

        # Calculate total distance travelled
        dist = math.hypot(new_position.x - self.x, new_position.y - self.y)
        self.distance_travelled += dist
        
        # Calculate the distance to checkpoint
        new_distance_to_checkpoint = Float64
        new_distance_to_checkpoint.data = abs(math.sqrt(((new_position.x - CHECKPOINT_X) ** 2) +
                                                        (new_position.y - CHECKPOINT_Y) ** 2))

        if new_distance_to_checkpoint.data < self.current_distance_to_checkpoint:
            self.closer_to_checkpoint = True
        else:
            self.closer_to_checkpoint = False

        # Update the distance to checkpoint
        self.current_distance_to_checkpoint = new_distance_to_checkpoint.data

        # update the current position
        self.x = new_position.x
        self.y = new_position.y

    
    '''
    DO NOT EDIT - Function to receive Collision data from a ROSTopic
    '''
    def callback_collision(self, data):
        # Listen for a collision with anything in the environment
        collsion_states = data.states
        if len(collsion_states) > 0:
            self.collide = True

    
    '''
    DO NOT EDIT - Function to wrote episodic rewards to CloudWatch
    '''
    def send_reward_to_cloudwatch(self, reward):
        try:
            extra = {
                'Episode_Reward': reward,
                'Episode_Steps': self.steps,
                'DistanceToCheckpoint': self.current_distance_to_checkpoint
            }
            elk_logger.info('episodic_rewards', extra=extra)
        except Exception as err:
            print("Error in the send_reward_to_cloudwatch function: {}".format(err))


'''
DO NOT EDIT - Inheritance class to convert discrete actions to continuous actions
'''
class MarsDiscreteEnv(MarsEnv):
    def __init__(self):
        MarsEnv.__init__(self)
        print("New Martian Gym environment created...")
        
        # actions -> straight, left, right
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        # Convert discrete to continuous
        if action == 0:  # turn left
            steering = 1.0
            throttle = 3.00
        elif action == 1:  # turn right
            steering = -1.0
            throttle = 3.00
        elif action == 2:  # straight
            steering = 0
            throttle = 3.00
        else:  # should not be here
            raise ValueError("Invalid action")

        continuous_action = [steering, throttle]

        return super().step(continuous_action)