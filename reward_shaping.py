import cv2
import matplotlib
import numpy as numpy
from scipy.spatial.distance import euclidean

GUIDERAILS_X_MIN = -50
GUIDERAILS_X_MAX = 3
GUIDERAILS_Y_MIN = -7
GUIDERAILS_Y_MAX = 7

CHECKPOINT_X = -44.25
CHECKPOINT_Y = -4

def reward_function(x, y, ct):
        '''
        :return: reward as float
                 done as boolean
        '''

        GOAL_THRESHOLD = 0.75
        return_reward = 0

        # Has the Rover reached the destination
        if x >= CHECKPOINT_X - GOAL_THRESHOLD and x <= CHECKPOINT_X + GOAL_THRESHOLD:
            if y >= CHECKPOINT_Y - GOAL_THRESHOLD and y <= CHECKPOINT_Y + GOAL_THRESHOLD:
                reward = 80000
                return_reward = reward

        # If it has not reached the check point is it still on the map?
        if x < (GUIDERAILS_X_MIN - .45) or x > (GUIDERAILS_X_MAX + .45):
            return_reward = -1
        if y < (GUIDERAILS_Y_MIN - .45) or y > (GUIDERAILS_Y_MAX + .45):
            return_reward = -1

        # Has LIDAR registered a hit
        if ct <= 0.9:
            return_reward = -500

        # Return due to episode ending event
        if return_reward != 0:
            return return_reward, True
        

        # No Episode ending events - continue to calculate reward
        reward = 0

        cur_pos = (x, y)
        goal_pos = (-44.25, -4)
        current_distance_to_checkpoint = euclidean(cur_pos, goal_pos)

        # polynomial interpolation {0, 50}, {45, 0}, {12, 20}, {32,3}
        reward += 3 * np.polyval(np.poly1d([-0.00072884291634291634291634,
                                                0.083631588319088319088319088, 
                                                -3.39862567987567987567987567, 
                                                50]), current_distance_to_checkpoint)

        #Stay away from collisions
        if ct < 3.8:
            reward = reward + (-0.25 * (5 - 1.05263 * ct)  # linear fit {{0, 5}, {4.5, 1}

        return reward, False

        return 0, False



fig, ax = plt.subplots()
ax.imshow(data)