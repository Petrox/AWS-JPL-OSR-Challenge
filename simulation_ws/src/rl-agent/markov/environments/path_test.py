import numpy as np

GUIDE_POINTS = [[0, 0], [-9.2, -3.3], [-15.4, -3.5], [-26.1, -4.3], [-36.2, -2.9], [-44.255, -4.05], [-48, -4.05]]
CHECKPOINT_X = -44.255
CHECKPOINT_Y = -4.05
x = -44.259999  # float(input())
y = -4.09999# float(input())
current_distance_to_checkpoint = np.sqrt(np.square(CHECKPOINT_X - x) + np.square(CHECKPOINT_Y - y))
print(current_distance_to_checkpoint)

if x > CHECKPOINT_X - 0.005 and x <= CHECKPOINT_X + 0.005:
    if y <= CHECKPOINT_Y + 0.05 and y > CHECKPOINT_Y - 0.05:
        print("success!")
line = 0
for i in range(len(GUIDE_POINTS)):
    if x <= GUIDE_POINTS[i][0]:
        line = i
if not line == len(GUIDE_POINTS) - 1:
    # print(line == len(GUIDE_POINTS) - 1)
    num_p1 = (GUIDE_POINTS[line + 1][1] - GUIDE_POINTS[line][1]) * x
    num_p2 = (GUIDE_POINTS[line + 1][0] - GUIDE_POINTS[line][0]) * y
    num_p3 = (GUIDE_POINTS[line + 1][0] * GUIDE_POINTS[line][1]) - (GUIDE_POINTS[line + 1][1] * GUIDE_POINTS[line][0])
    den = np.sqrt(np.square(GUIDE_POINTS[line + 1][1] - GUIDE_POINTS[line][1]) + np.square(GUIDE_POINTS[line + 1][0] - GUIDE_POINTS[line][0]))
    distance_from_path = abs(num_p1 - num_p2 + num_p3)/ den
else:
    distance_from_path = 5
print(distance_from_path)

if current_distance_to_checkpoint < 1:
    distance_reward = ((1 - current_distance_to_checkpoint) ** 5)
else:
    distance_reward = 0
print(distance_reward)