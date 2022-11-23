import numpy as np
np.random.seed(42)

starting_position = 1
cliff_position = 0
end_position = 5
goal_reward = 5
cliff_reward = 0

def reward(position) -> int :
    if (position == cliff_position):
        return cliff_reward
    elif (position == end_position):
        return goal_reward
    return 0

def is_terminal(position) -> bool :
    if position == cliff_position:
        return True
    if position == end_position:
        return True
    return False

def strategy() -> int :
    if (np.random.random() >= 0.5):
        return 1
    return -1

left_position_reward_sums = np.zeros(end_position + 1)
left_position_hit_counter = np.zeros(end_position + 1)
right_position_reward_sums = np.zeros(end_position + 1)
right_position_hit_counter = np.zeros(end_position + 1)

n_epochs = 10000

for i in range(n_epochs):
    left_position_log = []
    right_position_log = []
    current_position = starting_position

    while True:
        # position_log.append(current_position)

        if is_terminal(current_position):
            print(f'In terminal position: {current_position}')
            break

        direction = strategy()
        if direction == -1:
            left_position_log.append(current_position)
        else:
            right_position_log.append(current_position)

        current_position += direction

    epoch_reward = reward(current_position)
    for p in left_position_log:
        left_position_hit_counter[p] += 1
        left_position_reward_sums[p] += epoch_reward

    for p in right_position_log:
        right_position_hit_counter[p] += 1
        right_position_reward_sums[p] += epoch_reward

    epoch_left_expected_return = left_position_reward_sums / left_position_hit_counter
    epoch_right_expected_return = right_position_reward_sums / right_position_hit_counter
    print(epoch_left_expected_return)
    print(epoch_right_expected_return)

