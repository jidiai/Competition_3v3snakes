def single_action(single_action_space):
    player = []
    for j in range(len(single_action_space)):
        each = [0] * single_action_space[j].n
        idx = single_action_space[j].sample()
        each[idx] = 1
        player.append(each)
    return player


def my_controller(observation_list, action_space_list, obs_space_list):
    joint_action = []
    for i in range(len(observation_list)):
        player = single_action(action_space_list[0])
        joint_action.append(player)
    return joint_action
