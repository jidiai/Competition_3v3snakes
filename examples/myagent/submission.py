import torch, os
from .dqn import DQN
from env.chooseenv import make


game = make("snakes_3v3", conf=None)
single_action_space = game.get_single_action_space(0)
action_dim = 4
model = DQN(game.board_width * game.board_height, action_dim, single_action_space)
model_path = os.path.dirname(os.path.abspath(__file__)) + '/dqn_policy_snakes_3v3.pth'
model.eval_net.load_state_dict(torch.load(model_path))


def my_controller(observation_list, action_space_list, obs_space_list):
    joint_action = []
    for i in range(len(action_space_list)):
        player = model.choose_action(observation_list[i])[0].tolist()
        joint_action.append(player)
    return joint_action
