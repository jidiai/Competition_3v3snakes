import os
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
sys.path.append(str(base_dir))
from algo import RLAgent
from common import *

HIDDEN_SIZE = 256

agent = RLAgent(26, 4, 3)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_2000.pth"
agent.load_model(actor_net)


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 26
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    ctrl_agent_index = [obs['controlled_snake_index']]
    observation = get_observations(obs, ctrl_agent_index, obs_dim, board_width, board_height)
    actions = agent.choose_action(observation)
    return actions
