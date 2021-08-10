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
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]
    observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)
    actions = agent.select_action_to_env(observation, indexs.index(o_index-2))
    return actions
