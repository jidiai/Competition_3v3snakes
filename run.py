from env.chooseenv import make
import torch

def run_game(g, submission=True, model=None):
    print(model)
    while not g.is_terminal():
        print("step%d" % g.step_cnt)
        if hasattr(g, "env_core"):
            g.env_core.render()
            print("action space is ", g.env_core.action_space)
            print("observation space is ", g.env_core.observation_space)
        obs_list = g.get_grid_many_observation(g.current_state, range(g.n_player))
        obs_space_list = g.get_grid_many_obs_space(range(g.n_player))
        if submission:
            joint_act = []
            # TODO: Be advised: please return the following format as p_actions, when submission!
            joint_act.extend(my_controller(obs_list[0:g.agent_nums[0]], g.joint_action_space[0:g.agent_nums[0]], obs_space_list)[:])
            print('++++++++++ joint_act', joint_act)
            joint_act.extend(fake_agent(obs_list[0:g.agent_nums[1]], g.joint_action_space[0:g.agent_nums[0]], obs_space_list)[:])
        else:
            joint_act = my_agent(obs_list, g.joint_action_space, obs_space_list, model, g.is_act_continuous)
        print('joint action', joint_act)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        print("reward is ", reward)
        print("next state is ", next_state)
        if info_before:
            print(info_before)
        if info_after:
            print(info_after)
        print("--------------------------------------------------------")

    print("winner", g.check_win())
    print("winner_information", str(g.won))

def run_game_vector(g, model=None):
    print(model)
    while not g.is_terminal():
        print("step%d" % g.step_cnt)
        if hasattr(g, "env_core"):
            g.env_core.render()
            print("action space is ", g.env_core.action_space)
            print("observation space is ", g.env_core.observation_space)
        obs_list = g.get_vector_many_observation(g.current_state, range(g.n_player))
        print("observation is ", obs_list)
        obs_space_list = g.get_vector_many_obs_space(range(g.n_player))
        joint_act = my_agent(obs_list, g.joint_action_space, obs_space_list, model, g.is_act_continuous)
        print('joint action', joint_act)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        print("reward is ", reward)
        print("next state is ", next_state)
        if info_before:
            print(info_before)
        if info_after:
            print(info_after)
        print("--------------------------------------------------------")

    print("winner", g.check_win())
    print("winner_information", str(g.won))

def render_game(g, model, fps=1):
    import pygame
    pygame.init()
    screen = pygame.display.set_mode(g.grid.size)
    pygame.display.set_caption(g.game_name)
    clock = pygame.time.Clock()

    while not g.is_terminal():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        print("step %d" % g.step_cnt)
        obs_list = g.get_grid_many_observation(g.current_state, range(g.n_player))
        obs_space_list = g.get_grid_many_obs_space(range(g.n_player))

        joint_act = my_agent(obs_list, g.joint_action_space, obs_space_list, model)
        next_state, reward, done, info_before, info_after = g.step(joint_act)
        print(info_before)
        pygame.surfarray.blit_array(screen, g.render_board().transpose(1, 0, 2))
        pygame.display.flip()
        if info_after:
            print(info_after)
        clock.tick(fps)

    print("winner", g.check_win())
    print("winner_information", str(g.won))

def fake_agent(observation_list, action_space_list, obs_space_list, model=None, is_act_continuous=False):
    joint_action = []
    for i in range(len(action_space_list)):
        if not model:
            player = sample(action_space_list[i], is_act_continuous)
        else:
            player = model.choose_action(observation_list[i])[0].tolist()

        joint_action.append(player)
    return joint_action

if __name__ == "__main__":
    env_type = "snakes_3v3"
    model_name = ""
    test_or_train = ""
    render_mode = False
    submission = True

    game = make(env_type, conf=None)
    action_dim = game.action_dim
    input_dimension = game.input_dimension
    print(input_dimension, action_dim)

    model = None
    if render_mode:
        render_game(game, model)
    else:
        if game.is_obs_continuous:
            run_game_vector(game, model)
        else:
            if submission:
                from examples.random.submission import *
                run_game(game, submission=True)
            else:
                run_game(game, model)


