import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 1000


class Net(nn.Module):
    def __init__(self, state_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_dim, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(50, output_dim)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, state_dim, output_dim, single_action_space):
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.eval_net, self.target_net = Net(state_dim, output_dim), Net(state_dim, output_dim)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        self.single_action_space = single_action_space[0].n
        self.a_space_dim = len(single_action_space)
        self.a_single_dim = single_action_space[0].n
        self.a_len = self.a_space_dim * self.a_single_dim
        self.memory = np.zeros((MEMORY_CAPACITY, state_dim * 2 + output_dim + 3))
        self.buffer = True
        self.off_policy = False
        self.update_timesteps = 50

    def choose_action(self, x):
        x = torch.reshape(torch.FloatTensor(x), [-1, self.state_dim])
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(0, self.output_dim)
        one_hot_action = [0 for _ in range(self.output_dim)]
        one_hot_action[action] = 1
        one_hot_action = np.array(one_hot_action).reshape(self.single_action_space)
        if self.a_space_dim == 1:
            return np.array([one_hot_action]), np.array([one_hot_action]), action, action
        else:
            index = []
            for i in range(len(one_hot_action)):
                for j in range(len(one_hot_action[i])):
                    if one_hot_action[i][j] == 1:
                        index.append([i, j])
                        break
            index = index[0]
            eye_mat = np.eye(self.a_single_dim)
            multi_dim_action = eye_mat[index]
            return multi_dim_action,  one_hot_action, action, action

    def store_transition(self, state, action, action_logprob, next_state, reward, is_terminal):
        s = np.reshape(state, [1, self.state_dim])
        a = np.reshape(action, [1, self.output_dim])
        r = np.reshape(reward, [1, 1])
        s_ = np.reshape(next_state, [1, self.state_dim])
        a_logprob = np.reshape(action_logprob, [1, 1])
        done = np.reshape(np.array(is_terminal), [1, 1])
        transition = np.hstack((s, a, a_logprob, r, s_, done)).squeeze(0)
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        if self.memory_counter < MEMORY_CAPACITY:
            self.memory_counter += 1

    def learn(self):
        if self.memory_counter < BATCH_SIZE:
            return
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        sample_index = np.random.choice(self.memory_counter, self.learn_step_counter)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.state_dim])
        b_a = torch.LongTensor(b_memory[:, self.state_dim:self.state_dim+self.output_dim])
        b_r = torch.FloatTensor(b_memory[:, self.state_dim + self.output_dim + 1:self.state_dim + self.output_dim + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_dim-1:-1])
        q_eval = torch.gather(self.eval_net(b_s), dim=1, index=torch.argmax(b_a, dim=1).reshape([-1, 1]))
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * torch.max(q_next)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


